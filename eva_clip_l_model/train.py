"""Train an EVA‑02‑CLIP‑L model for image–text retrieval.

This script fine‑tunes a pre‑trained EVA‑02‑CLIP‑L student and
optionally distils it from a larger EVA or CLIP teacher.  It uses
open‑clip for model loading and implements a basic contrastive +
distillation training loop.

Note: due to the size of the student, adjust the batch size, learning
rate, and accumulation to suit your GPU hardware.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dataset_loader import ImageTextDataset, build_annotations


def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve config path relative to project root so it works from any cwd."""
    if not path:
        return path
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image

import open_clip


@dataclass
class Config:
    dataset_type: str
    image_root: str
    captions_json: Optional[str]
    coco_annotations: Optional[str]
    csv_path: Optional[str]
    filter_missing: bool
    val_split: float
    student_name: str
    teacher_name: Optional[str]
    embed_dim: int
    temperature: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup: int
    use_teacher: bool
    distill_weight: float
    clip_grad_norm: float
    amp: bool
    ema_decay: float
    save_dir: str
    onnx_dir: str

    @staticmethod
    def from_yaml(path: str) -> 'Config':
        import yaml  # type: ignore
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        data = cfg['data']
        model = cfg['model']
        train = cfg['training']
        paths = cfg['paths']
        return Config(
            dataset_type=str(data.get('dataset_type', 'json')),
            image_root=data['image_root'],
            captions_json=data.get('captions_json'),
            coco_annotations=data.get('coco_annotations'),
            csv_path=data.get('csv_path'),
            filter_missing=bool(data.get('filter_missing', True)),
            val_split=float(data.get('val_split', 0.05)),
            student_name=model['student_name'],
            teacher_name=model.get('teacher_name'),
            embed_dim=model['embed_dim'],
            temperature=float(model['temperature']),
            batch_size=int(train['batch_size']),
            epochs=int(train['epochs']),
            lr=float(train['lr']),
            weight_decay=float(train['weight_decay']),
            warmup=int(train['warmup']),
            use_teacher=bool(train['use_teacher']),
            distill_weight=float(train['distill_weight']),
            clip_grad_norm=float(train['clip_grad_norm']),
            amp=bool(train['amp']),
            ema_decay=float(train['ema_decay']),
            save_dir=paths['save_dir'],
            onnx_dir=paths['onnx_dir'],
        )


def contrastive_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor, temperature: float) -> torch.Tensor:
    img_feats = F.normalize(img_feats, dim=-1)
    txt_feats = F.normalize(txt_feats, dim=-1)
    logits = img_feats @ txt_feats.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) / 2


def distillation_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    student_feats = F.normalize(student_feats, dim=-1)
    teacher_feats = F.normalize(teacher_feats, dim=-1)
    return F.mse_loss(student_feats, teacher_feats)


def main(config: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Resolve paths relative to project root (works whether run from root or model subdir)
    image_root = _resolve_path(config.image_root)
    captions_json = _resolve_path(config.captions_json)
    coco_annotations = _resolve_path(config.coco_annotations)
    csv_path = _resolve_path(config.csv_path)
    save_dir = _resolve_path(config.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # Load student
    student_model, _, preprocess = open_clip.create_model_and_transforms(config.student_name, pretrained=None, precision='fp32')
    tokenizer = open_clip.get_tokenizer(config.student_name)
    student_model.to(device)
    student_model.train()
    # Load teacher
    teacher_model = None
    if config.use_teacher and config.teacher_name:
        teacher_model, _, _ = open_clip.create_model_and_transforms(config.teacher_name, pretrained=None, precision='fp32')
        teacher_model.to(device)
        teacher_model.eval()
    # Dataset (json / coco / csv). filter_missing=True 시 존재하는 이미지 쌍만 사용.
    annotations = build_annotations(
        config.dataset_type,
        image_root,
        captions_json=captions_json,
        coco_annotations=coco_annotations,
        csv_path=csv_path,
        filter_missing=config.filter_missing,
    )
    dataset = ImageTextDataset(image_root, annotations, preprocess, tokenizer)
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # Optimiser
    optimiser = torch.optim.AdamW(student_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
    best_r10 = 0.0
    for epoch in range(1, config.epochs + 1):
        student_model.train()
        total_loss = 0.0
        for images, captions in train_loader:
            images = images.to(device)
            texts = tokenizer(captions).to(device)
            if texts.dim() == 3:
                texts = texts.squeeze(1)
            optimiser.zero_grad()
            with torch.amp.autocast('cuda', enabled=config.amp):
                s_img, s_txt = student_model(images, texts)
                loss = contrastive_loss(s_img, s_txt, config.temperature)
                if teacher_model is not None:
                    with torch.no_grad():
                        t_img, t_txt = teacher_model(images, texts)
                    loss += config.distill_weight * (distillation_loss(s_img, t_img) + distillation_loss(s_txt, t_txt))
            scaler.scale(loss).backward()
            if config.clip_grad_norm > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.clip_grad_norm)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        student_model.eval()
        metrics = evaluate(student_model, val_loader, device, config.temperature, tokenizer)
        r10 = metrics['R@10']
        print(f"Epoch {epoch} | Loss {avg_loss:.4f} | R@1 {metrics['R@1']:.3f} | R@5 {metrics['R@5']:.3f} | R@10 {metrics['R@10']:.3f}")
        if r10 > best_r10:
            best_r10 = r10
            torch.save({'model': student_model.state_dict(), 'config': config.__dict__}, os.path.join(save_dir, 'best.pt'))
        student_model.train()


def evaluate(model, loader: DataLoader, device: torch.device, temperature: float, tokenizer) -> Dict[str, float]:
    img_list = []
    txt_list = []
    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            texts = tokenizer(captions).to(device)
            if texts.dim() == 3:
                texts = texts.squeeze(1)
            s_img, s_txt = model(images, texts)
            img_list.append(F.normalize(s_img, dim=-1).cpu())
            txt_list.append(F.normalize(s_txt, dim=-1).cpu())
    img_matrix = torch.cat(img_list, dim=0)
    txt_matrix = torch.cat(txt_list, dim=0)
    sim = img_matrix @ txt_matrix.t() / temperature
    ranks = torch.argsort(sim, dim=1, descending=True)
    gt = torch.arange(ranks.shape[0])
    recalls = {}
    for k, name in [(1, 'R@1'), (5, 'R@5'), (10, 'R@10')]:
        correct = (ranks[:, :k] == gt.unsqueeze(1)).any(dim=1).float()
        recalls[name] = correct.mean().item()
    return recalls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EVA‑02‑CLIP‑L model')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not (Path.cwd() / config_path).exists():
        config_path = Path(__file__).resolve().parent / config_path
    cfg = Config.from_yaml(str(config_path.resolve()))
    main(cfg)
