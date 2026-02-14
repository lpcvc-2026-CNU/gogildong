"""Train a SigLIP 2 Base model for image–text retrieval.

This script uses the `open_clip` library to load a SigLIP 2 Base student and
optionally a larger SigLIP 2 teacher.  It trains the student on a
dataset of (image, caption) pairs using a mixture of contrastive and
distillation losses.

It is intentionally lightweight and omits features like distributed
training and advanced data augmentation.  You should extend it as
needed for competition use.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import open_clip


@dataclass
class Config:
    image_root: str
    captions_json: str
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
            image_root=data['image_root'],
            captions_json=data['captions_json'],
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


class ImageTextDataset(Dataset):
    def __init__(self, image_root: str, json_path: str, transform: transforms.Compose, tokenizer):
        self.image_root = Path(image_root)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_path = self.image_root / item['image']
        caption = item['caption']
        image = self.transform(Image.open(image_path).convert('RGB'))
        text_inputs = self.tokenizer([caption])
        return image, text_inputs


def contrastive_loss(image_feats: torch.Tensor, text_feats: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute the NT-Xent contrastive loss for a batch of image/text features."""
    # Normalise embeddings
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)
    # Similarity matrix (batch_size x batch_size)
    logits = image_feats @ text_feats.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) / 2


def distillation_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    """Mean squared error between student and teacher embeddings."""
    student_feats = F.normalize(student_feats, dim=-1)
    teacher_feats = F.normalize(teacher_feats, dim=-1)
    return F.mse_loss(student_feats, teacher_feats)


def main(config: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.save_dir, exist_ok=True)
    # Create models
    student_model, _, preprocess = open_clip.create_model_and_transforms(
        config.student_name, pretrained=None, precision='fp32')
    tokenizer = open_clip.get_tokenizer(config.student_name)
    student_model.to(device)
    student_model.train()
    teacher_model = None
    if config.use_teacher and config.teacher_name:
        teacher_model, _, _ = open_clip.create_model_and_transforms(
            config.teacher_name, pretrained=None, precision='fp32')
        teacher_model.to(device)
        teacher_model.eval()

    # Load dataset
    dataset = ImageTextDataset(config.image_root, config.captions_json, preprocess, tokenizer)
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Optimiser and scheduler
    optimiser = torch.optim.AdamW(student_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    # Exponential moving average of student weights
    ema_state = None
    best_recall10 = 0.0

    for epoch in range(1, config.epochs + 1):
        student_model.train()
        epoch_loss = 0.0
        for images, texts in train_loader:
            images = images.to(device)
            texts = {k: v.to(device) for k, v in texts.items()}
            optimiser.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.amp):
                image_embeds, text_embeds = student_model(images, texts)
                loss_c = contrastive_loss(image_embeds, text_embeds, config.temperature)
                loss = loss_c
                if teacher_model is not None:
                    with torch.no_grad():
                        t_image_embeds, t_text_embeds = teacher_model(images, texts)
                    loss_d = distillation_loss(image_embeds, t_image_embeds) + distillation_loss(text_embeds, t_text_embeds)
                    loss += config.distill_weight * loss_d
            scaler.scale(loss).backward()
            if config.clip_grad_norm > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), config.clip_grad_norm)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        # Validation
        student_model.eval()
        recalls = evaluate(student_model, val_loader, device, config.temperature)
        recall10 = recalls['R@10']
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Recall@1: {recalls['R@1']:.3f} R@5: {recalls['R@5']:.3f} R@10: {recalls['R@10']:.3f}")
        if recall10 > best_recall10:
            best_recall10 = recall10
            torch.save({'model': student_model.state_dict(), 'config': config.__dict__}, os.path.join(config.save_dir, 'best.pt'))
        student_model.train()


def evaluate(model, loader: DataLoader, device: torch.device, temperature: float) -> Dict[str, float]:
    """Compute recall@1/5/10 for a validation set."""
    # Collect all embeddings
    all_image_embeds = []
    all_text_embeds = []
    with torch.no_grad():
        for images, texts in loader:
            images = images.to(device)
            texts = {k: v.to(device) for k, v in texts.items()}
            img_feats, txt_feats = model(images, texts)
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = F.normalize(txt_feats, dim=-1)
            all_image_embeds.append(img_feats.cpu())
            all_text_embeds.append(txt_feats.cpu())
    image_matrix = torch.cat(all_image_embeds, dim=0)
    text_matrix = torch.cat(all_text_embeds, dim=0)
    # Similarity matrix
    sim = image_matrix @ text_matrix.t() / temperature
    # compute recall
    ranks = torch.argsort(sim, dim=1, descending=True)
    gt = torch.arange(ranks.shape[0])
    recalls = {}
    for k, name in [(1, 'R@1'), (5, 'R@5'), (10, 'R@10')]:
        correct = (ranks[:, :k] == gt.unsqueeze(1)).any(dim=1).float()
        recalls[name] = correct.mean().item()
    return recalls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SigLIP 2 model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    main(cfg)
