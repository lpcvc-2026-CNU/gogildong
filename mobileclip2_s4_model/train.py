"""Train a MobileCLIP 2 S4 model with optional dual‑teacher distillation.

This script extends the SigLIP training example to support multiple
teachers.  It loads a MobileCLIP2‑S4 student from `open_clip`, loads
teacher models specified in `config.yaml`, and trains the student on
a dataset of images and captions using contrastive and distillation
losses.

The code is a simplified reference implementation; add data
augmentation, distributed training and other techniques as needed.
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

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
    teacher_names: List[str]
    pretrained_tags: list
    embed_dim: int
    temperature: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup: int
    use_teacher: bool
    distill_weights: List[float]
    clip_grad_norm: float
    amp: bool
    ema_decay: float
    accumulation_steps: int
    save_dir: str
    onnx_dir: str
    log_interval: int = 10  # Log every N batches

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
            teacher_names=model.get('teacher_names', []),
            pretrained_tags=model.get('pretrained_tags', []),
            embed_dim=model['embed_dim'],
            temperature=float(model['temperature']),
            batch_size=int(train['batch_size']),
            epochs=int(train['epochs']),
            lr=float(train['lr']),
            weight_decay=float(train['weight_decay']),
            warmup=int(train['warmup']),
            use_teacher=bool(train['use_teacher']),
            distill_weights=[float(w) for w in train.get('distill_weights', [])],
            clip_grad_norm=float(train['clip_grad_norm']),
            amp=bool(train['amp']),
            ema_decay=float(train['ema_decay']),
            accumulation_steps=int(train.get('accumulation_steps', 1)),
            save_dir=paths['save_dir'],
            onnx_dir=paths['onnx_dir'],
            log_interval=int(train.get('log_interval', 10)),
        )


def contrastive_loss(image_feats: torch.Tensor, text_feats: torch.Tensor, temperature: float) -> torch.Tensor:
    image_feats = F.normalize(image_feats, dim=-1)
    text_feats = F.normalize(text_feats, dim=-1)
    logits = image_feats @ text_feats.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return (loss_i + loss_t) / 2


def distillation_loss(student_feats: torch.Tensor, teacher_feats: torch.Tensor) -> torch.Tensor:
    student_feats = F.normalize(student_feats, dim=-1)
    teacher_feats = F.normalize(teacher_feats, dim=-1)
    return F.mse_loss(student_feats, teacher_feats)


def _tokenize_texts(tokenizer, captions, device: torch.device, context_length: Optional[int] = None) -> torch.Tensor:
    """Tokenize captions with fallback for HF tokenizers lacking `batch_encode_plus`."""
    texts = [captions] if isinstance(captions, str) else list(captions)
    try:
        if context_length is not None:
            try:
                token_ids = tokenizer(texts, context_length=context_length)
            except TypeError:
                token_ids = tokenizer(texts)
        else:
            token_ids = tokenizer(texts)
    except AttributeError as err:
        if 'batch_encode_plus' not in str(err):
            raise
        hf_tokenizer = getattr(tokenizer, 'tokenizer', None)
        if hf_tokenizer is None:
            raise

        clean_fn = getattr(tokenizer, 'clean_fn', None)
        if callable(clean_fn):
            texts = [clean_fn(t) for t in texts]

        max_len = context_length or getattr(tokenizer, 'context_length', None) or 77
        encoded = hf_tokenizer(
            texts,
            return_tensors='pt',
            max_length=max_len,
            padding='max_length',
            truncation=True,
        )
        token_ids = encoded['input_ids'] if isinstance(encoded, dict) else encoded.input_ids

        if getattr(tokenizer, 'strip_sep_token', False):
            sep_token_id = getattr(hf_tokenizer, 'sep_token_id', None)
            if sep_token_id is not None:
                token_ids = torch.where(token_ids == sep_token_id, torch.zeros_like(token_ids), token_ids)

    if token_ids.dim() == 3:
        token_ids = token_ids.squeeze(1)
    return token_ids.to(device)


def _model_features(model, images: torch.Tensor, texts) -> tuple:
    """Return (image_features, text_features) from model; handles dict or tuple output (open_clip)."""
    out = model(images, texts)
    if isinstance(out, dict):
        return out["image_features"], out["text_features"]
    return out[0], out[1]


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_gpu_memory() -> str:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return f"{allocated:.0f}/{reserved:.0f}MB"
    return "N/A"


def main(config: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Resolve paths relative to project root (works whether run from root or model subdir)
    image_root = _resolve_path(config.image_root)
    captions_json = _resolve_path(config.captions_json)
    coco_annotations = _resolve_path(config.coco_annotations)
    csv_path = _resolve_path(config.csv_path)
    save_dir = _resolve_path(config.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    torch.cuda.empty_cache()
    
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Student Model: {config.student_name}")
    print(f"Teachers: {config.teacher_names if config.use_teacher else 'None'}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Accumulation Steps: {config.accumulation_steps}")
    print(f"Effective Batch Size: {config.batch_size * config.accumulation_steps}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.lr}")
    print(f"AMP Enabled: {config.amp}")
    print(f"Save Directory: {save_dir}")
    print("=" * 80)
    print()
    
    # Load student
    # 224로 통일해 teacher(ViT 224)와 호환 (student 기본 256이면 충돌 방지)
    print("Loading student model...")
    student_model, _, preprocess = open_clip.create_model_and_transforms(
        config.student_name, pretrained='dfndr2b', precision='fp32', force_image_size=224
    )
    tokenizer = open_clip.get_tokenizer(config.student_name)
    student_model.to(device)
    student_model.train()
    print(f"✓ Student model loaded: {config.student_name}")
    
    # Load teachers + 각 teacher용 tokenizer (context_length 64/77 등 호환)
    teacher_models = []
    teacher_tokenizers = []
    if config.use_teacher:
        print("\nLoading teacher models...")
        for i, name in enumerate(config.teacher_names):
            tag = config.pretrained_tags[i]
            print(f"  [{i+1}/{len(config.teacher_names)}] Loading {name} ({tag})...")
            kwargs = {
                'model_name': name,
                'pretrained': tag,
                'precision': 'fp16',
                'force_image_size': 224,
            }
            if 'ViT-L-14' in name and tag == 'dfn2b':
                print(f"      --> Applying force_quick_gelu=True for {name}")
                kwargs['force_quick_gelu'] = True
            model, _, _ = open_clip.create_model_and_transforms(**kwargs)
            model.to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            teacher_models.append(model)
            teacher_tokenizers.append(open_clip.get_tokenizer(name))
            print(f"      ✓ Teacher {i+1} loaded")
        assert len(config.distill_weights) == len(teacher_models), "Number of distill_weights must match teacher_names"
        print(f"✓ All {len(teacher_models)} teacher models loaded")
    torch.cuda.empty_cache()

    # Teacher 1 (SigLIP2: 1152)용 프로젝션
    proj_t1 = torch.nn.Linear(768, 1152).to(device)

    # Teacher 2 (ViT-L-14)
    proj_t2 = torch.nn.Linear(768, 768).to(device)

    # Dataset (json / coco / csv). filter_missing=True 시 존재하는 이미지 쌍만 사용.
    print("\nLoading dataset...")
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
    print(f"✓ Dataset loaded: {len(dataset)} total samples")
    print(f"  Train: {train_size} samples ({len(train_loader)} batches)")
    print(f"  Val: {val_size} samples ({len(val_loader)} batches)")
    
    # Optimiser
    params = list(student_model.parameters()) + list(proj_t1.parameters()) + list(proj_t2.parameters())
    optimiser = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    accum = config.accumulation_steps
    steps_per_epoch = (len(train_loader) + accum - 1) // accum  # optimizer steps per epoch
    total_steps = config.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=config.amp)
    
    print(f"\nOptimizer steps per epoch: {steps_per_epoch}")
    print(f"Total optimizer steps: {total_steps}")
    print()
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()
    
    best_r10 = 0.0
    
    for epoch in range(1, config.epochs + 1):
        student_model.train()
        epoch_loss = 0.0
        epoch_start_time = time.time()
        batch_times = []

        # 에포크 시작 시 한 번 초기화
        optimiser.zero_grad()

        for batch_idx, (images, captions) in enumerate(train_loader):
            batch_start_time = time.time()
            
            images = images.to(device)
            # 원문 캡션 → 각 모델의 tokenizer로 토큰화 (context_length 64/77 등 호환)
            texts_student = _tokenize_texts(tokenizer, captions, device, context_length=getattr(student_model, 'context_length', None))
            with torch.amp.autocast('cuda', enabled=config.amp):
                img_embeds, txt_embeds = _model_features(student_model, images, texts_student)
                loss = contrastive_loss(img_embeds, txt_embeds, config.temperature)
                if teacher_models:
                    # 루프에서 i(인덱스)를 추가하여 어떤 스승인지 식별합니다.
                    for i, (weight, teacher, tok) in enumerate(zip(config.distill_weights, teacher_models, teacher_tokenizers)):
                        
                        # 1. 각 스승에 맞는 텍스트 토크나이징 및 특징 추출
                        texts_t = _tokenize_texts(tok, captions, device, context_length=getattr(teacher, 'context_length', None))
                        with torch.no_grad():
                            t_img, t_txt = _model_features(teacher, images, texts_t)
                        
                        # 2. 학생의 임베딩(768)을 스승의 차원에 맞게 투영(Projection)
                        # i=0: SigLIP2 (1152), i=1: ViT-L-14 (768)
                        if i == 0:
                            # 첫 번째 스승용 투영 레이어 적용 (768 -> 1152)
                            s_img_proj = proj_t1(img_embeds)
                            s_txt_proj = proj_t1(txt_embeds)
                        else:
                            # 두 번째 스승용 투영 레이어 적용 (768 -> 768)
                            # 사실 차원이 같으면 레이어가 없어도 되지만, 
                            # 별도의 학습 가능한 레이어를 두는 것이 증류 성능에 더 유리할 때가 많습니다.
                            s_img_proj = proj_t2(img_embeds)
                            s_txt_proj = proj_t2(txt_embeds)
                            
                        # 3. 투영된 벡터와 스승의 벡터로 Loss 계산
                        loss += weight * (distillation_loss(s_img_proj, t_img) + distillation_loss(s_txt_proj, t_txt))

                # Gradient Accumulation 적용
                loss = loss / accum
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader):
                if config.clip_grad_norm > 0:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(params, config.clip_grad_norm)
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                optimiser.zero_grad()
                
            epoch_loss += loss.item() * accum
            
            # Batch timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Batch-level logging
            if (batch_idx + 1) % config.log_interval == 0 or (batch_idx + 1) == len(train_loader):
                avg_batch_time = sum(batch_times[-config.log_interval:]) / len(batch_times[-config.log_interval:])
                batches_remaining = len(train_loader) - (batch_idx + 1)
                eta = avg_batch_time * batches_remaining
                current_lr = optimiser.param_groups[0]['lr']
                progress = (batch_idx + 1) / len(train_loader) * 100
                gpu_mem = get_gpu_memory()
                
                print(f"Epoch [{epoch}/{config.epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] ({progress:.1f}%) | "
                      f"Loss: {loss.item() * accum:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {avg_batch_time:.2f}s/batch | "
                      f"ETA: {format_time(eta)} | "
                      f"GPU: {gpu_mem}")
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        print()
        print("-" * 80)
        print(f"EPOCH {epoch} SUMMARY")
        print("-" * 80)
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Epoch Time: {format_time(epoch_time)}")
        print(f"Samples/sec: {len(train_dataset) / epoch_time:.1f}")
        print()
        
        # Evaluation
        print("Running validation...")
        student_model.eval()
        eval_start = time.time()
        recalls = evaluate(student_model, val_loader, device, config.temperature, tokenizer)
        eval_time = time.time() - eval_start
        r10 = recalls['R@10']
        
        print(f"Validation Results (took {format_time(eval_time)}):")
        print(f"  R@1:  {recalls['R@1']:.3f}")
        print(f"  R@5:  {recalls['R@5']:.3f}")
        print(f"  R@10: {recalls['R@10']:.3f}")
        
        if r10 > best_r10:
            improvement = r10 - best_r10
            best_r10 = r10
            save_path = os.path.join(save_dir, 'best.pt')
            torch.save({'model': student_model.state_dict(), 'config': config.__dict__}, save_path)
            print(f"✓ New best model saved! (R@10: {best_r10:.3f}, +{improvement:.3f})")
        else:
            print(f"  Best R@10: {best_r10:.3f} (no improvement)")
        
        print("=" * 80)
        print()
        
        student_model.train()
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best R@10: {best_r10:.3f}")
    print(f"Model saved to: {save_dir}")
    print("=" * 80)


def evaluate(model, loader: DataLoader, device: torch.device, temperature: float, tokenizer) -> Dict[str, float]:
    all_img = []
    all_txt = []
    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            texts = _tokenize_texts(tokenizer, captions, device, context_length=getattr(model, 'context_length', None))
            img_feats, txt_feats = _model_features(model, images, texts)
            img_feats = F.normalize(img_feats, dim=-1)
            txt_feats = F.normalize(txt_feats, dim=-1)
            all_img.append(img_feats.cpu())
            all_txt.append(txt_feats.cpu())
    image_matrix = torch.cat(all_img, dim=0)
    text_matrix = torch.cat(all_txt, dim=0)
    sim = image_matrix @ text_matrix.t() / temperature
    ranks = torch.argsort(sim, dim=1, descending=True)
    gt = torch.arange(ranks.shape[0])
    recalls = {}
    for k, name in [(1, 'R@1'), (5, 'R@5'), (10, 'R@10')]:
        correct = (ranks[:, :k] == gt.unsqueeze(1)).any(dim=1).float()
        recalls[name] = correct.mean().item()
    return recalls


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileCLIP2‑S4 model')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not (Path.cwd() / config_path).exists():
        config_path = Path(__file__).resolve().parent / config_path
    cfg = Config.from_yaml(str(config_path.resolve()))
    main(cfg)