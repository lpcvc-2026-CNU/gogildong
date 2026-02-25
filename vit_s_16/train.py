"""Train a ViT-S/16 model with single-teacher distillation (EVA02-E-14-plus).

Hardware target: RTX 4070 (12 GB VRAM).
  - Student  : ViT-S/16, fp32  (~22M params)
  - Teacher  : EVA02-E-14-plus, fp16 (~4.4B params, frozen)
  - No gradient accumulation (add later after checking train time)

Supports json / coco / csv dataset formats defined in dataset_loader.py.
Progress is printed every `log_interval` batches with a live bar.

Usage:
    # From project root:
    python vit_s_16_model/train.py --config vit_s_16_model/config.yaml

    # From model subdirectory:
    python train.py --config config.yaml
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root — works whether launched from root or model subdir
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dataset_loader import ImageTextDataset, build_annotations


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return path
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import open_clip


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
    teacher_name: str
    pretrained_tag: str
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
    log_interval: int = 10

    @staticmethod
    def from_yaml(path: str) -> "Config":
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        data  = cfg["data"]
        model = cfg["model"]
        train = cfg["training"]
        paths = cfg["paths"]
        return Config(
            dataset_type     = str(data.get("dataset_type", "json")),
            image_root       = data["image_root"],
            captions_json    = data.get("captions_json"),
            coco_annotations = data.get("coco_annotations"),
            csv_path         = data.get("csv_path"),
            filter_missing   = bool(data.get("filter_missing", True)),
            val_split        = float(data.get("val_split", 0.05)),
            student_name     = model["student_name"],
            teacher_name     = model["teacher_name"],
            pretrained_tag   = model["pretrained_tag"],
            embed_dim        = model["embed_dim"],
            temperature      = float(model["temperature"]),
            batch_size       = int(train["batch_size"]),
            epochs           = int(train["epochs"]),
            lr               = float(train["lr"]),
            weight_decay     = float(train["weight_decay"]),
            warmup           = int(train["warmup"]),
            use_teacher      = bool(train["use_teacher"]),
            distill_weight   = float(train["distill_weight"]),
            clip_grad_norm   = float(train["clip_grad_norm"]),
            amp              = bool(train["amp"]),
            ema_decay        = float(train["ema_decay"]),
            save_dir         = paths["save_dir"],
            onnx_dir         = paths["onnx_dir"],
            log_interval     = int(train.get("log_interval", 10)),
        )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def contrastive_loss(img: torch.Tensor, txt: torch.Tensor, temperature: float) -> torch.Tensor:
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    logits  = img @ txt.t() / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(student, dim=-1), F.normalize(teacher, dim=-1))


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def _tokenize(tokenizer, captions, device: torch.device, context_length: Optional[int] = None) -> torch.Tensor:
    texts = [captions] if isinstance(captions, str) else list(captions)
    try:
        ids = tokenizer(texts, context_length=context_length) if context_length else tokenizer(texts)
    except TypeError:
        ids = tokenizer(texts)
    except AttributeError as e:
        if "batch_encode_plus" not in str(e):
            raise
        hf = getattr(tokenizer, "tokenizer", None)
        if hf is None:
            raise
        fn = getattr(tokenizer, "clean_fn", None)
        if callable(fn):
            texts = [fn(t) for t in texts]
        max_len = context_length or getattr(tokenizer, "context_length", None) or 77
        enc = hf(texts, return_tensors="pt", max_length=max_len, padding="max_length", truncation=True)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    if ids.dim() == 3:
        ids = ids.squeeze(1)
    return ids.to(device)


def _features(model, images: torch.Tensor, texts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    out = model(images, texts)
    if isinstance(out, dict):
        return out["image_features"], out["text_features"]
    return out[0], out[1]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def fmt_time(s: float) -> str:
    if s < 60:   return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


def gpu_mem() -> str:
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**2
        r = torch.cuda.memory_reserved()  / 1024**2
        return f"{a:.0f}/{r:.0f}MB"
    return "N/A"


def bar(width: int = 80) -> None: print("=" * width, flush=True)
def sep(width: int = 80) -> None: print("-" * width, flush=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate(model, loader: DataLoader, device: torch.device,
             temperature: float, tokenizer) -> Dict[str, float]:
    all_img, all_txt = [], []
    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            texts  = _tokenize(tokenizer, captions, device,
                               context_length=getattr(model, "context_length", None))
            img_f, txt_f = _features(model, images, texts)
            all_img.append(F.normalize(img_f, dim=-1).cpu())
            all_txt.append(F.normalize(txt_f, dim=-1).cpu())

    img_mat = torch.cat(all_img)
    txt_mat = torch.cat(all_txt)
    sim     = img_mat @ txt_mat.t() / temperature
    ranks   = torch.argsort(sim, dim=1, descending=True)
    gt      = torch.arange(ranks.shape[0])

    recalls: Dict[str, float] = {}
    for k, name in [(1, "R@1"), (5, "R@5"), (10, "R@10")]:
        correct = (ranks[:, :k] == gt.unsqueeze(1)).any(dim=1).float()
        recalls[name] = correct.mean().item()
    return recalls


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_root       = _resolve_path(config.image_root)
    captions_json    = _resolve_path(config.captions_json)
    coco_annotations = _resolve_path(config.coco_annotations)
    csv_path         = _resolve_path(config.csv_path)
    save_dir         = _resolve_path(config.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    torch.cuda.empty_cache()

    # ── Banner ──────────────────────────────────────────────────────────────
    bar()
    print("  ViT-S/16  |  LPCVC 2026 Image-Text Retrieval  |  Training", flush=True)
    bar()
    print(f"  Device          : {device}", flush=True)
    print(f"  Student         : {config.student_name}", flush=True)
    if config.use_teacher:
        print(f"  Teacher         : {config.teacher_name}  [{config.pretrained_tag}]", flush=True)
        print(f"  Distill weight  : {config.distill_weight}", flush=True)
    else:
        print("  Teacher         : None (contrastive only)", flush=True)
    print(f"  Batch size      : {config.batch_size}", flush=True)
    print(f"  Epochs          : {config.epochs}", flush=True)
    print(f"  Learning rate   : {config.lr}", flush=True)
    print(f"  Temperature     : {config.temperature}", flush=True)
    print(f"  AMP             : {config.amp}", flush=True)
    print(f"  Save dir        : {save_dir}", flush=True)
    bar()
    print()

    # ── Student ─────────────────────────────────────────────────────────────
    print("Loading student model...", flush=True)
    student, _, preprocess = open_clip.create_model_and_transforms(
        config.student_name, pretrained=None, precision="fp32", force_image_size=224,
    )
    tokenizer = open_clip.get_tokenizer(config.student_name)
    student.to(device).train()

    total_p     = sum(p.numel() for p in student.parameters()) / 1e6
    trainable_p = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6

    # Probe actual student output dim — do NOT use config.embed_dim which may be wrong
    # (e.g. ViT-S/16 outputs 384, but config may say 512)
    student.eval()
    with torch.no_grad():
        _di = torch.zeros(1, 3, 224, 224, device=device)
        _dt = _tokenize(tokenizer, ["a"], device)
        _si, _ = _features(student, _di, _dt)
        s_dim = _si.shape[-1]
        del _di, _dt, _si
    student.train()
    print(f"  ✓ {config.student_name}  —  {total_p:.1f}M total  /  {trainable_p:.1f}M trainable  |  embed_dim={s_dim}", flush=True)

    # ── Teacher (single, frozen, fp16) ──────────────────────────────────────
    teacher     = None
    teacher_tok = None
    proj        = None   # linear projection: student_dim → teacher_dim

    if config.use_teacher:
        print(f"\nLoading teacher: {config.teacher_name}  [{config.pretrained_tag}] ...", flush=True)
        t_model, _, _ = open_clip.create_model_and_transforms(
            config.teacher_name,
            pretrained=config.pretrained_tag,
            precision="fp16",
            force_image_size=224,
        )
        t_model.to(device).eval()
        for p in t_model.parameters():
            p.requires_grad = False
        teacher     = t_model
        teacher_tok = open_clip.get_tokenizer(config.teacher_name)

        # Probe teacher output dim — teacher is fp16, input must match
        _dummy_img_t = torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float16)
        _dummy_txt_t = _tokenize(teacher_tok, ["a"], device)
        with torch.no_grad():
            _t_probe, _ = _features(teacher, _dummy_img_t, _dummy_txt_t)
        t_dim = _t_probe.shape[-1]
        del _dummy_img_t, _dummy_txt_t, _t_probe
        torch.cuda.empty_cache()

        # Projection head uses probed dims — not config values — to avoid shape mismatch
        proj = torch.nn.Linear(s_dim, t_dim).to(device)
        print(f"  ✓ Teacher loaded  |  teacher dim={t_dim}  |  projection {s_dim}→{t_dim}", flush=True)

    # ── Dataset ─────────────────────────────────────────────────────────────
    print("\nLoading dataset...", flush=True)
    annotations = build_annotations(
        config.dataset_type, image_root,
        captions_json=captions_json,
        coco_annotations=coco_annotations,
        csv_path=csv_path,
        filter_missing=config.filter_missing,
    )
    dataset    = ImageTextDataset(image_root, annotations, preprocess, tokenizer)
    val_size   = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"  ✓ Total  : {len(dataset):,} samples", flush=True)
    print(f"     Train : {train_size:,}  ({len(train_loader)} batches)", flush=True)
    print(f"     Val   : {val_size:,}   ({len(val_loader)} batches)", flush=True)

    # ── Optimiser ───────────────────────────────────────────────────────────
    all_params = list(student.parameters())
    if proj is not None:
        all_params += list(proj.parameters())

    optimiser   = torch.optim.AdamW(all_params, lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_steps)
    scaler      = torch.amp.GradScaler("cuda", enabled=config.amp)

    print(f"\n  Total optimiser steps : {total_steps}  ({len(train_loader)} per epoch)", flush=True)
    print()
    bar()
    print("  STARTING TRAINING", flush=True)
    bar()
    print()

    best_r10  = 0.0
    BAR_WIDTH = 30

    # ════════════════════════════════════════════════════════════════════════
    # Epoch loop
    # ════════════════════════════════════════════════════════════════════════
    for epoch in range(1, config.epochs + 1):
        student.train()
        epoch_loss  = 0.0
        epoch_start = time.time()
        batch_times: List[float] = []
        total_batches = len(train_loader)

        for batch_idx, (images, captions) in enumerate(train_loader):
            t0     = time.time()
            images = images.to(device)

            texts_s = _tokenize(tokenizer, captions, device,
                                 context_length=getattr(student, "context_length", None))

            optimiser.zero_grad()

            with torch.amp.autocast("cuda", enabled=config.amp):
                img_emb, txt_emb = _features(student, images, texts_s)
                loss = contrastive_loss(img_emb, txt_emb, config.temperature)

                if teacher is not None:
                    texts_t = _tokenize(teacher_tok, captions, device,
                                        context_length=getattr(teacher, "context_length", None))
                    with torch.no_grad():
                        t_img, t_txt = _features(teacher, images, texts_t)

                    # Project student embeddings to teacher dim
                    # Cast both to float32: s_*_proj may be fp16 under autocast,
                    # and t_img/t_txt are fp16 from the teacher — mse_loss requires matching dtypes
                    s_img_proj = proj(img_emb).float()
                    s_txt_proj = proj(txt_emb).float()
                    loss += config.distill_weight * (
                        distillation_loss(s_img_proj, t_img.float()) +
                        distillation_loss(s_txt_proj, t_txt.float())
                    )

            scaler.scale(loss).backward()
            if config.clip_grad_norm > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(all_params, config.clip_grad_norm)
            # Track scale before step to detect whether optimiser actually ran.
            # GradScaler skips the optimiser when gradients contain inf/nan (common
            # on the first AMP batch), which would trigger the "scheduler before
            # optimizer" warning if scheduler.step() runs unconditionally.
            scale_before = scaler.get_scale()
            scaler.step(optimiser)
            scaler.update()
            # Only advance the LR schedule when the optimiser actually stepped
            if scaler.get_scale() == scale_before:
                scheduler.step()

            epoch_loss += loss.item()
            batch_times.append(time.time() - t0)

            # ── Batch progress line ───────────────────────────────────────
            if (batch_idx + 1) % config.log_interval == 0 or (batch_idx + 1) == total_batches:
                done     = batch_idx + 1
                progress = done / total_batches
                filled   = int(BAR_WIDTH * progress)
                pbar     = "█" * filled + "░" * (BAR_WIDTH - filled)

                recent = batch_times[-config.log_interval:]
                avg_t  = sum(recent) / len(recent)
                eta    = avg_t * (total_batches - done)
                cur_lr = optimiser.param_groups[0]["lr"]

                print(
                    f"  Epoch [{epoch:>2}/{config.epochs}] "
                    f"[{pbar}] {progress*100:5.1f}%  "
                    f"batch {done:>5}/{total_batches}  |  "
                    f"loss {loss.item():7.4f}  |  "
                    f"lr {cur_lr:.2e}  |  "
                    f"{avg_t:.2f}s/batch  |  "
                    f"ETA {fmt_time(eta)}  |  "
                    f"GPU {gpu_mem()}",
                    flush=True,
                )

        # ── Epoch summary ─────────────────────────────────────────────────
        avg_loss   = epoch_loss / total_batches
        epoch_time = time.time() - epoch_start

        print()
        sep()
        print(f"  EPOCH {epoch}/{config.epochs} COMPLETE", flush=True)
        sep()
        print(f"  Avg loss    : {avg_loss:.4f}", flush=True)
        print(f"  Epoch time  : {fmt_time(epoch_time)}", flush=True)
        print(f"  Throughput  : {train_size / epoch_time:.1f} samples/sec", flush=True)
        print()

        # ── Validation ───────────────────────────────────────────────────
        print("  Running validation...", flush=True)
        student.eval()
        val_start = time.time()
        recalls   = evaluate(student, val_loader, device, config.temperature, tokenizer)
        val_time  = time.time() - val_start
        r10       = recalls["R@10"]

        print(f"  Validation  (took {fmt_time(val_time)}):", flush=True)
        print(f"    R@1  : {recalls['R@1']:.4f}  ({recalls['R@1']*100:.2f}%)", flush=True)
        print(f"    R@5  : {recalls['R@5']:.4f}  ({recalls['R@5']*100:.2f}%)", flush=True)
        print(f"    R@10 : {recalls['R@10']:.4f}  ({recalls['R@10']*100:.2f}%)", flush=True)

        if r10 > best_r10:
            improvement = r10 - best_r10
            best_r10    = r10
            torch.save({"model": student.state_dict(), "config": config.__dict__},
                       os.path.join(save_dir, "best.pt"))
            print(f"  ✓ New best model saved!  R@10 = {best_r10:.4f}  (+{improvement:.4f})", flush=True)
        else:
            print(f"  Best R@10 so far: {best_r10:.4f}  (no improvement)", flush=True)

        bar()
        print()
        student.train()

    # ── Done ─────────────────────────────────────────────────────────────────
    bar()
    print("  TRAINING COMPLETE", flush=True)
    bar()
    print(f"  Best R@10 : {best_r10:.4f}", flush=True)
    print(f"  Checkpoint: {save_dir}/best.pt", flush=True)
    bar()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT-S/16 for LPCVC 2026 image-text retrieval")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute() and not (Path.cwd() / config_path).exists():
        config_path = Path(__file__).resolve().parent / config_path

    cfg = Config.from_yaml(str(config_path.resolve()))
    main(cfg)