"""
Evaluation script for LPCVC 2026 Track 1.

Usage:
    python evaluate.py --checkpoint checkpoints/stage3_epoch005.pt --val_data data/val
"""

import argparse
import logging
import torch

from utils.config import get_default_config
from models.student import build_student_model
from data.dataset import CLIPTextTokenizer, build_dataloader
from utils.metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LPCVC 2026 Track 1 model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_default_config()

    # Load model
    model = build_student_model(cfg.model).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # Build dataloader
    tokenizer = CLIPTextTokenizer(cfg.model.tokenizer_name)
    val_loader = build_dataloader(
        data_path=args.val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=4,
        is_train=False,
        use_teacher_transforms=False,
    )

    # Evaluate
    metrics = evaluate_model(model, val_loader, device=args.device)

    print("\n" + "="*50)
    print("  LPCVC 2026 Track 1 — Evaluation Results")
    print("="*50)
    for k, v in sorted(metrics.items()):
        print(f"  {k:<20}: {v:.4f}")
    print("="*50)
    print(f"  Primary metric (mean R@10): {metrics.get('mean_R@10', 0):.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
