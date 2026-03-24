"""
Evaluation script for LPCVC 2026 Track 1.

Usage:
    python evaluate.py --checkpoint checkpoints/stage3_epoch005.pt
"""

import argparse
import logging
import torch

from utils.config import load_config        # ← get_default_config() 제거
from models.student import build_student_model
from data.dataset import StudentTokenizer, build_dataloader
from utils.metrics import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LPCVC 2026 Track 1 model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--val_data",   type=str, default=None,
                        help="검증 데이터 경로 (미지정 시 config.data.val_path 사용)")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # 모델 로드
    model = build_student_model(cfg).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # DataLoader
    tokenizer = StudentTokenizer(cfg)
    val_path  = args.val_data or cfg.data.val_path
    val_loader = build_dataloader(
        data_path=val_path,
        tokenizer=tokenizer,
        cfg=cfg,
        is_train=False,
        use_teacher_transforms=False,
    )

    # 평가
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
