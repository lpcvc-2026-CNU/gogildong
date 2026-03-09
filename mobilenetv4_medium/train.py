"""
Main training script for LPCVC 2026 Track 1.

Usage:
    python train.py                                          # 전체 3단계
    python train.py --stage 1                               # Stage 1만
    python train.py --stage 2 --resume checkpoints/stage1_epoch010.pt
    python train.py --stage 3 --resume checkpoints/stage2_epoch030.pt
    python train.py --config my_config.yaml --stage all    # 커스텀 설정 파일
"""

import argparse
import logging
import os
import random

import numpy as np
import torch

from utils.config import load_config, save_config
from models.student import build_student_model
from models.teacher import TeacherManager
from data.dataset import CLIPTextTokenizer, build_dataloader
from training.trainer import Stage1Trainer, Stage2Trainer, Stage3Trainer
from utils.metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="LPCVC 2026 Track 1 Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="YAML 설정 파일 경로")
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--resume", type=str, default=None, help="이어서 학습할 체크포인트 경로")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_teachers", action="store_true", help="스승 모델 없이 실행 (디버깅용)")
    parser.add_argument("--max_samples", type=int, default=None, help="데이터셋 샘플 수 제한 (디버깅용)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    set_seed(cfg.training.seed)
    device = args.device
    logger.info(f"디바이스: {device}  |  설정 파일: {args.config}")

    # 학습 재현을 위해 사용된 config 복사 저장
    os.makedirs(cfg.training.output_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.training.output_dir, "used_config.yaml"))

    # --- 토크나이저 ---
    tokenizer = CLIPTextTokenizer(cfg)

    # --- DataLoader ---
    use_teachers = not args.no_teachers
    train_loader = build_dataloader(
        data_path=cfg.data.train_path,
        tokenizer=tokenizer,
        cfg=cfg,
        is_train=True,
        use_teacher_transforms=use_teachers,
        max_samples=args.max_samples,
    )

    val_loader = None
    if os.path.exists(cfg.data.val_path):
        val_loader = build_dataloader(
            data_path=cfg.data.val_path,
            tokenizer=tokenizer,
            cfg=cfg,
            is_train=False,
            use_teacher_transforms=False,
        )

    # --- 학생 모델 ---
    logger.info("학생 모델 구축 중...")
    model = build_student_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"학생 모델 파라미터 수: {total_params:,}")

    # --- 스승 모델 ---
    teacher_manager = None
    if use_teachers:
        logger.info("스승 모델 로드 중 (시간이 걸릴 수 있습니다)...")
        try:
            teacher_manager = TeacherManager(
                cfg=cfg,
                device=device,
                load_siglip=(args.stage in ["all", "2", "3"]),
                load_dfn=True,
            )
        except Exception as e:
            logger.warning(f"스승 모델 로드 실패: {e}")
            logger.warning("스승 없이 학습을 진행합니다.")

    # --- 단계별 학습 ---
    stages_to_run = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    resume_path   = args.resume

    for stage_num in stages_to_run:
        logger.info(f"\n{'='*60}\n  Stage {stage_num} 시작\n{'='*60}\n")

        if stage_num == 1:
            trainer = Stage1Trainer(model, teacher_manager, cfg, device)
            trainer.run(train_loader, resume_from=resume_path)
            resume_path = str(trainer.output_dir / f"stage1_epoch{cfg.training.stage1.epochs:03d}.pt")

        elif stage_num == 2:
            trainer = Stage2Trainer(model, teacher_manager, cfg, device)
            trainer.run(train_loader, resume_from=resume_path)
            resume_path = str(trainer.output_dir / f"stage2_epoch{cfg.training.stage2.epochs:03d}.pt")

        elif stage_num == 3:
            trainer = Stage3Trainer(model, teacher_manager, cfg, device)
            trainer.run(train_loader, resume_from=resume_path)

        # 각 단계 후 검증
        if val_loader is not None:
            logger.info(f"[Stage {stage_num}] 검증 데이터 평가 중...")
            metrics = evaluate_model(model, val_loader, device=device)
            logger.info(f"[Stage {stage_num}] 검증 결과: {metrics}")

    logger.info("\n학습 완료!")


if __name__ == "__main__":
    main()
