"""
Main training script for LPCVC 2026 Track 1 (Single Teacher: SigLIP2).

변경 이력
─────────
- Teacher 임베딩 사전 캐싱 흐름 도입
  1. cache_loader (shuffle=False, teacher_transform=True) 로 SigLIP2 임베딩 계산
  2. teacher_manager.offload() → GPU 에서 SigLIP2 해제
  3. train_loader (shuffle=True, teacher_transform=False) 로 학생 학습
  4. 학습 루프에서 batch["sample_idx"] 로 캐시 슬라이싱

Usage:
    python train.py                                          # 전체 3단계
    python train.py --stage 1                               # Stage 1만
    python train.py --stage 2 --resume checkpoints/stage1_epoch010.pt
    python train.py --stage 3 --resume checkpoints/stage2_epoch030.pt
    python train.py --no_teacher                            # 캐시 없이 L_CLIP 만 (디버깅)
    python train.py --rebuild_cache                         # 캐시 파일 무시하고 재계산
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from config import load_config, save_config
from student import build_student_model
from teacher import TeacherManager
from dataset import StudentTokenizer, build_dataloader
from trainer import Stage1Trainer, Stage2Trainer, Stage3Trainer
from metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="LPCVC 2026 Track 1 Training")
    parser.add_argument("--config",        type=str, default="config.yaml")
    parser.add_argument("--stage",         type=str, default="all",
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--resume",        type=str, default=None,
                        help="이어서 학습할 체크포인트 경로")
    parser.add_argument("--device",        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_teacher",    action="store_true",
                        help="teacher/캐시 없이 L_CLIP 만 학습 (디버깅용)")
    parser.add_argument("--max_samples",   type=int, default=None,
                        help="데이터셋 샘플 수 제한 (디버깅용)")
    parser.add_argument("--cache_path",    type=str, default="siglip2_cache.pt",
                        help="임베딩 캐시 저장 경로 (기본: siglip2_cache.pt)")
    parser.add_argument("--rebuild_cache", action="store_true",
                        help="기존 캐시를 무시하고 재계산")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    set_seed(cfg.training.seed)
    device = args.device
    logger.info(f"디바이스: {device}  |  설정 파일: {args.config}")

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.training.output_dir, "used_config.yaml"))

    # ── 토크나이저 ──────────────────────────────────────────────────────────
    tokenizer = StudentTokenizer(cfg)

    # ── 학생 모델 ───────────────────────────────────────────────────────────
    logger.info("학생 모델 구축 중...")
    model = build_student_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  └─ 학습 가능 파라미터: {total_params:,}")

    # ── Teacher 임베딩 캐싱 ─────────────────────────────────────────────────
    embedding_cache = None

    if not args.no_teacher:
        cache_path = Path(args.cache_path)

        # 캐시 재계산 플래그
        if args.rebuild_cache and cache_path.exists():
            cache_path.unlink()
            logger.info(f"[Cache] 기존 캐시 삭제: {cache_path}")

        if cache_path.exists():
            # ── 캐시 존재: 바로 로드 ────────────────────────────────────────
            logger.info(f"[Cache] 캐시 파일 발견 → 로드만 수행 (teacher 로드 불필요)")
            embedding_cache = torch.load(cache_path, map_location="cpu")
            n   = embedding_cache["img"].shape[0]
            dim = embedding_cache["img"].shape[1]
            logger.info(f"  └─ {n:,}개 샘플 | dim={dim}")

        else:
            # ── 캐시 미존재: teacher 로드 후 계산 ──────────────────────────
            logger.info("SigLIP2 teacher 로드 중 (캐시 빌드용)...")
            teacher_manager = TeacherManager(cfg=cfg, device=device)

            # 캐시 빌드 전용 DataLoader
            # - shuffle=False  : sample_idx 와 텐서 위치 일치 보장
            # - use_teacher_transforms=True : SigLIP2 384×384 이미지 변환 포함
            # - is_train=False : eval transform (student 측, 캐싱 시엔 중요하지 않음)
            logger.info("캐시 빌드용 DataLoader 생성 (shuffle=False, teacher_transform=True)...")
            cache_loader = build_dataloader(
                data_path=cfg.data.train_path,
                tokenizer=tokenizer,
                cfg=cfg,
                is_train=False,               # student transform 은 eval 버전 사용
                use_teacher_transforms=True,  # SigLIP2 teacher 이미지 변환 포함
                max_samples=args.max_samples,
                shuffle=False,               # ← 반드시 False (인덱스 순서 보장)
            )

            # 사전 계산 실행
            embedding_cache = teacher_manager.build_embedding_cache(
                cache_dataloader=cache_loader,
                cache_path=str(cache_path),
                fp16=True,
            )

            # Teacher GPU 메모리 해제 (이후 불필요)
            teacher_manager.offload()
            del teacher_manager, cache_loader
            torch.cuda.empty_cache()
            logger.info("SigLIP2 teacher GPU 오프로드 완료 — 학습 시 GPU 여유 증가")

    # ── 학습용 DataLoader ───────────────────────────────────────────────────
    # teacher transform 불필요 → DataLoader 처리 부담 감소 / 속도 향상
    logger.info("학습용 DataLoader 생성 (shuffle=True, teacher_transform=False)...")
    train_loader = build_dataloader(
        data_path=cfg.data.train_path,
        tokenizer=tokenizer,
        cfg=cfg,
        is_train=True,
        use_teacher_transforms=False,  # 캐시 빌드 완료 후 불필요
        max_samples=args.max_samples,
        shuffle=True,
    )

    # ── 검증 DataLoader ─────────────────────────────────────────────────────
    val_loader = None
    if os.path.exists(cfg.data.val_path):
        val_loader = build_dataloader(
            data_path=cfg.data.val_path,
            tokenizer=tokenizer,
            cfg=cfg,
            is_train=False,
            use_teacher_transforms=False,
        )

    # ── 단계별 학습 ─────────────────────────────────────────────────────────
    stages_to_run = [1, 2, 3] if args.stage == "all" else [int(args.stage)]
    resume_path   = args.resume

    for stage_num in stages_to_run:
        logger.info(f"\n{'='*60}\n  Stage {stage_num} 시작\n{'='*60}\n")

        # StageTrainer 는 teacher_manager 대신 embedding_cache 를 받음
        trainer_kwargs = dict(
            model=model,
            embedding_cache=embedding_cache,   # None 이면 L_CLIP 만 학습
            cfg=cfg,
            device=device,
        )

        if stage_num == 1:
            trainer = Stage1Trainer(**trainer_kwargs)
            trainer.run(train_loader, resume_from=resume_path)
            resume_path = str(
                trainer.output_dir / f"stage1_epoch{cfg.training.stage1.epochs:03d}.pt"
            )

        elif stage_num == 2:
            trainer = Stage2Trainer(**trainer_kwargs)
            trainer.run(train_loader, resume_from=resume_path)
            resume_path = str(
                trainer.output_dir / f"stage2_epoch{cfg.training.stage2.epochs:03d}.pt"
            )

        elif stage_num == 3:
            trainer = Stage3Trainer(**trainer_kwargs)
            trainer.run(train_loader, resume_from=resume_path)

        # 각 단계 후 검증
        if val_loader is not None:
            logger.info(f"[Stage {stage_num}] 검증 데이터 평가 중...")
            metrics = evaluate_model(model, val_loader, device=device)
            logger.info(f"[Stage {stage_num}] 검증 결과: {metrics}")

    logger.info("\n학습 완료!")


if __name__ == "__main__":
    main()
