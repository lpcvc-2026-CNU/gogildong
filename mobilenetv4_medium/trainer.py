"""
Stage-aware trainer for LPCVC 2026 Track 1 (Single Teacher: SigLIP2).

변경 이력
─────────
- teacher_manager 실시간 호출 제거
- embedding_cache (dict: {"img": Tensor, "txt": Tensor}) 기반으로 교체
- train_one_step: batch["sample_idx"] 로 캐시 슬라이싱

토크나이저 흐름:
  - batch["student_input_ids"] / ["student_attention_mask"] → DistilBERT (학생)
  - embedding_cache["img"][indices], embedding_cache["txt"][indices] → SigLIP2 임베딩 (캐시)
"""

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from student import StudentCLIP
from loss import TotalLoss
from config import ConfigNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer & LR 스케줄러
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(params, lr=lr)


def build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Stage 2 동적 λ 스케줄
# ---------------------------------------------------------------------------

def get_stage2_lambdas(epoch: int, total_epochs: int, s_cfg: ConfigNode):
    """
    λ1 고정, λ2↓ λ3↑ 선형 보간 후 합=1 정규화.
    Returns: (lambda1, lambda2, lambda3)
    """
    t  = epoch / max(1, total_epochs - 1)
    l2 = s_cfg.lambda2_start + t * (s_cfg.lambda2_end - s_cfg.lambda2_start)
    l3 = s_cfg.lambda3_start + t * (s_cfg.lambda3_end - s_cfg.lambda3_start)
    total = s_cfg.lambda1 + l2 + l3
    return s_cfg.lambda1 / total, l2 / total, l3 / total


# ---------------------------------------------------------------------------
# 단일 스텝 학습
# ---------------------------------------------------------------------------

def train_one_step(
    model: StudentCLIP,
    batch: dict,
    embedding_cache: Optional[dict],   # {"img": Tensor(N,D), "txt": Tensor(N,D)} — CPU
    loss_fn: TotalLoss,
    optimizer,
    scaler: GradScaler,
    scheduler,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    max_grad_norm: float,
    device: str,
    amp_device_type: str,
    amp_enabled: bool,
) -> dict:
    """
    배치 키:
        student_image           (B, 3, 224, 224)
        student_input_ids       (B, 77)
        student_attention_mask  (B, 77)
        sample_idx              (B,)  LongTensor — 캐시 인덱싱용
        caption                 List[str]         — 로깅용 (forward 에 불필요)
    """
    student_images = batch["student_image"].to(device)
    student_ids    = batch["student_input_ids"].to(device)
    student_mask   = batch["student_attention_mask"].to(device)
    indices        = batch["sample_idx"]               # CPU LongTensor

    optimizer.zero_grad()

    with autocast(device_type=amp_device_type, enabled=amp_enabled):
        # ── 학생 forward ────────────────────────────────────────────────────
        image_embeds, text_embeds, logit_scale, img_proj_sig, txt_proj_sig = model(
            student_images, student_ids, student_mask, return_projections=True
        )

        # ── 캐시에서 teacher 임베딩 슬라이싱 ──────────────────────────────
        # teacher forward 를 완전히 제거하고 사전 계산된 캐시를 사용합니다.
        sig_img_e = sig_txt_e = None
        if embedding_cache is not None:
            # fp16 캐시 → autocast 내에서 자동으로 모델 dtype 에 맞춰짐
            sig_img_e = embedding_cache["img"][indices].to(device, non_blocking=True)
            sig_txt_e = embedding_cache["txt"][indices].to(device, non_blocking=True)

        # ── 손실 계산 ───────────────────────────────────────────────────────
        losses = loss_fn(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            logit_scale=logit_scale,
            img_proj_sig=img_proj_sig,
            txt_proj_sig=txt_proj_sig,
            sig_image_embeds=sig_img_e,
            sig_text_embeds=sig_txt_e,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
        )

    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    return {k: v.item() for k, v in losses.items()}


# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------

class StageTrainer:
    """
    Args:
        embedding_cache: build_embedding_cache() 반환값.
                         None 이면 증류 loss 없이 L_CLIP 만 학습 (디버깅용).
    """

    def __init__(
        self,
        model: StudentCLIP,
        embedding_cache: Optional[dict],
        cfg: ConfigNode,
        device: str,
    ):
        self.model           = model
        self.embedding_cache = embedding_cache
        self.cfg             = cfg
        self.device          = device
        self.output_dir      = Path(cfg.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = TotalLoss(kl_temperature=cfg.training.kl_temperature)

        is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.amp_device_type = "cuda" if is_cuda else "cpu"
        self.amp_enabled     = is_cuda
        self.scaler          = GradScaler(self.amp_device_type, enabled=self.amp_enabled)

    def save_checkpoint(self, stage: int, epoch: int):
        path = self.output_dir / f"stage{stage}_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": self.model.state_dict(),
        }, path)
        logger.info(f"[Stage {stage}] 체크포인트 저장: {path}")

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info(f"체크포인트 로드: {path}")

    def run_epoch(self, dataloader, optimizer, scheduler, l1, l2, l3) -> dict:
        self.model.train()
        totals = {"total": 0.0, "l_clip": 0.0, "l_mse": 0.0, "l_kl": 0.0}
        n = 0
        for batch in dataloader:
            step = train_one_step(
                model=self.model,
                batch=batch,
                embedding_cache=self.embedding_cache,
                loss_fn=self.loss_fn,
                optimizer=optimizer,
                scaler=self.scaler,
                scheduler=scheduler,
                lambda1=l1,
                lambda2=l2,
                lambda3=l3,
                max_grad_norm=self.cfg.training.max_grad_norm,
                device=self.device,
                amp_device_type=self.amp_device_type,
                amp_enabled=self.amp_enabled,
            )
            for k in totals:
                totals[k] += step.get(k, 0.0)
            n += 1
        return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Stage 1: SigLIP2 Warm-up (텍스트 인코더 동결)
# ---------------------------------------------------------------------------

class Stage1Trainer(StageTrainer):
    """텍스트 인코더 동결 + SigLIP2 캐시 MSE 위주."""

    def run(self, dataloader, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        s = self.cfg.training.stage1
        if s.freeze_text:
            self.model.freeze_text_encoder()
            logger.info("[Stage 1] 텍스트 인코더 FROZEN")

        optimizer   = build_optimizer(self.model, s.lr, s.weight_decay)
        total_steps = len(dataloader) * s.epochs
        scheduler   = build_lr_scheduler(optimizer, total_steps, s.warmup_steps)

        for epoch in range(s.epochs):
            losses = self.run_epoch(
                dataloader, optimizer, scheduler,
                l1=s.lambda1, l2=s.lambda2, l3=s.lambda3,
            )
            logger.info(
                f"[Stage1 {epoch+1:02d}/{s.epochs}] "
                f"total={losses['total']:.4f}  clip={losses['l_clip']:.4f}  "
                f"mse={losses['l_mse']:.4f}  kl={losses['l_kl']:.4f}"
            )
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(1, epoch + 1)

        self.save_checkpoint(1, s.epochs)


# ---------------------------------------------------------------------------
# Stage 2: SigLIP2 KD 메인 학습 (동적 λ)
# ---------------------------------------------------------------------------

class Stage2Trainer(StageTrainer):
    """텍스트 인코더 해제 + 동적 λ2/λ3."""

    def run(self, dataloader, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        s = self.cfg.training.stage2
        self.model.unfreeze_text_encoder()
        logger.info("[Stage 2] 텍스트 인코더 UNFROZEN")

        optimizer   = build_optimizer(self.model, s.lr, s.weight_decay)
        total_steps = len(dataloader) * s.epochs
        scheduler   = build_lr_scheduler(optimizer, total_steps, s.warmup_steps)

        for epoch in range(s.epochs):
            l1, l2, l3 = get_stage2_lambdas(epoch, s.epochs, s)
            losses = self.run_epoch(dataloader, optimizer, scheduler, l1, l2, l3)
            logger.info(
                f"[Stage2 {epoch+1:02d}/{s.epochs}] "
                f"λ=({l1:.3f},{l2:.3f},{l3:.3f}) | "
                f"total={losses['total']:.4f}  clip={losses['l_clip']:.4f}  "
                f"mse={losses['l_mse']:.4f}  kl={losses['l_kl']:.4f}"
            )
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(2, epoch + 1)

        self.save_checkpoint(2, s.epochs)


# ---------------------------------------------------------------------------
# Stage 3: QAT Fine-tuning
# ---------------------------------------------------------------------------

class Stage3Trainer(StageTrainer):
    """양자화 보정 + L_CLIP 비중 상향."""

    def run(self, dataloader, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        s = self.cfg.training.stage3
        self._enable_qat(s)

        optimizer   = build_optimizer(self.model, s.lr, s.weight_decay)
        total_steps = len(dataloader) * s.epochs
        scheduler   = build_lr_scheduler(optimizer, total_steps, s.warmup_steps)

        for epoch in range(s.epochs):
            losses = self.run_epoch(
                dataloader, optimizer, scheduler,
                l1=s.lambda1, l2=s.lambda2, l3=s.lambda3,
            )
            logger.info(
                f"[Stage3 {epoch+1:02d}/{s.epochs}] "
                f"total={losses['total']:.4f}  clip={losses['l_clip']:.4f}  "
                f"mse={losses['l_mse']:.4f}  kl={losses['l_kl']:.4f}"
            )

        self.save_checkpoint(3, s.epochs)

    def _enable_qat(self, s_cfg):
        if s_cfg.get("qat_image_encoder", True):
            try:
                self.model.image_encoder.qconfig = \
                    torch.quantization.get_default_qat_qconfig("fbgemm")
                torch.quantization.prepare_qat(self.model.image_encoder, inplace=True)
                logger.info("[Stage 3] QAT 준비 완료 — image_encoder")
            except Exception as e:
                logger.warning(f"[Stage 3] image_encoder QAT 실패: {e}")

        if s_cfg.get("qat_text_encoder", False):
            try:
                self.model.text_encoder.qconfig = \
                    torch.quantization.get_default_qat_qconfig("fbgemm")
                torch.quantization.prepare_qat(self.model.text_encoder, inplace=True)
                logger.info("[Stage 3] QAT 준비 완료 — text_encoder")
            except Exception as e:
                logger.warning(f"[Stage 3] text_encoder QAT 실패: {e}")
