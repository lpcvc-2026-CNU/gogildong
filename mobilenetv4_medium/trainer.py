"""
Stage-aware trainer for LPCVC 2026 Track 1 (Single Teacher: SigLIP2).

토크나이저 흐름:
  - 배치의 student_input_ids / student_attention_mask → DistilBERT (학생)
  - 배치의 caption (List[str])                        → SigLIP2.encode_text() (자체 processor)
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from student import StudentCLIP
from teacher import TeacherManager
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
    def _lr_lambda(step):
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
    t = epoch / max(1, total_epochs - 1)   # 0.0 → 1.0

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
    teacher_manager: TeacherManager,
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
      student_image          (B, 3, 224, 224)
      student_input_ids      (B, 77)
      student_attention_mask (B, 77)
      teacher_image_sig      (B, 3, H, H)   — SigLIP2 입력
      caption                List[str]       — SigLIP2 텍스트 인코딩용
    """
    student_images = batch["student_image"].to(device)
    student_ids    = batch["student_input_ids"].to(device)
    student_mask   = batch["student_attention_mask"].to(device)
    captions       = batch["caption"]

    optimizer.zero_grad()

    with autocast(device_type=amp_device_type, enabled=amp_enabled):
        # ── 학생 forward ────────────────────────────────────────────
        image_embeds, text_embeds, logit_scale, img_proj_sig, txt_proj_sig = model(
            student_images, student_ids, student_mask, return_projections=True
        )

        # ── SigLIP2 스승 forward ────────────────────────────────────
        sig_img_e = sig_txt_e = None
        if teacher_manager is not None and "teacher_image_sig" in batch:
            sig_imgs  = batch["teacher_image_sig"].to(device)
            sig_img_e = teacher_manager.get_image_embeds(sig_imgs)
            sig_txt_e = teacher_manager.get_text_embeds(captions)

        # ── 손실 계산 ───────────────────────────────────────────────
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
    def __init__(self, model: StudentCLIP, teacher_manager, cfg: ConfigNode, device: str):
        self.model           = model
        self.teacher_manager = teacher_manager
        self.cfg             = cfg
        self.device          = device
        self.output_dir      = Path(cfg.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = TotalLoss(kl_temperature=cfg.training.kl_temperature)
        is_cuda_device = str(device).startswith("cuda") and torch.cuda.is_available()
        self.amp_device_type = "cuda" if is_cuda_device else "cpu"
        self.amp_enabled = is_cuda_device
        self.scaler  = GradScaler(self.amp_device_type, enabled=self.amp_enabled)

    def save_checkpoint(self, stage: int, epoch: int):
        path = self.output_dir / f"stage{stage}_epoch{epoch:03d}.pt"
        torch.save({"epoch": epoch, "stage": stage,
                    "model_state_dict": self.model.state_dict()}, path)
        logger.info(f"[Stage {stage}] 저장: {path}")

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
                self.model, batch, self.teacher_manager, self.loss_fn,
                optimizer, self.scaler, scheduler,
                l1, l2, l3,
                self.cfg.training.max_grad_norm, self.device,
                self.amp_device_type, self.amp_enabled,
            )
            for k in totals:
                totals[k] += step.get(k, 0.0)
            n += 1
        return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Stage 1: SigLIP2 Warm-up
# ---------------------------------------------------------------------------

class Stage1Trainer(StageTrainer):
    """텍스트 인코더 동결 + SigLIP2 MSE 위주."""

    def run(self, dataloader, resume_from=None):
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
            logger.info(f"[Stage1 {epoch+1}/{s.epochs}] {losses}")
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(1, epoch + 1)
        self.save_checkpoint(1, s.epochs)


# ---------------------------------------------------------------------------
# Stage 2: SigLIP2 KD (동적 λ)
# ---------------------------------------------------------------------------

class Stage2Trainer(StageTrainer):
    """텍스트 인코더 해제 + 동적 λ2/λ3."""

    def run(self, dataloader, resume_from=None):
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
                f"[Stage2 {epoch+1}/{s.epochs}] "
                f"λ1={l1:.3f} λ2={l2:.3f} λ3={l3:.3f} | {losses}"
            )
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(2, epoch + 1)
        self.save_checkpoint(2, s.epochs)


# ---------------------------------------------------------------------------
# Stage 3: QAT Fine-tuning
# ---------------------------------------------------------------------------

class Stage3Trainer(StageTrainer):
    """양자화 보정 + L_CLIP 비중 상향."""

    def run(self, dataloader, resume_from=None):
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
            logger.info(f"[Stage3 {epoch+1}/{s.epochs}] {losses}")
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
