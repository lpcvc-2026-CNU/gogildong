"""
Stage-aware trainer for LPCVC 2026 Track 1.

모든 하이퍼파라미터는 cfg(ConfigNode) 에서 읽습니다.
Stage 1/2/3 클래스가 각각 독립적으로 실행 가능합니다.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from models.student import StudentCLIP
from models.teacher import TeacherManager
from training.loss import TotalLoss
from utils.config import ConfigNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer 및 LR 스케줄러
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    """bias / LayerNorm 파라미터에는 weight decay 미적용."""
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
    """Linear warmup → Cosine decay LR 스케줄러."""
    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Stage 2 동적 λ 가중치
# ---------------------------------------------------------------------------

def get_stage2_lambdas(epoch: int, total_epochs: int, stage_cfg: ConfigNode):
    """
    Stage 2 전용: 에폭 진행에 따라 λ2(MSE)를 줄이고 λ3(KL)를 늘립니다.
    λ1 은 고정, 세 값의 합이 1이 되도록 정규화합니다.
    """
    progress = epoch / max(1, total_epochs - 1)
    l2 = stage_cfg.lambda2_start + progress * (stage_cfg.lambda2_end - stage_cfg.lambda2_start)
    l3 = stage_cfg.lambda3_start + progress * (stage_cfg.lambda3_end - stage_cfg.lambda3_start)
    total = stage_cfg.lambda1 + l2 + l3
    return stage_cfg.lambda1 / total, l2 / total, l3 / total


# ---------------------------------------------------------------------------
# 단일 스텝 학습 함수
# ---------------------------------------------------------------------------

def train_one_step(
    model: StudentCLIP,
    batch: dict,
    teacher_manager: Optional[TeacherManager],
    loss_fn: TotalLoss,
    optimizer,
    scaler: GradScaler,
    scheduler,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    max_grad_norm: float,
    device: str,
) -> dict:
    """단일 배치에 대한 순전파/역전파/파라미터 업데이트."""
    student_images  = batch["student_image"].to(device)
    input_ids       = batch["input_ids"].to(device)
    attention_mask  = batch["attention_mask"].to(device)

    optimizer.zero_grad()

    with autocast():
        (
            image_embeds, text_embeds, logit_scale,
            img_proj_sig, img_proj_dfn,
            txt_proj_sig, txt_proj_dfn,
        ) = model(student_images, input_ids, attention_mask, return_projections=True)

        sig_img_e = sig_txt_e = dfn_img_e = dfn_txt_e = None

        if teacher_manager is not None:
            if teacher_manager.siglip2 is not None and "teacher_image_sig" in batch:
                sig_imgs  = batch["teacher_image_sig"].to(device)
                sig_img_e = teacher_manager.get_siglip_image_embeds(sig_imgs)
                sig_txt_e = teacher_manager.get_siglip_text_embeds(input_ids, attention_mask)

            if teacher_manager.dfn is not None and "teacher_image_dfn" in batch:
                dfn_imgs  = batch["teacher_image_dfn"].to(device)
                dfn_img_e = teacher_manager.get_dfn_image_embeds(dfn_imgs)
                dfn_txt_e = teacher_manager.get_dfn_text_embeds(input_ids)

        losses = loss_fn(
            image_embeds=image_embeds,
            text_embeds=text_embeds,
            logit_scale=logit_scale,
            img_proj_dfn=img_proj_dfn,
            txt_proj_dfn=txt_proj_dfn,
            siglip_image_embeds=sig_img_e,
            siglip_text_embeds=sig_txt_e,
            dfn_image_embeds=dfn_img_e,
            dfn_text_embeds=dfn_txt_e,
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
    """공통 베이스 트레이너."""

    def __init__(
        self,
        model: StudentCLIP,
        teacher_manager: Optional[TeacherManager],
        cfg: ConfigNode,
        device: str,
    ):
        self.model           = model
        self.teacher_manager = teacher_manager
        self.cfg             = cfg
        self.device          = device
        self.output_dir      = Path(cfg.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = TotalLoss(kl_temperature=cfg.training.kl_temperature)
        self.scaler  = GradScaler()

    def save_checkpoint(self, stage: int, epoch: int):
        path = self.output_dir / f"stage{stage}_epoch{epoch:03d}.pt"
        torch.save({"epoch": epoch, "stage": stage, "model_state_dict": self.model.state_dict()}, path)
        logger.info(f"[Stage {stage}] Checkpoint 저장: {path}")

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        logger.info(f"Checkpoint 로드: {path}")

    def run_epoch(self, dataloader, optimizer, scheduler, lambda1, lambda2, lambda3) -> dict:
        self.model.train()
        totals = {"total": 0.0, "l_clip": 0.0, "l_mse": 0.0, "l_kl": 0.0}
        n = 0
        for batch in dataloader:
            step = train_one_step(
                self.model, batch, self.teacher_manager, self.loss_fn,
                optimizer, self.scaler, scheduler,
                lambda1, lambda2, lambda3,
                self.cfg.training.max_grad_norm, self.device,
            )
            for k in totals:
                totals[k] += step.get(k, 0.0)
            n += 1
        return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

class Stage1Trainer(StageTrainer):
    """Stage 1: DFN Warm-up — 텍스트 인코더 동결, L_MSE 위주."""

    def run(self, dataloader, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        s = self.cfg.training.stage1
        if s.freeze_text:
            self.model.freeze_text_encoder()
            logger.info("[Stage 1] 텍스트 인코더 FROZEN")

        optimizer  = build_optimizer(self.model, s.lr, s.weight_decay)
        total_steps = len(dataloader) * s.epochs
        scheduler  = build_lr_scheduler(optimizer, total_steps, s.warmup_steps)

        for epoch in range(s.epochs):
            losses = self.run_epoch(dataloader, optimizer, scheduler, s.lambda1, s.lambda2, s.lambda3)
            logger.info(f"[Stage1 {epoch+1}/{s.epochs}] {losses}")
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(1, epoch + 1)

        self.save_checkpoint(1, s.epochs)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

class Stage2Trainer(StageTrainer):
    """Stage 2: Dual Teacher KD — 텍스트 인코더 해제, 동적 λ."""

    def run(self, dataloader, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        s = self.cfg.training.stage2
        if not s.freeze_text:
            self.model.unfreeze_text_encoder()
            logger.info("[Stage 2] 텍스트 인코더 UNFROZEN")

        optimizer   = build_optimizer(self.model, s.lr, s.weight_decay)
        total_steps = len(dataloader) * s.epochs
        scheduler   = build_lr_scheduler(optimizer, total_steps, s.warmup_steps)

        for epoch in range(s.epochs):
            l1, l2, l3 = get_stage2_lambdas(epoch, s.epochs, s)
            losses = self.run_epoch(dataloader, optimizer, scheduler, l1, l2, l3)
            logger.info(
                f"[Stage2 {epoch+1}/{s.epochs}] λ1={l1:.3f} λ2={l2:.3f} λ3={l3:.3f} | {losses}"
            )
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(2, epoch + 1)

        self.save_checkpoint(2, s.epochs)


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

class Stage3Trainer(StageTrainer):
    """Stage 3: QAT Fine-tuning — 양자화 보정, L_CLIP 비중 상향."""

    def run(self, dataloader, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        s = self.cfg.training.stage3
        self._enable_qat()

        optimizer   = build_optimizer(self.model, s.lr, s.weight_decay)
        total_steps = len(dataloader) * s.epochs
        scheduler   = build_lr_scheduler(optimizer, total_steps, s.warmup_steps)

        for epoch in range(s.epochs):
            losses = self.run_epoch(dataloader, optimizer, scheduler, s.lambda1, s.lambda2, s.lambda3)
            logger.info(f"[Stage3 {epoch+1}/{s.epochs}] {losses}")

        self.save_checkpoint(3, s.epochs)

    def _enable_qat(self):
        try:
            self.model.image_encoder.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
            torch.quantization.prepare_qat(self.model.image_encoder, inplace=True)
            logger.info("[Stage 3] QAT 준비 완료 (image_encoder)")
        except Exception as e:
            logger.warning(f"[Stage 3] QAT 설정 실패 (ONNX export 에는 영향 없음): {e}")
