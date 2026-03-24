"""
Loss functions for LPCVC 2026 Track 1 (Single Teacher: SigLIP2).

L_total = λ1 * L_CLIP + λ2 * L_MSE(proj_sig, sig_embeds) + λ3 * L_KL(sig)

- L_CLIP : Symmetric InfoNCE — student 자체 대조 학습
- L_MSE  : Feature mimicking — student projection → SigLIP2 공간 MSE
- L_KL   : Soft label distillation — student 유사도 분포가 SigLIP2를 따르도록
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# L_CLIP: Symmetric InfoNCE
# ---------------------------------------------------------------------------

def clip_contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        image_embeds: (B, D) — L2-normalized
        text_embeds:  (B, D) — L2-normalized
        logit_scale:  scalar — exp(learnable log temperature)
    """
    B      = image_embeds.shape[0]
    labels = torch.arange(B, device=image_embeds.device)

    logits_i2t = logit_scale * image_embeds @ text_embeds.T
    loss = (F.cross_entropy(logits_i2t, labels) +
            F.cross_entropy(logits_i2t.T, labels)) / 2.0
    return loss


# ---------------------------------------------------------------------------
# L_MSE: Feature Mimicking
# ---------------------------------------------------------------------------

def feature_mimicking_loss(
    student_proj: torch.Tensor,
    teacher_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between L2-normalized student projection and (already normalized) teacher embeds.

    Args:
        student_proj:  (B, D_teacher) — raw projection output
        teacher_embeds: (B, D_teacher) — L2-normalized teacher embeddings
    """
    student_norm = F.normalize(student_proj, dim=-1)
    return F.mse_loss(student_norm, teacher_embeds.detach())


# ---------------------------------------------------------------------------
# L_KL: Soft Label Distribution Distillation
# ---------------------------------------------------------------------------

def _sim_distribution(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: float,
):
    """Return softmax i2t and t2i similarity distributions."""
    logits   = (image_embeds @ text_embeds.T) / temperature
    dist_i2t = torch.softmax(logits,   dim=-1)
    dist_t2i = torch.softmax(logits.T, dim=-1)
    return dist_i2t, dist_t2i


def kl_distillation_loss(
    student_image_embeds: torch.Tensor,
    student_text_embeds: torch.Tensor,
    teacher_image_embeds: torch.Tensor,
    teacher_text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    kl_temperature: float = 4.0,
) -> torch.Tensor:
    """
    KL(teacher_dist || student_dist) — symmetric across i2t and t2i.

    Student temperature is derived from logit_scale so it remains consistent
    with the contrastive loss.
    """
    student_temp = kl_temperature / logit_scale.detach().clamp(min=1e-4)

    stu_i2t, stu_t2i = _sim_distribution(
        student_image_embeds, student_text_embeds, temperature=student_temp
    )
    tch_i2t, tch_t2i = _sim_distribution(
        teacher_image_embeds, teacher_text_embeds, temperature=kl_temperature
    )

    kl_i2t = F.kl_div(
        torch.log(stu_i2t + 1e-8),
        tch_i2t.detach(),
        reduction="batchmean",
    )
    kl_t2i = F.kl_div(
        torch.log(stu_t2i + 1e-8),
        tch_t2i.detach(),
        reduction="batchmean",
    )
    return (kl_i2t + kl_t2i) / 2.0


# ---------------------------------------------------------------------------
# TotalLoss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    L_total = λ1 * L_CLIP + λ2 * L_MSE(sig) + λ3 * L_KL(sig)

    forward() 인자:
      student outputs  : image_embeds, text_embeds, logit_scale
      student proj     : img_proj_sig, txt_proj_sig  (SigLIP2 공간)
      teacher embeds   : sig_image_embeds, sig_text_embeds  (nullable)
      loss weights     : lambda1, lambda2, lambda3
    """

    def __init__(self, kl_temperature: float = 4.0):
        super().__init__()
        self.kl_temperature = kl_temperature

    def forward(
        self,
        # ── student outputs ──────────────────────────────────────────
        image_embeds: torch.Tensor,          # (B, D) normalized
        text_embeds: torch.Tensor,           # (B, D) normalized
        logit_scale: torch.Tensor,           # scalar
        # ── student projections (SigLIP2 공간) ──────────────────────
        img_proj_sig: Optional[torch.Tensor] = None,   # (B, D_sig)
        txt_proj_sig: Optional[torch.Tensor] = None,   # (B, D_sig)
        # ── SigLIP2 teacher embeddings ───────────────────────────────
        sig_image_embeds: Optional[torch.Tensor] = None,  # (B, D_sig) normalized
        sig_text_embeds: Optional[torch.Tensor] = None,   # (B, D_sig) normalized
        # ── loss weights ─────────────────────────────────────────────
        lambda1: float = 0.3,   # L_CLIP
        lambda2: float = 0.4,   # L_MSE
        lambda3: float = 0.3,   # L_KL
    ) -> dict:
        losses = {}

        # L_CLIP — 항상 계산
        losses["l_clip"] = clip_contrastive_loss(image_embeds, text_embeds, logit_scale)

        # L_MSE — teacher가 있고 lambda2 > 0 일 때
        l_mse = torch.zeros(1, device=image_embeds.device)
        if lambda2 > 0 and img_proj_sig is not None and sig_image_embeds is not None:
            l_mse = (
                feature_mimicking_loss(img_proj_sig, sig_image_embeds) +
                feature_mimicking_loss(txt_proj_sig, sig_text_embeds)
            ) / 2.0
        losses["l_mse"] = l_mse

        # L_KL — teacher가 있고 lambda3 > 0 일 때
        l_kl = torch.zeros(1, device=image_embeds.device)
        if lambda3 > 0 and sig_image_embeds is not None:
            l_kl = kl_distillation_loss(
                student_image_embeds=image_embeds,
                student_text_embeds=text_embeds,
                teacher_image_embeds=sig_image_embeds,
                teacher_text_embeds=sig_text_embeds,
                logit_scale=logit_scale,
                kl_temperature=self.kl_temperature,
            )
        losses["l_kl"] = l_kl

        losses["total"] = lambda1 * losses["l_clip"] + lambda2 * l_mse + lambda3 * l_kl
        return losses
