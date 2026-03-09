"""
Loss functions for LPCVC 2026 Track 1 dual distillation training.

L_total = λ1 * L_CLIP + λ2 * L_MSE(feat_DFN) + λ3 * L_KL(dist_SigLIP2 + dist_DFN)

- L_CLIP  : Standard symmetric contrastive (InfoNCE) loss for student.
- L_MSE   : Feature mimicking loss – student projections match DFN embeddings.
- L_KL    : Soft label distillation – student similarity matches teacher's.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# L_CLIP: Symmetric InfoNCE (Contrastive) Loss
# ---------------------------------------------------------------------------

def clip_contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss (identical to original CLIP).

    Args:
        image_embeds: (B, D) – L2-normalized
        text_embeds:  (B, D) – L2-normalized
        logit_scale:  scalar – exp(learnable log temperature)

    Returns:
        Scalar loss.
    """
    batch_size = image_embeds.shape[0]
    labels = torch.arange(batch_size, device=image_embeds.device)

    logits_i2t = logit_scale * image_embeds @ text_embeds.T   # (B, B)
    logits_t2i = logits_i2t.T

    loss_i2t = F.cross_entropy(logits_i2t, labels)
    loss_t2i = F.cross_entropy(logits_t2i, labels)

    return (loss_i2t + loss_t2i) / 2.0


# ---------------------------------------------------------------------------
# L_MSE: Feature Mimicking Loss (DFN warm-up)
# ---------------------------------------------------------------------------

def feature_mimicking_loss(
    student_proj: torch.Tensor,
    teacher_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss between student projected embeddings and teacher embeddings.
    Both inputs are L2-normalized before MSE computation.

    Args:
        student_proj:  (B, D_teacher) – student projection into teacher space
        teacher_embeds: (B, D_teacher) – teacher embeddings (already normalized)

    Returns:
        Scalar MSE loss.
    """
    student_norm = F.normalize(student_proj, dim=-1)
    return F.mse_loss(student_norm, teacher_embeds.detach())


# ---------------------------------------------------------------------------
# L_KL: Soft Label Distribution Distillation
# ---------------------------------------------------------------------------

def compute_similarity_distribution(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: float = 4.0,
) -> tuple:
    """
    Compute i2t and t2i softmax distributions over similarity logits.

    Args:
        image_embeds: (B, D) – normalized
        text_embeds:  (B, D) – normalized
        temperature:  Scaling temperature (higher = softer distribution)

    Returns:
        dist_i2t: (B, B)
        dist_t2i: (B, B)
    """
    logits = (image_embeds @ text_embeds.T) / temperature
    dist_i2t = torch.softmax(logits, dim=-1)
    dist_t2i = torch.softmax(logits.T, dim=-1)
    return dist_i2t, dist_t2i


def kl_distillation_loss(
    student_i2t: torch.Tensor,
    student_t2i: torch.Tensor,
    teacher_i2t: torch.Tensor,
    teacher_t2i: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence loss between student and teacher similarity distributions.
    KL(teacher || student)  — student learns to match teacher's distribution.

    Args:
        student_i2t:  (B, B) student image→text softmax distribution
        student_t2i:  (B, B) student text→image softmax distribution
        teacher_i2t:  (B, B) teacher image→text softmax distribution (target)
        teacher_t2i:  (B, B) teacher text→image softmax distribution (target)

    Returns:
        Scalar KL loss.
    """
    # KL(P || Q) = sum(P * log(P/Q))
    # F.kl_div expects log-probabilities as input, probabilities as target
    kl_i2t = F.kl_div(
        torch.log(student_i2t + 1e-8),
        teacher_i2t.detach(),
        reduction="batchmean",
    )
    kl_t2i = F.kl_div(
        torch.log(student_t2i + 1e-8),
        teacher_t2i.detach(),
        reduction="batchmean",
    )
    return (kl_i2t + kl_t2i) / 2.0


def dual_teacher_kl_loss(
    student_image_embeds: torch.Tensor,
    student_text_embeds: torch.Tensor,
    siglip_image_embeds: torch.Tensor,
    siglip_text_embeds: torch.Tensor,
    dfn_image_embeds: torch.Tensor,
    dfn_text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
    kl_temperature: float = 4.0,
) -> torch.Tensor:
    """
    Combined KL distillation from both teachers.
    Student learns the average similarity distribution of SigLIP2 and DFN.

    Args:
        student_image_embeds: (B, D_student) – normalized
        student_text_embeds:  (B, D_student) – normalized
        siglip_image_embeds:  (B, D_sig)     – normalized
        siglip_text_embeds:   (B, D_sig)     – normalized
        dfn_image_embeds:     (B, D_dfn)     – normalized
        dfn_text_embeds:      (B, D_dfn)     – normalized
        logit_scale:          scalar
        kl_temperature:       Temperature for teacher distributions

    Returns:
        Scalar combined KL loss.
    """
    # --- Student distributions (using student's own logit scale) ---
    student_temp = kl_temperature / logit_scale.detach()
    stu_i2t, stu_t2i = compute_similarity_distribution(
        student_image_embeds, student_text_embeds, temperature=student_temp
    )

    # --- SigLIP2 teacher distributions ---
    sig_i2t, sig_t2i = compute_similarity_distribution(
        siglip_image_embeds, siglip_text_embeds, temperature=kl_temperature
    )

    # --- DFN teacher distributions ---
    dfn_i2t, dfn_t2i = compute_similarity_distribution(
        dfn_image_embeds, dfn_text_embeds, temperature=kl_temperature
    )

    # Average the two teacher distributions
    avg_i2t = (sig_i2t + dfn_i2t) / 2.0
    avg_t2i = (sig_t2i + dfn_t2i) / 2.0

    return kl_distillation_loss(stu_i2t, stu_t2i, avg_i2t, avg_t2i)


# ---------------------------------------------------------------------------
# Combined Total Loss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    Unified loss module combining L_CLIP, L_MSE, and L_KL.

    L_total = λ1 * L_CLIP + λ2 * L_MSE(DFN) + λ3 * L_KL(avg_teacher)
    """

    def __init__(self, kl_temperature: float = 4.0):
        super().__init__()
        self.kl_temperature = kl_temperature

    def forward(
        self,
        # Student outputs
        image_embeds: torch.Tensor,       # (B, D) normalized
        text_embeds: torch.Tensor,        # (B, D) normalized
        logit_scale: torch.Tensor,        # scalar
        # Student projections (for MSE / KD on teacher spaces)
        img_proj_dfn: Optional[torch.Tensor] = None,   # (B, D_dfn)
        txt_proj_dfn: Optional[torch.Tensor] = None,   # (B, D_dfn)
        # Teacher embeddings
        siglip_image_embeds: Optional[torch.Tensor] = None,
        siglip_text_embeds: Optional[torch.Tensor] = None,
        dfn_image_embeds: Optional[torch.Tensor] = None,
        dfn_text_embeds: Optional[torch.Tensor] = None,
        # Loss weights
        lambda1: float = 0.3,    # L_CLIP
        lambda2: float = 0.4,    # L_MSE
        lambda3: float = 0.3,    # L_KL
    ) -> dict:
        """
        Returns a dict with individual loss components and total loss.
        """
        losses = {}

        # L_CLIP: always computed
        l_clip = clip_contrastive_loss(image_embeds, text_embeds, logit_scale)
        losses["l_clip"] = l_clip

        # L_MSE: DFN feature mimicking
        l_mse = torch.tensor(0.0, device=image_embeds.device)
        if lambda2 > 0 and img_proj_dfn is not None and dfn_image_embeds is not None:
            l_mse_img = feature_mimicking_loss(img_proj_dfn, dfn_image_embeds)
            l_mse_txt = feature_mimicking_loss(txt_proj_dfn, dfn_text_embeds)
            l_mse = (l_mse_img + l_mse_txt) / 2.0
        losses["l_mse"] = l_mse

        # L_KL: dual teacher distribution distillation
        l_kl = torch.tensor(0.0, device=image_embeds.device)
        if (
            lambda3 > 0
            and siglip_image_embeds is not None
            and dfn_image_embeds is not None
        ):
            l_kl = dual_teacher_kl_loss(
                student_image_embeds=image_embeds,
                student_text_embeds=text_embeds,
                siglip_image_embeds=siglip_image_embeds,
                siglip_text_embeds=siglip_text_embeds,
                dfn_image_embeds=dfn_image_embeds,
                dfn_text_embeds=dfn_text_embeds,
                logit_scale=logit_scale,
                kl_temperature=self.kl_temperature,
            )
        losses["l_kl"] = l_kl

        # Total
        total = lambda1 * l_clip + lambda2 * l_mse + lambda3 * l_kl
        losses["total"] = total

        return losses
