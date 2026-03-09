"""
Student CLIP model for LPCVC 2026 Track 1.

Image Encoder : MobileNetV4-Small  (224x224 입력, GAP -> embed_dim)
Text Encoder  : DistilBERT-Base    (CLS 토큰 -> embed_dim)
Projection    : DualProjectionHeads (shared embed -> SigLIP2 공간 & DFN 공간)

모든 차원/모델명은 cfg(ConfigNode) 에서 읽습니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig

from models.projection import DualProjectionHeads
from utils.config import ConfigNode


# ---------------------------------------------------------------------------
# Image Encoder
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    MobileNetV4-Small backbone + Linear projection → shared embed_dim.

    Input : (B, 3, student_image_input_size, student_image_input_size)
    Output: (B, embed_dim)  – unnormalized
    """

    def __init__(self, cfg: ConfigNode, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model.student_image_backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        backbone_out_dim = self.backbone.num_features
        self.proj = nn.Linear(backbone_out_dim, cfg.model.embed_dim, bias=False)
        self.norm = nn.LayerNorm(cfg.model.embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features   = self.backbone(pixel_values)
        embeddings = self.norm(self.proj(features))
        return embeddings


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    DistilBERT-Base + Linear projection → shared embed_dim.

    Input : input_ids (B, max_text_length), attention_mask (B, max_text_length)
    Output: (B, embed_dim) – unnormalized
    """

    def __init__(self, cfg: ConfigNode, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.distilbert = DistilBertModel.from_pretrained(
                cfg.model.student_text_backbone
            )
        else:
            self.distilbert = DistilBertModel(DistilBertConfig())

        hidden_dim = self.distilbert.config.hidden_size
        self.proj  = nn.Linear(hidden_dim, cfg.model.embed_dim, bias=False)
        self.norm  = nn.LayerNorm(cfg.model.embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs    = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS] 토큰
        return self.norm(self.proj(cls_output))

    def freeze(self):
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.distilbert.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Student CLIP
# ---------------------------------------------------------------------------

class StudentCLIP(nn.Module):
    """
    이미지-텍스트 대조 학습 + 이중 스승 지식 증류 학생 모델.

    학습 시: shared embed + projection heads (KD용) 모두 반환
    추론 시: L2-normalized shared embed 만 반환 (Recall@10 계산용)
    """

    def __init__(self, cfg: ConfigNode):
        super().__init__()
        self.cfg = cfg

        self.image_encoder = ImageEncoder(cfg)
        self.text_encoder  = TextEncoder(cfg)

        self.image_proj_heads = DualProjectionHeads(cfg)
        self.text_proj_heads  = DualProjectionHeads(cfg)

        # 학습 가능한 온도 파라미터 (초기값: ln(1/0.07) ≈ 2.659)
        import math
        init_val = -math.log(cfg.training.temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_val)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """L2-normalized 이미지 임베딩 반환 (추론용)."""
        return F.normalize(self.image_encoder(pixel_values), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """L2-normalized 텍스트 임베딩 반환 (추론용)."""
        return F.normalize(self.text_encoder(input_ids, attention_mask), dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_projections: bool = True,
    ):
        image_feats = self.image_encoder(pixel_values)
        text_feats  = self.text_encoder(input_ids, attention_mask)

        image_embeds = F.normalize(image_feats, dim=-1)
        text_embeds  = F.normalize(text_feats, dim=-1)
        logit_scale  = self.logit_scale.exp()

        if not return_projections:
            return image_embeds, text_embeds, logit_scale

        img_proj_sig, img_proj_dfn = self.image_proj_heads(image_feats)
        txt_proj_sig, txt_proj_dfn = self.text_proj_heads(text_feats)

        return (
            image_embeds, text_embeds, logit_scale,
            img_proj_sig, img_proj_dfn,
            txt_proj_sig, txt_proj_dfn,
        )

    def freeze_text_encoder(self):
        self.text_encoder.freeze()

    def unfreeze_text_encoder(self):
        self.text_encoder.unfreeze()


def build_student_model(cfg: ConfigNode) -> StudentCLIP:
    return StudentCLIP(cfg)
