"""
Student CLIP model for LPCVC 2026 Track 1 (Single Teacher: SigLIP2).

토크나이저 정책:
  대회 평가 규격: openai/clip-vit-base-patch32 로 tokenize 한 (1×77) 정수 텐서 입력.
  텍스트 인코더: CLIP ViT-B/32 의 text encoder (CLIPTextModel) 를 그대로 사용.
  - 입력: input_ids (B, 77), attention_mask (B, 77)
  - 출력: pooler_output — EOS 토큰 표현, hidden_size=512 (embed_dim 과 일치)
  - vocab 교체 불필요: CLIP tokenizer ↔ CLIP text model 이 동일 vocabulary 공유.

추론 시에는 shared L2-normalized embedding 만 사용.
SigLIPProjectionHead 는 학습 시에만 활성화.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import CLIPTextModel, CLIPTextConfig

from projection import SigLIPProjectionHead
from config import ConfigNode


# ---------------------------------------------------------------------------
# Image Encoder (변경 없음)
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    MobileNetV4-Medium + Linear → shared embed_dim.

    Input : (B, 3, 224, 224)
    Output: (B, embed_dim)  unnormalized
    """

    def __init__(self, cfg: ConfigNode, pretrained: bool = True):
        super().__init__()
        input_size = cfg.model.student_image_input_size
        self.backbone = timm.create_model(
            cfg.model.student_image_backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        backbone_out_dim = self._infer_backbone_out_dim(input_size)
        self.proj = nn.Linear(backbone_out_dim, cfg.model.embed_dim, bias=False)
        self.norm = nn.LayerNorm(cfg.model.embed_dim)

    def _infer_backbone_out_dim(self, input_size: int) -> int:
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            feats = self.backbone(dummy)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.ndim > 2:
                feats = torch.flatten(feats, start_dim=1)
        if was_training:
            self.backbone.train()
        return int(feats.shape[-1])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)
        return self.norm(self.proj(feats))


# ---------------------------------------------------------------------------
# Text Encoder — CLIP ViT-B/32 text encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    CLIP ViT-B/32 텍스트 인코더 + (필요 시) Linear → embed_dim.

    CLIPTextModel 의 pooler_output 을 사용합니다.
      - pooler_output: 마지막 [EOS] 토큰의 hidden state (hidden_size = 512)
      - CLIP ViT-B/32 기준 hidden_size == embed_dim == 512 이므로
        별도 projection 없이 LayerNorm 만 적용합니다.
      - hidden_size ≠ embed_dim 인 경우 자동으로 Linear projection 을 추가합니다.

    DistilBERT 대비 변경점:
      - word_embeddings vocab 교체 불필요: CLIPTokenizer 와 CLIPTextModel 이
        동일한 49408-token vocabulary 를 공유합니다.
      - [CLS] 토큰 대신 [EOS] 토큰 기반 pooler_output 사용.
      - attention_mask 는 그대로 입력받아 내부 self-attention 에서 활용합니다.

    Input : input_ids (B, 77), attention_mask (B, 77)
    Output: (B, embed_dim)  unnormalized
    """

    def __init__(self, cfg: ConfigNode, pretrained: bool = True):
        super().__init__()
        model_name = cfg.model.clip_tokenizer_name  # "openai/clip-vit-base-patch32"

        if pretrained:
            self.clip_text = CLIPTextModel.from_pretrained(model_name)
        else:
            text_config = CLIPTextConfig()
            self.clip_text = CLIPTextModel(text_config)

        clip_hidden_dim = self.clip_text.config.hidden_size  # CLIP ViT-B/32 → 512

        # embed_dim 과 차이가 있는 경우에만 projection 추가
        if clip_hidden_dim != cfg.model.embed_dim:
            self.proj: nn.Module = nn.Linear(
                clip_hidden_dim, cfg.model.embed_dim, bias=False
            )
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            self.proj = nn.Identity()

        self.norm = nn.LayerNorm(cfg.model.embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, 77)  int64 — CLIPTokenizer 출력
            attention_mask: (B, 77)  int64 — 패딩 마스크 (1=유효, 0=패딩)
        Returns:
            (B, embed_dim)  unnormalized
        """
        outputs = self.clip_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # pooler_output: EOS 토큰의 표현 — (B, hidden_size)
        feats = outputs.pooler_output
        return self.norm(self.proj(feats))

    def freeze(self):
        for p in self.clip_text.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.clip_text.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Student CLIP (변경 없음 — TextEncoder 교체만으로 자동 반영)
# ---------------------------------------------------------------------------

class StudentCLIP(nn.Module):
    """
    Full student model: ImageEncoder + TextEncoder(CLIP ViT-B/32) + SigLIPProjectionHead.

    SigLIPProjectionHead: student embed → SigLIP2 공간 (MSE + KL 증류)
    추론 시에는 shared L2-normalized embed 만 사용.
    """

    def __init__(self, cfg: ConfigNode):
        super().__init__()
        self.image_encoder = ImageEncoder(cfg)
        self.text_encoder  = TextEncoder(cfg)
        # 단일 스승(SigLIP2)용 projection head — 이미지/텍스트 공유
        self.proj_head     = SigLIPProjectionHead(cfg)

        init_val = -math.log(cfg.training.temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_val)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """L2-normalized 이미지 임베딩 (추론용)."""
        return F.normalize(self.image_encoder(pixel_values), dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """L2-normalized 텍스트 임베딩 (추론용)."""
        return F.normalize(self.text_encoder(input_ids, attention_mask), dim=-1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_projections: bool = True,
    ):
        """
        Returns:
            return_projections=False : (image_embeds, text_embeds, logit_scale)
            return_projections=True  : 위 3개 + (img_proj_sig, txt_proj_sig)
        """
        image_feats  = self.image_encoder(pixel_values)
        text_feats   = self.text_encoder(input_ids, attention_mask)

        image_embeds = F.normalize(image_feats, dim=-1)
        text_embeds  = F.normalize(text_feats,  dim=-1)
        logit_scale  = self.logit_scale.exp()

        if not return_projections:
            return image_embeds, text_embeds, logit_scale

        img_proj_sig = self.proj_head(image_feats)
        txt_proj_sig = self.proj_head(text_feats)

        return image_embeds, text_embeds, logit_scale, img_proj_sig, txt_proj_sig

    def freeze_text_encoder(self):
        self.text_encoder.freeze()

    def unfreeze_text_encoder(self):
        self.text_encoder.unfreeze()


def build_student_model(cfg: ConfigNode) -> StudentCLIP:
    return StudentCLIP(cfg)
