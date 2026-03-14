"""
Student CLIP model for LPCVC 2026 Track 1.

[ 토크나이저 정책 - 중요 ]
대회 평가 규격: openai/clip-vit-base-patch32 로 tokenize 한 (1×77) 정수 텐서 입력.
DistilBERT 기본 vocab(30522)은 이 토큰 ID를 수용하지 못함.
→ TextEncoder 초기화 시 word_embeddings 레이어를 CLIP vocab 크기(49408)로 교체.
  나머지 DistilBERT 가중치(Attention, FFN)는 사전학습 값 유지.
  교체된 embedding만 학습 중 랜덤 초기화 후 fine-tune.
"""

import math
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
    MobileNetV4-Small + Linear → shared embed_dim.

    Input : (B, 3, 224, 224)
    Output: (B, embed_dim)  unnormalized
    """

    def __init__(self, cfg: ConfigNode, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model.student_image_backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.proj = nn.Linear(self.backbone.num_features, cfg.model.embed_dim, bias=False)
        self.norm = nn.LayerNorm(cfg.model.embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(self.backbone(pixel_values)))


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    DistilBERT-Base + Linear → shared embed_dim.

    대회 규격 대응:
      - CLIP tokenizer(vocab=49408) 출력 token ID 를 입력으로 받음.
      - DistilBERT 의 word_embeddings 를 Embedding(49408, 768) 로 교체.
      - 나머지 DistilBERT 가중치(Transformer layers)는 사전학습 값 유지.

    Input : input_ids (B, 77), attention_mask (B, 77)
    Output: (B, embed_dim)  unnormalized
    """

    def __init__(self, cfg: ConfigNode, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.distilbert = DistilBertModel.from_pretrained(cfg.model.student_text_backbone)
        else:
            self.distilbert = DistilBertModel(DistilBertConfig())

        hidden_dim = self.distilbert.config.hidden_size  # 768

        # CLIP vocab 크기로 word embedding 교체
        # DistilBERT position/layer norm 등은 token ID 독립적이므로 그대로 사용 가능
        clip_vocab_size = cfg.model.clip_vocab_size  # 49408
        if self.distilbert.config.vocab_size != clip_vocab_size:
            self.distilbert.embeddings.word_embeddings = nn.Embedding(
                clip_vocab_size, hidden_dim, padding_idx=1  # CLIP pad_token_id=1
            )
            # Xavier 초기화 (랜덤 초기화보다 안정적)
            nn.init.xavier_uniform_(self.distilbert.embeddings.word_embeddings.weight)

        self.proj = nn.Linear(hidden_dim, cfg.model.embed_dim, bias=False)
        self.norm = nn.LayerNorm(cfg.model.embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs    = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]   # [CLS] 토큰
        return self.norm(self.proj(cls_output))

    def freeze(self):
        """DistilBERT 가중치 동결 (Stage 1 용)."""
        for p in self.distilbert.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """DistilBERT 가중치 해제 (Stage 2+ 용)."""
        for p in self.distilbert.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Student CLIP
# ---------------------------------------------------------------------------

class StudentCLIP(nn.Module):
    """
    Full student model: ImageEncoder + TextEncoder + DualProjectionHeads.

    Projection heads 역할:
      head_sig : student embed → SigLIP2 공간  (KL 증류 + MSE feature mimicking)
      head_dfn : student embed → DFN 공간      (KL 증류 + MSE feature mimicking)

    추론 시에는 shared L2-normalized embed 만 사용.
    Projection head 는 학습 시에만 활성화.
    """

    def __init__(self, cfg: ConfigNode):
        super().__init__()
        self.cfg = cfg

        self.image_encoder    = ImageEncoder(cfg)
        self.text_encoder     = TextEncoder(cfg)
        self.image_proj_heads = DualProjectionHeads(cfg)
        self.text_proj_heads  = DualProjectionHeads(cfg)

        # 학습 가능한 온도 (초기값: -ln(temperature))
        init_val = -math.log(cfg.training.temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_val)

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """L2-normalized 이미지 임베딩 (추론용)."""
        return F.normalize(self.image_encoder(pixel_values), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
            return_projections=True  : 위 3개 + (img_proj_sig, img_proj_dfn,
                                                    txt_proj_sig, txt_proj_dfn)
        """
        image_feats  = self.image_encoder(pixel_values)
        text_feats   = self.text_encoder(input_ids, attention_mask)

        image_embeds = F.normalize(image_feats, dim=-1)
        text_embeds  = F.normalize(text_feats,  dim=-1)
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
