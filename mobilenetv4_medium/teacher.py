"""
Teacher model loaders for LPCVC 2026 Track 1.
학습 시에만 사용 — 추론/제출 시 불러오지 않습니다.

Teacher 1: SigLIP2 (HuggingFace transformers)
Teacher 2: DFN ViT-H-14 (open_clip)

모든 모델명/크기는 cfg 에서 읽습니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from utils.config import ConfigNode


# ---------------------------------------------------------------------------
# SigLIP2 Teacher
# ---------------------------------------------------------------------------

class SigLIP2Teacher(nn.Module):
    """SigLIP2 스승 모델 (완전 동결)."""

    def __init__(self, cfg: ConfigNode, device: str = "cpu"):
        super().__init__()
        model_name = cfg.model.siglip2_model_name
        try:
            from transformers import SiglipModel
            self.model = SiglipModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"SigLIP2 로드 실패 ({model_name}). "
                f"transformers>=4.39 설치 여부 확인. 오류: {e}"
            )
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.get_image_features(pixel_values=pixel_values), dim=-1)

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
            dim=-1,
        )


# ---------------------------------------------------------------------------
# DFN Teacher
# ---------------------------------------------------------------------------

class DFNTeacher(nn.Module):
    """DFN ViT-H-14 스승 모델 (완전 동결, open_clip)."""

    def __init__(self, cfg: ConfigNode, device: str = "cpu"):
        super().__init__()
        try:
            import open_clip
            self.model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14-378-quickgelu",
                pretrained="dfn5b",
                device=device,
            )
        except Exception as e:
            raise RuntimeError(
                f"DFN 로드 실패. open_clip_torch 설치 여부 확인. 오류: {e}"
            )
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(pixel_values), dim=-1)

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.normalize(self.model.encode_text(input_ids), dim=-1)


# ---------------------------------------------------------------------------
# Teacher Manager
# ---------------------------------------------------------------------------

class TeacherManager:
    """두 스승 모델을 통합 관리하는 매니저."""

    def __init__(
        self,
        cfg: ConfigNode,
        device: str = "cuda",
        load_siglip: bool = True,
        load_dfn: bool = True,
    ):
        self.device = device
        self.siglip2: Optional[SigLIP2Teacher] = None
        self.dfn:     Optional[DFNTeacher]     = None

        if load_siglip:
            print(f"[Teacher] SigLIP2 로드: {cfg.model.siglip2_model_name}")
            self.siglip2 = SigLIP2Teacher(cfg, device=device).to(device)

        if load_dfn:
            print(f"[Teacher] DFN 로드: {cfg.model.dfn_model_name}")
            self.dfn = DFNTeacher(cfg, device=device).to(device)

    def get_siglip_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.siglip2 is not None
        return self.siglip2.encode_image(pixel_values)

    def get_siglip_text_embeds(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        assert self.siglip2 is not None
        return self.siglip2.encode_text(input_ids, attention_mask)

    def get_dfn_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.dfn is not None
        return self.dfn.encode_image(pixel_values)

    def get_dfn_text_embeds(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self.dfn is not None
        return self.dfn.encode_text(input_ids, attention_mask)
