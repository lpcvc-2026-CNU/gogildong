"""
Teacher model loaders for LPCVC 2026 Track 1.
학습 시에만 사용 — 추론/제출 시 로드하지 않습니다.

핵심 원칙: 각 스승 모델은 자신의 전용 토크나이저를 직접 보유합니다.
  - SigLIP2Teacher : AutoProcessor (HuggingFace 전용 processor)
  - DFNTeacher     : open_clip.get_tokenizer (CLIP-style BPE)

trainer.py 에서 raw caption 문자열을 넘기면, 각 스승이 자체 토크나이저로 처리합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from utils.config import ConfigNode


# ---------------------------------------------------------------------------
# SigLIP2 Teacher
# ---------------------------------------------------------------------------

class SigLIP2Teacher(nn.Module):
    """
    SigLIP2 스승 모델.
    텍스트 입력: raw 문자열 List → 내부 AutoProcessor 로 토크나이징
    이미지 입력: 이미 전처리된 pixel_values tensor
    """

    def __init__(self, cfg: ConfigNode, device: str = "cpu"):
        super().__init__()
        model_name = cfg.model.siglip2_model_name
        try:
            from transformers import SiglipModel, AutoProcessor
            self.model     = SiglipModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"SigLIP2 로드 실패 ({model_name}). "
                f"transformers>=4.39 설치 여부 확인. 오류: {e}"
            )
        self._device = device
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, siglip2_input_size, siglip2_input_size) — 전처리 완료
        Returns:
            (B, siglip_teacher_dim) — L2-normalized
        """
        return F.normalize(
            self.model.get_image_features(pixel_values=pixel_values),
            dim=-1,
        )

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: raw 캡션 문자열 리스트 (SigLIP2 전용 processor 로 토크나이징)
        Returns:
            (B, siglip_teacher_dim) — L2-normalized
        """
        device = self._get_device()
        # SigLIP2 processor: padding/truncation 은 모델 기본값 사용
        encoded = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        return F.normalize(
            self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ),
            dim=-1,
        )


# ---------------------------------------------------------------------------
# DFN Teacher
# ---------------------------------------------------------------------------

class DFNTeacher(nn.Module):
    """
    DFN ViT-H-14 스승 모델 (open_clip).
    텍스트 입력: raw 문자열 List → 내부 open_clip tokenizer 로 토크나이징
    """

    def __init__(self, cfg: ConfigNode, device: str = "cpu"):
        super().__init__()
        try:
            import open_clip
            self.model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14-378-quickgelu",
                pretrained="dfn5b",
                device=device,
            )
            # open_clip 전용 CLIP-style BPE 토크나이저
            self.tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
        except Exception as e:
            raise RuntimeError(
                f"DFN 로드 실패. open_clip_torch 설치 여부 확인. 오류: {e}"
            )
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, dfn_input_size, dfn_input_size) — 전처리 완료
        Returns:
            (B, dfn_teacher_dim) — L2-normalized
        """
        return F.normalize(self.model.encode_image(pixel_values), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: raw 캡션 문자열 리스트 (open_clip BPE tokenizer 로 처리)
        Returns:
            (B, dfn_teacher_dim) — L2-normalized
        """
        device    = self._get_device()
        token_ids = self.tokenizer(texts).to(device)   # (B, context_length)
        return F.normalize(self.model.encode_text(token_ids), dim=-1)


# ---------------------------------------------------------------------------
# Teacher Manager
# ---------------------------------------------------------------------------

class TeacherManager:
    """두 스승 모델을 통합 관리. Trainer 에서 raw text 를 넘기면 됩니다."""

    def __init__(
        self,
        cfg: ConfigNode,
        device: str = "cuda",
        load_siglip: bool = True,
        load_dfn: bool = True,
    ):
        self.device   = device
        self.siglip2: Optional[SigLIP2Teacher] = None
        self.dfn:     Optional[DFNTeacher]     = None

        if load_siglip:
            print(f"[Teacher] SigLIP2 로드: {cfg.model.siglip2_model_name}")
            self.siglip2 = SigLIP2Teacher(cfg, device=device).to(device)

        if load_dfn:
            print(f"[Teacher] DFN 로드: {cfg.model.dfn_model_name}")
            self.dfn = DFNTeacher(cfg, device=device).to(device)

    # ---- Image embeddings ----

    def get_siglip_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.siglip2 is not None, "SigLIP2 teacher 미로드"
        return self.siglip2.encode_image(pixel_values)

    def get_dfn_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.dfn is not None, "DFN teacher 미로드"
        return self.dfn.encode_image(pixel_values)

    # ---- Text embeddings (raw text → 각 모델 전용 토크나이저 사용) ----

    def get_siglip_text_embeds(self, texts: List[str]) -> torch.Tensor:
        """SigLIP2 전용 processor 로 텍스트를 토크나이징 후 인코딩."""
        assert self.siglip2 is not None, "SigLIP2 teacher 미로드"
        return self.siglip2.encode_text(texts)

    def get_dfn_text_embeds(self, texts: List[str]) -> torch.Tensor:
        """open_clip BPE 토크나이저로 텍스트를 토크나이징 후 인코딩."""
        assert self.dfn is not None, "DFN teacher 미로드"
        return self.dfn.encode_text(texts)
