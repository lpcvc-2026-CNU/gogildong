"""
Teacher model loader for LPCVC 2026 Track 1 (Single Teacher: SigLIP2).
학습 시에만 사용 — 추론/제출 시 로드하지 않습니다.

SigLIP2Teacher 는 raw caption 문자열을 받아 자체 AutoProcessor 로 처리합니다.
Trainer 에서 raw text 를 넘기면 됩니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from utils.config import ConfigNode


class SigLIP2Teacher(nn.Module):
    """
    SigLIP2 스승 모델 (frozen).

    이미지: 전처리된 pixel_values tensor 를 직접 받음.
    텍스트: raw 문자열 List → 내부 AutoProcessor 로 토크나이징.
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
        self._freeze()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, siglip2_input_size, siglip2_input_size)
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
            texts: raw 캡션 문자열 리스트
        Returns:
            (B, siglip_teacher_dim) — L2-normalized
        """
        device = self._device()
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


class TeacherManager:
    """SigLIP2 단일 스승 모델 관리."""

    def __init__(self, cfg: ConfigNode, device: str = "cuda"):
        self.device  = device
        print(f"[Teacher] SigLIP2 로드: {cfg.model.siglip2_model_name}")
        self.siglip2 = SigLIP2Teacher(cfg, device=device).to(device)

    @torch.no_grad()
    def get_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.siglip2.encode_image(pixel_values)

    @torch.no_grad()
    def get_text_embeds(self, texts: List[str]) -> torch.Tensor:
        return self.siglip2.encode_text(texts)
