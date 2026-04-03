"""
Teacher model loader for LPCVC 2026 Track 1 (Single Teacher: SigLIP2).
학습 시에만 사용 — 추론/제출 시 로드하지 않습니다.

변경 이력
─────────
- TeacherManager.build_embedding_cache() 추가
  : 학습 전 전체 데이터셋에 대한 SigLIP2 이미지/텍스트 임베딩을 사전 계산하고
    디스크에 캐싱합니다. 이후 teacher 모델은 GPU에서 해제됩니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Any

from config import ConfigNode


class SigLIP2Teacher(nn.Module):
    """
    SigLIP2 스승 모델 (frozen).

    이미지: 전처리된 pixel_values tensor 를 직접 받음.
    텍스트: raw 문자열 List → 내부 AutoProcessor 로 토크나이징.
    """

    def __init__(self, cfg: ConfigNode):
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

    def _as_feature_tensor(self, outputs: Any, modality: str) -> torch.Tensor:
        """
        Normalize HF model outputs across SigLIP/SigLIP2 variants.
        Some versions return a Tensor directly, others return a model output object.
        """
        if isinstance(outputs, torch.Tensor):
            return outputs

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            return outputs.text_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0, :]
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            first = outputs[0]
            if isinstance(first, torch.Tensor):
                return first

        raise TypeError(
            f"Unexpected {modality} feature output type: {type(outputs)}. "
            "Could not extract a Tensor."
        )

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, siglip2_input_size, siglip2_input_size)
        Returns:
            (B, siglip_teacher_dim) — L2-normalized
        """
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        image_features = self._as_feature_tensor(image_features, modality="image")
        return F.normalize(image_features, dim=-1)

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
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        text_features = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = self._as_feature_tensor(text_features, modality="text")
        return F.normalize(text_features, dim=-1)


class TeacherManager:
    """SigLIP2 단일 스승 모델 관리 및 임베딩 캐시 빌드."""

    def __init__(self, cfg: ConfigNode, device: str = "cuda"):
        self.device  = device
        self.cfg     = cfg
        print(f"[Teacher] SigLIP2 로드: {cfg.model.siglip2_model_name}")
        self.siglip2 = SigLIP2Teacher(cfg).to(device)

    # ── 단건 임베딩 (캐시 빌드 시 내부 사용) ──────────────────────────────

    @torch.no_grad()
    def get_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.siglip2.encode_image(pixel_values)

    @torch.no_grad()
    def get_text_embeds(self, texts: List[str]) -> torch.Tensor:
        return self.siglip2.encode_text(texts)

    # ── 전체 데이터셋 임베딩 사전 계산 ────────────────────────────────────

    @torch.no_grad()
    def build_embedding_cache(
        self,
        cache_dataloader,          # shuffle=False, use_teacher_transforms=True 로 빌드된 DataLoader
        cache_path: str = "siglip2_cache.pt",
        fp16: bool = True,         # 캐시를 fp16 으로 저장해 디스크/메모리 절약
    ) -> dict:
        """
        전체 데이터셋에 대한 SigLIP2 이미지·텍스트 임베딩을 사전 계산합니다.
        캐시 파일이 이미 존재하면 로드만 하고 바로 반환합니다.

        캐시 파일 구조:
            {
                "img": Tensor (N, embed_dim),   # L2-normalized, fp16
                "txt": Tensor (N, embed_dim),   # L2-normalized, fp16
            }

        학습 루프에서 batch["sample_idx"] 를 인덱스로 사용해 슬라이싱합니다.

        Args:
            cache_dataloader : shuffle=False DataLoader (sample_idx 순서 보장 필수)
            cache_path       : 캐시 저장/로드 경로
            fp16             : True 이면 float16 으로 저장 (약 절반 디스크 사용)

        Returns:
            {"img": Tensor(N, D), "txt": Tensor(N, D)}  — CPU 에 저장됨
        """
        cache_file = Path(cache_path)

        # ── 캐시 존재 시 바로 반환 ──────────────────────────────────────────
        if cache_file.exists():
            print(f"[Cache] 기존 캐시 로드: {cache_file}")
            cache = torch.load(cache_file, map_location="cpu")
            n     = cache["img"].shape[0]
            dim   = cache["img"].shape[1]
            dtype = cache["img"].dtype
            print(f"  └─ {n:,}개 샘플 | dim={dim} | dtype={dtype}")
            return cache

        # ── 신규 캐시 계산 ──────────────────────────────────────────────────
        dataset_size = len(cache_dataloader.dataset)
        print(
            f"[Cache] SigLIP2 임베딩 사전 계산 시작\n"
            f"  └─ 총 {dataset_size:,}개 샘플 | fp16={fp16}"
        )

        # embedding dim 동적 탐지 (dummy forward)
        dummy_txt = self.siglip2.encode_text(["a"])
        embed_dim = dummy_txt.shape[-1]
        del dummy_txt

        store_dtype = torch.float16 if fp16 else torch.float32
        img_cache   = torch.zeros(dataset_size, embed_dim, dtype=store_dtype)
        txt_cache   = torch.zeros(dataset_size, embed_dim, dtype=store_dtype)

        processed = 0
        log_interval = max(1, len(cache_dataloader) // 20)  # 약 5% 마다 로그

        for batch_idx, batch in enumerate(cache_dataloader):
            indices  = batch["sample_idx"]          # (B,) LongTensor
            captions = batch["caption"]             # List[str]

            # 이미지 임베딩
            if "teacher_image_sig" in batch:
                sig_imgs = batch["teacher_image_sig"].to(self.device)
                img_embs = self.get_image_embeds(sig_imgs).cpu().to(store_dtype)
            else:
                # teacher transform 없이 빌드된 경우 student 이미지로 fallback
                # (권장 X — use_teacher_transforms=True 로 dataloader 빌드할 것)
                raise RuntimeError(
                    "캐시 빌드용 DataLoader 에 'teacher_image_sig' 가 없습니다.\n"
                    "build_dataloader(..., use_teacher_transforms=True) 로 생성하세요."
                )

            # 텍스트 임베딩
            txt_embs = self.get_text_embeds(captions).cpu().to(store_dtype)

            # 인덱스 기반 저장 (shuffle=False 이면 순서 보장, 만약을 위해 인덱스 사용)
            img_cache[indices] = img_embs
            txt_cache[indices] = txt_embs

            processed += len(indices)
            if (batch_idx + 1) % log_interval == 0 or processed == dataset_size:
                pct = processed / dataset_size * 100
                print(f"  [{processed:>7,} / {dataset_size:,}]  {pct:5.1f}% 완료")

        cache = {"img": img_cache, "txt": txt_cache}

        # ── 디스크 저장 ─────────────────────────────────────────────────────
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_file)
        size_mb = cache_file.stat().st_size / 1024 ** 2
        print(
            f"[Cache] 저장 완료: {cache_file}\n"
            f"  └─ {dataset_size:,}개 | dim={embed_dim} | {size_mb:.1f} MB"
        )
        return cache

    def offload(self):
        """캐시 빌드 완료 후 GPU 메모리 해제."""
        del self.siglip2
        torch.cuda.empty_cache()
        print("[Teacher] GPU 메모리 해제 완료 (캐시 빌드 후 teacher 오프로드)")
