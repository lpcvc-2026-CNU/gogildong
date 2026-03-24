"""
Dataset and DataLoader utilities for LPCVC 2026 Track 1 (Single Teacher: SigLIP2).

토크나이저 정책:
  - 학생 텍스트 인코더 → CLIPTokenizer (vocab=49408, max_len=77)
  - SigLIP2 스승 텍스트 → AutoProcessor (teacher.py 내부 처리)

Dataset 은 student 토큰(student_input_ids, student_attention_mask)과
raw caption 문자열("caption") 을 함께 배치에 포함시킵니다.
"""

import os
import csv
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from transformers import CLIPTokenizer

from utils.config import ConfigNode


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_student_train_transform(cfg: ConfigNode) -> T.Compose:
    aug  = cfg.data.augmentation
    size = cfg.model.student_image_input_size
    return T.Compose([
        T.RandomResizedCrop(
            size,
            scale=(aug.random_crop_scale_min, aug.random_crop_scale_max),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
        T.ColorJitter(
            brightness=aug.color_jitter_brightness,
            contrast=aug.color_jitter_contrast,
            saturation=aug.color_jitter_saturation,
            hue=aug.color_jitter_hue,
        ),
        T.ToTensor(),
        T.Normalize(mean=cfg.data.image_mean, std=cfg.data.image_std),
    ])


def get_student_eval_transform(cfg: ConfigNode) -> T.Compose:
    size = cfg.model.student_image_input_size
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=cfg.data.image_mean, std=cfg.data.image_std),
    ])


def get_siglip_teacher_transform(cfg: ConfigNode) -> T.Compose:
    """SigLIP2 스승용 transform."""
    size = cfg.data.teacher_image_size
    aug  = cfg.data.augmentation
    return T.Compose([
        T.RandomResizedCrop(
            size,
            scale=(aug.random_crop_scale_min, aug.random_crop_scale_max),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=cfg.data.teacher_mean, std=cfg.data.teacher_std),
    ])


# ---------------------------------------------------------------------------
# Tokenizer (학생 전용)
# ---------------------------------------------------------------------------

class StudentTokenizer:
    """
    학생 모델용 토크나이저 (대회 규격: openai/clip-vit-base-patch32, max_len=77).
    스승 모델에는 절대 재사용하지 마세요.
    """

    def __init__(self, cfg: ConfigNode):
        self.tokenizer  = CLIPTokenizer.from_pretrained(cfg.model.clip_tokenizer_name)
        self.max_length = cfg.model.max_text_length

    def __call__(self, texts: List[str]) -> dict:
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageCaptionDataset(Dataset):
    """
    이미지-캡션 쌍 데이터셋.

    data_path 에 captions.csv 또는 captions.tsv 가 있어야 합니다.
    형식: image_path,caption

    반환 dict:
      - student_image:           (3, 224, 224)
      - student_input_ids:       (77,)  — CLIP token IDs
      - student_attention_mask:  (77,)
      - caption:                 str    — raw 문자열 (SigLIP2 스승용)
      - teacher_image_sig:       (3, H, H)  [선택] SigLIP2 스승 입력
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: StudentTokenizer,
        student_transform: Callable,
        teacher_transform_sig: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path             = Path(data_path)
        self.tokenizer             = tokenizer
        self.student_transform     = student_transform
        self.teacher_transform_sig = teacher_transform_sig

        self.samples = self._load_samples(max_samples)
        print(f"[Dataset] {len(self.samples)} 샘플 로드 완료: '{data_path}'")

    def _load_samples(self, max_samples: Optional[int]) -> List[Tuple[str, str]]:
        csv_path = self.data_path / "captions.csv"
        tsv_path = self.data_path / "captions.tsv"

        if csv_path.exists():
            delimiter, target = ",", csv_path
        elif tsv_path.exists():
            delimiter, target = "\t", tsv_path
        else:
            raise FileNotFoundError(
                f"captions.csv 또는 captions.tsv 없음: {self.data_path}"
            )

        samples = []
        with open(target, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=delimiter):
                img_path = row.get("image_path") or row.get("filepath", "")
                caption  = row.get("caption")   or row.get("text", "")
                if not img_path or not caption:
                    continue
                if not os.path.isabs(img_path):
                    img_path = str(self.data_path / img_path)
                samples.append((img_path, caption))

        return samples[:max_samples] if max_samples else samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, caption = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=0)

        result = {
            "student_image": self.student_transform(image),
            "caption":       caption,
        }
        if self.teacher_transform_sig is not None:
            result["teacher_image_sig"] = self.teacher_transform_sig(image)
        return result

    def collate_fn(self, batch: List[dict]) -> dict:
        """
        배치 collation: 학생 텍스트는 StudentTokenizer 로 처리.
        caption raw text 도 유지 (SigLIP2 스승 re-tokenization 용).
        """
        captions  = [item["caption"] for item in batch]
        tokenized = self.tokenizer(captions)

        result = {
            "student_image":          torch.stack([item["student_image"] for item in batch]),
            "student_input_ids":      tokenized["input_ids"],
            "student_attention_mask": tokenized["attention_mask"],
            "caption":                captions,
        }
        if "teacher_image_sig" in batch[0]:
            result["teacher_image_sig"] = torch.stack(
                [item["teacher_image_sig"] for item in batch]
            )
        return result


# ---------------------------------------------------------------------------
# DataLoader 팩토리
# ---------------------------------------------------------------------------

def build_dataloader(
    data_path: str,
    tokenizer: StudentTokenizer,
    cfg: ConfigNode,
    is_train: bool = True,
    use_teacher_transforms: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    student_transform = (
        get_student_train_transform(cfg) if is_train
        else get_student_eval_transform(cfg)
    )
    teacher_sig = get_siglip_teacher_transform(cfg) if use_teacher_transforms else None

    dataset = ImageCaptionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        student_transform=student_transform,
        teacher_transform_sig=teacher_sig,
        max_samples=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=is_train,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=dataset.collate_fn,
    )
