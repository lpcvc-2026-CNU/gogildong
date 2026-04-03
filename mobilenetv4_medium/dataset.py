"""
Dataset and DataLoader utilities.

Supported dataset layouts:
1) CSV/TSV captions in a directory
2) COCO captions layout

변경 이력
─────────
- __getitem__ 에 sample_idx 추가 (teacher 임베딩 캐시 인덱싱용)
- collate_fn 에 sample_idx 스택 추가
- build_dataloader 에 shuffle 파라미터 추가 (캐시 빌드 시 shuffle=False 강제)
"""

import csv
import json
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPTokenizer

from config import ConfigNode


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
    """SigLIP2 teacher 전용 이미지 변환 (384×384, [-1, 1] 정규화)."""
    size = cfg.data.teacher_image_size
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=cfg.data.teacher_mean, std=cfg.data.teacher_std),
    ])


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class StudentTokenizer:
    """CLIP tokenizer (고정 규격: openai/clip-vit-base-patch32, max_len=77)."""

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
    이미지-캡션 데이터셋. CSV/TSV 및 COCO JSON 지원.

    __getitem__ 반환 키:
        student_image       (3, H, W)
        caption             str
        sample_idx          int  ← teacher 임베딩 캐시 인덱싱용 (신규)
        teacher_image_sig   (3, 384, 384)  — use_teacher_transform=True 일 때만
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: StudentTokenizer,
        student_transform: Callable,
        teacher_transform_sig: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        split: str = "train",
        sample_ratio: float = 1.0,
        sample_seed: int = 42,
    ):
        self.data_path            = Path(data_path)
        self.tokenizer            = tokenizer
        self.student_transform    = student_transform
        self.teacher_transform_sig = teacher_transform_sig
        self.split                = split

        self.samples = self._load_samples(
            max_samples=max_samples,
            sample_ratio=sample_ratio,
            sample_seed=sample_seed,
        )
        print(
            f"[Dataset] split={self.split}, samples={len(self.samples):,} "
            f"loaded from '{self.data_path}'"
        )

    # ── 샘플 로딩 ────────────────────────────────────────────────────────────

    def _load_samples(self, max_samples, sample_ratio, sample_seed) -> List[Tuple[str, str]]:
        samples = self._load_samples_from_csv_or_tsv()
        if not samples:
            samples = self._load_samples_from_coco()

        if not samples:
            raise FileNotFoundError(
                "캡션 데이터를 찾을 수 없습니다. 다음 경로 중 하나가 있어야 합니다:\n"
                f"- {self.data_path / 'captions.csv'}\n"
                f"- {self.data_path / 'captions.tsv'}\n"
                f"- {self.data_path / 'annotations' / 'captions_train2017.json'}"
            )

        if max_samples is not None:
            return samples[:max_samples]

        if self.split == "train" and 0.0 < sample_ratio < 1.0:
            subset_count = max(1, int(len(samples) * sample_ratio))
            rng = random.Random(sample_seed)
            rng.shuffle(samples)
            samples = samples[:subset_count]

        return samples

    def _load_samples_from_csv_or_tsv(self) -> List[Tuple[str, str]]:
        csv_path = self.data_path / "captions.csv"
        tsv_path = self.data_path / "captions.tsv"

        if csv_path.exists():
            delimiter, target = ",", csv_path
        elif tsv_path.exists():
            delimiter, target = "\t", tsv_path
        else:
            return []

        samples: List[Tuple[str, str]] = []
        with open(target, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=delimiter):
                img_path = row.get("image_path") or row.get("filepath", "")
                caption  = row.get("caption") or row.get("text", "")
                if not img_path or not caption:
                    continue
                if not os.path.isabs(img_path):
                    img_path = str(self.data_path / img_path)
                samples.append((img_path, caption.strip()))
        return samples

    def _load_samples_from_coco(self) -> List[Tuple[str, str]]:
        split_name = "train2017" if self.split == "train" else "val2017"
        ann_name   = (
            "captions_train2017.json" if self.split == "train"
            else "captions_val2017.json"
        )
        ann_path  = self.data_path / "annotations" / ann_name
        image_dir = self.data_path / split_name

        if not ann_path.exists() or not image_dir.exists():
            return []

        with open(ann_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        id_to_name = {
            int(img["id"]): img["file_name"]
            for img in payload.get("images", [])
        }

        samples: List[Tuple[str, str]] = []
        for ann in payload.get("annotations", []):
            image_id  = ann.get("image_id")
            caption   = (ann.get("caption") or "").strip()
            file_name = id_to_name.get(int(image_id)) if image_id is not None else None
            if not file_name or not caption:
                continue
            samples.append((str(image_dir / file_name), caption))
        return samples

    # ── Dataset interface ────────────────────────────────────────────────────

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
            "sample_idx":    idx,          # ← teacher 캐시 인덱싱용
        }
        if self.teacher_transform_sig is not None:
            result["teacher_image_sig"] = self.teacher_transform_sig(image)
        return result

    def collate_fn(self, batch: List[dict]) -> dict:
        captions   = [item["caption"] for item in batch]
        tokenized  = self.tokenizer(captions)

        result = {
            "student_image":          torch.stack([item["student_image"] for item in batch]),
            "student_input_ids":      tokenized["input_ids"],
            "student_attention_mask": tokenized["attention_mask"],
            "caption":                captions,
            # teacher 캐시 인덱싱용 — 항상 포함
            "sample_idx":             torch.tensor(
                                          [item["sample_idx"] for item in batch],
                                          dtype=torch.long,
                                      ),
        }
        if "teacher_image_sig" in batch[0]:
            result["teacher_image_sig"] = torch.stack(
                [item["teacher_image_sig"] for item in batch]
            )
        return result


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    data_path: str,
    tokenizer: StudentTokenizer,
    cfg: ConfigNode,
    is_train: bool = True,
    use_teacher_transforms: bool = True,
    max_samples: Optional[int] = None,
    shuffle: Optional[bool] = None,          # ← 신규: None 이면 is_train 값을 따름
) -> DataLoader:
    """
    Args:
        shuffle: 명시적으로 지정 시 is_train 과 무관하게 적용.
                 None 이면 is_train=True → shuffle=True, is_train=False → shuffle=False.
                 캐시 빌드용 DataLoader 는 shuffle=False 로 명시해야 인덱스가 일치함.
    """
    student_transform = (
        get_student_train_transform(cfg) if is_train
        else get_student_eval_transform(cfg)
    )
    teacher_sig = get_siglip_teacher_transform(cfg) if use_teacher_transforms else None

    split = "train" if is_train else "val"
    train_subset_ratio = cfg.data.get("train_subset_ratio", 1.0)

    dataset = ImageCaptionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        student_transform=student_transform,
        teacher_transform_sig=teacher_sig,
        max_samples=max_samples,
        split=split,
        sample_ratio=train_subset_ratio,
        sample_seed=cfg.training.seed,
    )

    actual_shuffle = shuffle if shuffle is not None else is_train

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=actual_shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=is_train,
        collate_fn=dataset.collate_fn,
    )
