"""
Dataset and DataLoader utilities for LPCVC 2026 Track 1.

모든 수치(이미지 크기, 정규화 파라미터, batch_size 등)는
config.yaml → cfg 객체에서 읽어옵니다.
하드코딩 없음.
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
# Transforms (모든 수치는 cfg 에서 주입받음)
# ---------------------------------------------------------------------------

def get_student_train_transform(cfg: ConfigNode) -> T.Compose:
    """학생 모델 학습용 증강 transform (224×224)."""
    aug = cfg.data.augmentation
    size = cfg.model.student_image_input_size
    mean = cfg.data.image_mean
    std  = cfg.data.image_std
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
        T.Normalize(mean=mean, std=std),
    ])


def get_student_eval_transform(cfg: ConfigNode) -> T.Compose:
    """학생 모델 평가/추론용 결정론적 transform (224×224)."""
    size = cfg.model.student_image_input_size
    mean = cfg.data.image_mean
    std  = cfg.data.image_std
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_siglip_teacher_transform(cfg: ConfigNode) -> T.Compose:
    """SigLIP2 스승 모델 transform (cfg.data.teacher_image_size)."""
    size = cfg.data.teacher_image_size
    aug  = cfg.data.augmentation
    mean = cfg.data.teacher_mean
    std  = cfg.data.teacher_std
    return T.Compose([
        T.RandomResizedCrop(
            size,
            scale=(aug.random_crop_scale_min, aug.random_crop_scale_max),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_dfn_teacher_transform(cfg: ConfigNode) -> T.Compose:
    """DFN 스승 모델 transform (cfg.model.dfn_input_size)."""
    size = cfg.model.dfn_input_size
    mean = cfg.data.teacher_mean
    std  = cfg.data.teacher_std
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class CLIPTextTokenizer:
    """
    CLIP 토크나이저 래퍼.
    토크나이저 이름과 최대 길이는 cfg 에서 읽습니다.
    """

    def __init__(self, cfg: ConfigNode):
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model.tokenizer_name)
        self.max_length = cfg.model.max_text_length

    def __call__(self, texts: List[str]) -> dict:
        """
        Args:
            texts: 문자열 리스트
        Returns:
            {'input_ids': (B, max_length), 'attention_mask': (B, max_length)}
        """
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageCaptionDataset(Dataset):
    """
    이미지-캡션 쌍 데이터셋.

    data_path 디렉토리에 captions.csv 또는 captions.tsv 가 있어야 합니다.
    CSV 형식: image_path,caption  (또는 filepath,text)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: CLIPTextTokenizer,
        student_transform: Callable,
        teacher_transform_sig: Optional[Callable] = None,
        teacher_transform_dfn: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.student_transform = student_transform
        self.teacher_transform_sig = teacher_transform_sig
        self.teacher_transform_dfn = teacher_transform_dfn

        self.samples = self._load_samples(max_samples)
        print(f"[Dataset] {len(self.samples)} samples from '{data_path}'")

    def _load_samples(self, max_samples: Optional[int]) -> List[Tuple[str, str]]:
        """(image_path, caption) 리스트 로드."""
        csv_path = self.data_path / "captions.csv"
        tsv_path = self.data_path / "captions.tsv"

        if csv_path.exists():
            delimiter, target = ",", csv_path
        elif tsv_path.exists():
            delimiter, target = "\t", tsv_path
        else:
            raise FileNotFoundError(
                f"captions.csv 또는 captions.tsv 를 찾을 수 없습니다: {self.data_path}"
            )

        samples = []
        with open(target, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
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
            "caption": caption,
        }
        if self.teacher_transform_sig is not None:
            result["teacher_image_sig"] = self.teacher_transform_sig(image)
        if self.teacher_transform_dfn is not None:
            result["teacher_image_dfn"] = self.teacher_transform_dfn(image)

        return result

    def collate_fn(self, batch: List[dict]) -> dict:
        """배치 내 캡션 토크나이징을 일괄 처리."""
        captions  = [item["caption"] for item in batch]
        tokenized = self.tokenizer(captions)

        result = {
            "student_image":  torch.stack([item["student_image"] for item in batch]),
            "input_ids":      tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "caption":        captions,
        }
        if "teacher_image_sig" in batch[0]:
            result["teacher_image_sig"] = torch.stack(
                [item["teacher_image_sig"] for item in batch]
            )
        if "teacher_image_dfn" in batch[0]:
            result["teacher_image_dfn"] = torch.stack(
                [item["teacher_image_dfn"] for item in batch]
            )
        return result


# ---------------------------------------------------------------------------
# DataLoader 팩토리 (모든 수치는 cfg 에서 읽음)
# ---------------------------------------------------------------------------

def build_dataloader(
    data_path: str,
    tokenizer: CLIPTextTokenizer,
    cfg: ConfigNode,
    is_train: bool = True,
    use_teacher_transforms: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    DataLoader 생성 팩토리.

    Args:
        data_path:              데이터셋 경로
        tokenizer:              CLIPTextTokenizer 인스턴스
        cfg:                    config.yaml 로드 결과 (ConfigNode)
        is_train:               True 이면 증강 transform 사용
        use_teacher_transforms: 스승 이미지 transform 포함 여부 (Stage 1~2 에서만 True)
        max_samples:            디버깅용 샘플 수 제한
    """
    student_transform = (
        get_student_train_transform(cfg) if is_train
        else get_student_eval_transform(cfg)
    )
    teacher_transform_sig = (
        get_siglip_teacher_transform(cfg) if use_teacher_transforms else None
    )
    teacher_transform_dfn = (
        get_dfn_teacher_transform(cfg) if use_teacher_transforms else None
    )

    dataset = ImageCaptionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        student_transform=student_transform,
        teacher_transform_sig=teacher_transform_sig,
        teacher_transform_dfn=teacher_transform_dfn,
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
