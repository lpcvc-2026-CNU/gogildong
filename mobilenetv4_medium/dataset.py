"""
Dataset and DataLoader utilities for LPCVC 2026 Track 1.

[ 토크나이저 정책 ]
  - 학생 텍스트 인코더 (DistilBERT) → DistilBertTokenizerFast 전용 vocab 사용
  - 스승 SigLIP2 텍스트 → AutoProcessor (teacher.py 내부 처리)
  - 스승 DFN 텍스트    → open_clip tokenizer (teacher.py 내부 처리)

  Dataset 은 DistilBERT 토큰(student_input_ids, student_attention_mask)과
  raw caption 문자열("caption") 을 함께 배치에 포함시킵니다.
  Trainer 에서 스승 모델에 raw text 를 넘기면 각 Teacher 가 자체 토크나이저로 처리합니다.

  모든 수치(이미지 크기, 정규화, batch_size 등)는 cfg 에서 읽습니다.
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
    """학생 모델 학습용 증강 (224×224, ImageNet 정규화)."""
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
    """학생 모델 평가/추론용 결정론적 transform."""
    size = cfg.model.student_image_input_size
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=cfg.data.image_mean, std=cfg.data.image_std),
    ])


def get_siglip_teacher_transform(cfg: ConfigNode) -> T.Compose:
    """SigLIP2 스승용 transform (cfg.data.teacher_image_size)."""
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


def get_dfn_teacher_transform(cfg: ConfigNode) -> T.Compose:
    """DFN 스승용 transform (cfg.model.dfn_input_size)."""
    size = cfg.model.dfn_input_size
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=cfg.data.teacher_mean, std=cfg.data.teacher_std),
    ])


# ---------------------------------------------------------------------------
# Tokenizer (학생 모델 전용)
# ---------------------------------------------------------------------------

class StudentTokenizer:
    """
    학생 모델(DistilBERT + CLIP vocab embedding)용 토크나이저.

    대회 규격에 맞게 openai/clip-vit-base-patch32 CLIPTokenizer 를 사용.
    student TextEncoder 의 word_embedding 이 CLIP vocab(49408) 으로 교체되어 있으므로
    CLIP token ID 를 그대로 넣어도 됩니다.

    스승 모델에는 이 token ID 를 절대 재사용하지 마세요.
    각 스승은 TeacherManager 내부에서 전용 토크나이저를 사용합니다.
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
      - student_image:           (3, 224, 224)      학생 입력
      - student_input_ids:       (student_max_len,) DistilBERT 토큰 ID
      - student_attention_mask:  (student_max_len,) DistilBERT attention mask
      - caption:                 str                raw 문자열 (스승 토크나이저용)
      - teacher_image_sig:       (3, H, H)  [선택] SigLIP2 스승 입력
      - teacher_image_dfn:       (3, H', H')[선택] DFN 스승 입력
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: StudentTokenizer,
        student_transform: Callable,
        teacher_transform_sig: Optional[Callable] = None,
        teacher_transform_dfn: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_path             = Path(data_path)
        self.tokenizer             = tokenizer
        self.student_transform     = student_transform
        self.teacher_transform_sig = teacher_transform_sig
        self.teacher_transform_dfn = teacher_transform_dfn

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
            "caption":       caption,   # raw 문자열 — 스승 토크나이저용
        }
        if self.teacher_transform_sig is not None:
            result["teacher_image_sig"] = self.teacher_transform_sig(image)
        if self.teacher_transform_dfn is not None:
            result["teacher_image_dfn"] = self.teacher_transform_dfn(image)
        return result

    def collate_fn(self, batch: List[dict]) -> dict:
        """
        배치 collation: 학생 텍스트는 DistilBERT 토크나이저로 처리.
        caption raw text 도 유지 (스승 re-tokenization 용).
        """
        captions  = [item["caption"] for item in batch]
        tokenized = self.tokenizer(captions)

        result = {
            "student_image":          torch.stack([item["student_image"] for item in batch]),
            "student_input_ids":      tokenized["input_ids"],      # DistilBERT 토큰
            "student_attention_mask": tokenized["attention_mask"],  # DistilBERT mask
            "caption":                captions,  # List[str] — 스승 모델 토크나이징용
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
    """
    DataLoader 생성 팩토리. 모든 수치는 cfg 에서 읽습니다.

    Args:
        data_path:              데이터셋 경로
        tokenizer:              StudentTokenizer 인스턴스
        cfg:                    ConfigNode (config.yaml)
        is_train:               True → 증강 transform 사용
        use_teacher_transforms: 스승 이미지 transform 포함 여부 (Stage 1~2)
        max_samples:            디버깅용 샘플 수 제한
    """
    student_transform = (
        get_student_train_transform(cfg) if is_train
        else get_student_eval_transform(cfg)
    )
    teacher_sig = get_siglip_teacher_transform(cfg) if use_teacher_transforms else None
    teacher_dfn = get_dfn_teacher_transform(cfg)    if use_teacher_transforms else None

    dataset = ImageCaptionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        student_transform=student_transform,
        teacher_transform_sig=teacher_sig,
        teacher_transform_dfn=teacher_dfn,
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
