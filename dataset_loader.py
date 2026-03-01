"""
공통 이미지–텍스트 데이터셋 로더.
- json: [{"image": "file.jpg", "caption": "..."}, ...] 형식의 단일 JSON 파일
- coco: MS COCO captions 형식 (annotations에 images, annotations 포함)
- csv: image_path,caption(,...) 형식 CSV (캡션/이미지 수 불일치 시 존재하는 쌍만 사용)
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image

log = logging.getLogger(__name__)


def load_coco_captions(coco_annotations_path: str, image_root: str) -> List[Dict[str, str]]:
    """
    MS COCO captions JSON을 읽어 [{"image": "file_name", "caption": "..."}, ...] 리스트로 변환.
    COCO 형식: {"images": [{"id": 1, "file_name": "COCO_train2017_xxx.jpg"}, ...],
                "annotations": [{"image_id": 1, "caption": "..."}, ...]}
    image_root 아래에 file_name이 위치한다고 가정 (예: image_root / "COCO_train2017_xxx.jpg").
    """
    with open(coco_annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    id_to_file: Dict[int, str] = {img['id']: img['file_name'] for img in data['images']}
    pairs: List[Dict[str, str]] = []
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in id_to_file:
            continue
        pairs.append({
            'image': id_to_file[image_id],
            'caption': ann['caption'],
        })
    return pairs


def load_json_captions(json_path: str) -> List[Dict[str, str]]:
    """단순 JSON 리스트 [{"image": "...", "caption": "..."}, ...] 로드."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_captions(
    csv_path: str,
    image_col: str = "image_path",
    caption_col: str = "caption",
) -> List[Dict[str, str]]:
    """
    CSV 파일 로드. 컬럼명은 image_path/image, caption 사용.
    캡션에 쉼표가 있어도 quoted CSV로 처리.
    """
    pairs: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return pairs
        # image_path 또는 image 컬럼 허용
        img_key = "image_path" if "image_path" in reader.fieldnames else "image"
        cap_key = "caption" if "caption" in reader.fieldnames else None
        if not cap_key:
            raise ValueError(f"CSV must have 'caption' column. Got: {reader.fieldnames}")
        for row in reader:
            img = (row.get(img_key) or "").strip()
            cap = (row.get(cap_key) or "").strip()
            if not img or not cap:
                continue
            pairs.append({"image": img, "caption": cap})
    return pairs


def filter_valid_pairs(
    pairs: List[Dict[str, str]],
    image_root: str,
    log_skip: bool = True,
) -> List[Dict[str, str]]:
    """
    이미지 파일이 실제로 존재하고 캡션이 비어 있지 않은 쌍만 남김.
    데이터가 잘려서 개수가 맞지 않을 때 사용.
    """
    root = Path(image_root)
    valid: List[Dict[str, str]] = []
    skipped = 0
    for p in pairs:
        img_path = root / p["image"]
        if not img_path.exists():
            skipped += 1
            continue
        if not (p.get("caption") or "").strip():
            skipped += 1
            continue
        valid.append(p)
    if log_skip and skipped:
        msg = (
            "filter_valid_pairs: kept %d pairs, skipped %d (missing image or empty caption)"
            % (len(valid), skipped)
        )
        log.info(msg)
        print(msg)  # 학습 시 콘솔에서도 확인 가능
    return valid


class ImageTextDataset(Dataset):
    """
    image_root 아래 이미지와 캡션 리스트로 (image, text) 쌍을 반환.
    annotations: [{"image": "상대경로 또는 파일명", "caption": "..."}, ...]
    """

    def __init__(
        self,
        image_root: str,
        annotations: List[Dict[str, str]],
        transform: transforms.Compose,
        tokenizer,
    ):
        self.image_root = Path(image_root)
        self.annotations = annotations
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_path = self.image_root / item['image']
        caption = item['caption']
        image = self.transform(Image.open(image_path).convert('RGB'))
        # 원문 캡션 반환 → 각 train 스크립트에서 해당 모델의 tokenizer로 토큰화 (context_length 호환)
        return image, caption


def build_annotations(
    dataset_type: str,
    image_root: str,
    captions_json: Optional[str] = None,
    coco_annotations: Optional[str] = None,
    csv_path: Optional[str] = None,
    filter_missing: bool = True,
) -> List[Dict[str, str]]:
    """
    config.data 설정에 따라 캡션 리스트를 로드.
    dataset_type == "json" -> captions_json 사용
    dataset_type == "coco" -> coco_annotations 사용 (MS COCO 형식)
    dataset_type == "csv" -> csv_path 사용 (image_path, caption 컬럼)
    filter_missing=True 이면 이미지 파일이 존재하는 쌍만 반환 (캡션/이미지 수 불일치 대응).
    """
    if dataset_type == "coco":
        if not coco_annotations:
            raise ValueError("dataset_type is 'coco' but coco_annotations path is not set in config.")
        pairs = load_coco_captions(coco_annotations, image_root)
    elif dataset_type == "csv":
        if not csv_path:
            raise ValueError("dataset_type is 'csv' but csv_path is not set in config.")
        pairs = load_csv_captions(csv_path)
    elif dataset_type == "json" or dataset_type is None:
        if not captions_json:
            raise ValueError("dataset_type is 'json' but captions_json path is not set in config.")
        pairs = load_json_captions(captions_json)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'json', 'coco', or 'csv'.")

    if filter_missing:
        pairs = filter_valid_pairs(pairs, image_root, log_skip=True)
    if not pairs:
        raise ValueError(
            "No valid (image, caption) pairs after loading. "
            "Check image_root and that image files exist, or set filter_missing: false."
        )
    return pairs


def build_random_subset(dataset: Dataset, subset_size: int, seed: int = 42) -> Subset:
    """Return a reproducible random Subset of *dataset* with exactly *subset_size* samples.

    Uses a seeded generator so the same subset is selected every run, making
    experiments reproducible.  If *subset_size* >= len(dataset) the original
    dataset is returned unchanged.

    Args:
        dataset:     Any torch Dataset (or Subset thereof).
        subset_size: Number of samples to keep.
        seed:        Random seed for index permutation.

    Returns:
        A torch.utils.data.Subset wrapping *dataset*.
    """
    total = len(dataset)
    if subset_size >= total:
        msg = (
            f"build_random_subset: subset_size={subset_size} >= dataset size={total}; "
            "returning the full dataset."
        )
        log.info(msg)
        print(msg)
        return dataset  # type: ignore[return-value]

    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(total, generator=g)[:subset_size].tolist()

    msg = (
        f"build_random_subset: selected {subset_size:,} / {total:,} samples "
        f"(seed={seed})"
    )
    log.info(msg)
    print(msg)

    return Subset(dataset, indices)