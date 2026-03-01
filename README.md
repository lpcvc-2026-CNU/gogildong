# LPCVC 2026 Track 1 Baselines

이 저장소는 LPCVC 2026 Track 1(Image-Text Retrieval)을 위한 3개 모델 실험 폴더를 포함합니다.
각 폴더는 학습(`train.py`) -> ONNX 내보내기(`export_onnx.py`) -> AI Hub 컴파일/프로파일(`compile_and_profile.py`) -> 추론(`inference.py`) 흐름을 제공합니다.

- 대회 링크: https://lpcv.ai/2026LPCVC/image-text-retrieval/
- 데이터셋 준비: [DATASET.md](DATASET.md)
- 폴더별 학습 상세 가이드: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

## 폴더 구성

| 폴더 | 학생 모델 | teacher/distillation | 특징 |
|---|---|---|---|
| `vit_s_16` | `ViT-S-16` | `EVA02-E-14-plus` 단일 teacher (선택적) | gradient accumulation, warmup_ratio 기반 스케줄 |
| `mobileclip2_s4_model` | `MobileCLIP2-S4` | 2개 teacher (`ViT-SO400M-14-SigLIP2`, `ViT-L-14`) | dual-teacher distillation + teacher별 projection |
| `siglip2_model` | `siglip2_base` | `siglip2_so400m` 단일 teacher (선택적) | 가장 단순한 baseline 학습 루프 |

## 빠른 시작

### 1) 의존성 설치

```bash
pip install -r requirements.txt
```

### 2) 데이터 경로 설정

각 폴더의 `config.yaml`에서 아래를 맞춥니다.

- `data.dataset_type`: `coco` / `json` / `csv`
- `data.image_root`
- `data.captions_json` 또는 `data.coco_annotations` 또는 `data.csv_path`

### 3) 학습 실행 예시

```bash
python vit_s_16/train.py --config vit_s_16/config.yaml
python mobileclip2_s4_model/train.py --config mobileclip2_s4_model/config.yaml
python siglip2_model/train.py --config siglip2_model/config.yaml
```

### 4) ONNX 내보내기 및 AI Hub 실행 예시

```bash
python vit_s_16/export_onnx.py --ckpt runs/vit_s_16/best.pt --out_dir vit_s_16/exported_onnx
python vit_s_16/compile_and_profile.py --onnx_dir vit_s_16/exported_onnx --device "XR2 Gen 2 (Proxy)"
python vit_s_16/inference.py --image_job <IMAGE_JOB_ID> --text_job <TEXT_JOB_ID> --dataset_id <DATASET_ID>
```

`mobileclip2_s4_model`, `siglip2_model`도 동일한 순서로 실행합니다.

## 주의사항

- README의 모델 설명보다 실제 동작은 각 폴더의 `config.yaml`과 `train.py`가 기준입니다.
- 기본 학습 코드는 연구/실험용 템플릿입니다. 실제 제출용 성능 최적화(증강, DDP, 정교한 스케줄링)는 추가 구현이 필요합니다.
