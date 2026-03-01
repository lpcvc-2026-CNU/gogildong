# Folder-wise Training Guide

이 문서는 각 학습 폴더(`vit_s_16`, `mobileclip2_s4_model`, `siglip2_model`)의 **실제 `train.py` 동작 기준**으로 학습 방식을 정리합니다.

## 공통 흐름

1. `config.yaml` 로딩
2. 데이터셋 annotation 생성 (`dataset_loader.build_annotations`)
3. `ImageTextDataset` + train/val split
4. 학생 모델 forward 후 image/text 임베딩 계산
5. contrastive loss 계산
6. (선택) teacher 임베딩으로 distillation loss 추가
7. optimizer/scheduler step
8. epoch마다 Recall@1/5/10 평가 후 best checkpoint 저장

---

## 1) `vit_s_16` 폴더

### 입력/데이터

- `dataset_type`: `coco` / `json` / `csv`
- `subset_size` 지원: train split에서 랜덤 부분집합만 사용 가능
- split/subset에 seed를 적용해 재현성 확보

### 모델/teacher

- Student: `ViT-S-16`
- Teacher: `EVA02-E-14-plus` (`pretrained_tag` 사용)
- teacher는 fp16, `eval()`, `requires_grad=False`로 고정

### loss

- `contrastive_loss(img, txt, temperature)`
- `distillation_loss(student_proj, teacher)`
- total loss:
- `loss_c + distill_weight * (loss_d_img + loss_d_txt)`

### 차원 정렬

- student/teacher 출력 차원을 더미 forward로 탐지
- `Linear(student_dim -> teacher_dim)` projection head 생성

### 최적화

- AdamW
- gradient accumulation 사용 (`accumulation_steps`)
- Scheduler: `LinearLR` warmup + `CosineAnnealingLR` (`SequentialLR`)
- AMP + GradScaler + grad clipping

### 체크포인트

- 기준 metric: `R@10`
- 저장: `runs/vit_s_16/best.pt`

---

## 2) `mobileclip2_s4_model` 폴더

### 입력/데이터

- `dataset_type`: `coco` / `json` / `csv`
- random split 후 DataLoader 구성

### 모델/teacher

- Student: `MobileCLIP2-S4`
- Teachers: `teacher_names` 리스트로 다중 teacher 로딩
- 코드 기본값은 2개 teacher를 사용

### loss

- 기본 contrastive loss
- teacher별 distillation loss를 `distill_weights`로 가중합
- total loss:
- `loss_c + sum_i(weight_i * (loss_d_img_i + loss_d_txt_i))`

### 차원 정렬

- teacher 1용 projection: `Linear(768 -> 1152)`
- teacher 2용 projection: `Linear(768 -> 768)`
- 현재 코드는 teacher 수/구조를 사실상 2개 기준으로 구현

### 최적화

- AdamW
- gradient accumulation (`accumulation_steps`)
- Scheduler: `CosineAnnealingLR`
- AMP + GradScaler + grad clipping

### 체크포인트

- 기준 metric: `R@10`
- 저장: `runs/mobileclip2_s4/best.pt`

---

## 3) `siglip2_model` 폴더

### 입력/데이터

- `dataset_type`: `coco` / `json` / `csv`
- random split 후 DataLoader 구성

### 모델/teacher

- Student: `siglip2_base`
- Teacher(선택): `siglip2_so400m`
- `use_teacher`가 true이면 teacher forward 포함

### loss

- contrastive loss
- (선택) 단일 teacher distillation loss
- total loss:
- `loss_c + distill_weight * (loss_d_img + loss_d_txt)`

### 최적화

- AdamW
- Scheduler: `CosineAnnealingLR`
- AMP + GradScaler + grad clipping

### 체크포인트

- 기준 metric: `R@10`
- 저장: `runs/siglip2/best.pt`

---

## 폴더 선택 가이드

- `vit_s_16`: 단일 대형 teacher + accumulation/warmup까지 포함된 비교적 안정적인 학습 템플릿
- `mobileclip2_s4_model`: dual-teacher 실험에 적합하지만 projection/teacher 구성이 하드코딩되어 있어 확장 시 코드 수정 권장
- `siglip2_model`: 최소 baseline 확인용으로 가장 단순

## 실행 커맨드 요약

```bash
python vit_s_16/train.py --config vit_s_16/config.yaml
python mobileclip2_s4_model/train.py --config mobileclip2_s4_model/config.yaml
python siglip2_model/train.py --config siglip2_model/config.yaml
```

```bash
python vit_s_16/export_onnx.py --ckpt runs/vit_s_16/best.pt --out_dir vit_s_16/exported_onnx
python mobileclip2_s4_model/export_onnx.py --ckpt runs/mobileclip2_s4/best.pt --out_dir mobileclip2_s4_model/exported_onnx
python siglip2_model/export_onnx.py --ckpt runs/siglip2/best.pt --out_dir siglip2_model/exported_onnx
```
