# LPCVC 2026 Track 1 — Image-to-Text Retrieval

**Dual-Teacher Knowledge Distillation for Efficient VLP on Edge Devices**

> **대회**: [LPCVC 2026](https://lpcv.ai) · Track 1: Image-to-Text Retrieval  
> **평가 플랫폼**: Qualcomm XR2 Gen 2 (Qualcomm AI Hub)  
> **주요 지표**: Recall@10  
> **지연시간 기준**: 이미지 + 텍스트 인코더 합산 35 ms 미만

---

## 프로젝트 개요

대형 VLP(Vision-Language Pretraining) 모델 두 개를 스승(Teacher)으로 삼아 **이중 지식 증류(Dual-Teacher KD)**를 수행합니다. 경량 학생 모델이 스승의 시각-언어 이해를 흡수하여, 낮은 지연시간 제약을 지키면서도 높은 Recall@10을 달성하는 것이 목표입니다.

| 역할 | 모델 | 파라미터 수 | 비고 |
|------|------|------------|------|
| 학생 이미지 인코더 | MobileNetV4-Small | ~3.7M | 224×224, 대회 고정 |
| 학생 텍스트 인코더 | DistilBERT-Base | ~67M | CLIP vocab(49408)으로 embedding 교체 |
| 스승 1 | SigLIP2-SO400M-384 | ~400M | 학습 시에만 사용 |
| 스승 2 | DFN ViT-H-14-378 | ~632M | 학습 시에만 사용 |

---

## 아키텍처

```
입력 이미지 (1×3×224×224)
        │
        ▼
 ┌─────────────────────────────────┐
 │  MobileNetV4-Small              │
 │  Global Average Pooling         │
 │  Linear → LayerNorm (embed_dim) │ ──→ L2-norm ──→ image_embed (D=512)
 └───────────────┬─────────────────┘
                 │ (학습 시에만)
        ┌────────┴─────────┐
        │ DualProjectionHeads │
        │ head_dfn ──────────┼──→ DFN 공간 투영 (D=1024) ──→ L_MSE_DFN
        │ head_sig ──────────┼──→ SigLIP2 공간 투영 (D=1152) ──→ L_cosine_SIG
        └─────────────────────┘

입력 텍스트 (1×77 CLIP token IDs)
        │
        ▼
 ┌─────────────────────────────────────────────────────┐
 │  DistilBERT-Base                                     │
 │  word_embeddings: Embedding(49408, 768)  ← CLIP vocab│
 │  [CLS] 토큰 추출 → Linear → LayerNorm (embed_dim)    │ ──→ L2-norm ──→ text_embed (D=512)
 └──────────────────────────────────────────────────────┘
```

### 토크나이저 분리 정책

| 대상 | 토크나이저 | 이유 |
|------|-----------|------|
| 학생 (Student) | `openai/clip-vit-base-patch32` CLIPTokenizer, max_len=77 | **대회 평가 입력 규격 고정** |
| SigLIP2 Teacher | `AutoProcessor` (SigLIP2 전용) — `encode_text(raw_texts)` 내부 처리 | SigLIP2는 자체 vocabulary 사용 |
| DFN Teacher | `open_clip.get_tokenizer("ViT-H-14-378-quickgelu")` — `encode_text(raw_texts)` 내부 처리 | DFN은 CLIP-style BPE 사용 |

> ⚠ 스승 모델에 학생 CLIP token ID 를 재사용하면 SigLIP2 텍스트 임베딩이 망가집니다.  
> Trainer 는 raw caption 문자열(`batch["caption"]`)을 스승 모델에 넘기고,  
> 각 Teacher 가 자체 tokenizer 로 처리하는 구조입니다.

---

## 3단계 학습 전략

### Stage 1: DFN Warm-up (에폭 1–10)

> **목표**: DFN의 정제된 시각 특징으로 이미지 인코더 초기화

- DistilBERT **동결** (`freeze_text: true`)
- DFN 이미지 임베딩을 MSE로 모방 — λ2_dfn=0.9
- SigLIP2 및 KL 비활성 — λ2_sig=0, λ3=0
- L_CLIP 보조 — λ1=0.1

```
L_stage1 ≈ 0.1 × L_CLIP + 0.9 × L_MSE_DFN
```

**이유**: 파라미터가 적은 MobileNetV4를 노이즈 많은 데이터로 바로 대조 학습하면 수렴이 불안정합니다. DFN의 깨끗한 시각 특징을 먼저 학습해 안정적인 초기값을 확보합니다.

---

### Stage 2: Dual Teacher KD (에폭 11–40)

> **목표**: SigLIP2와 DFN의 관계 지식을 동시에 흡수

- DistilBERT **해제** (`freeze_text: false`)
- **동적 λ 스케줄**: 에폭 진행에 따라 선형 보간 후 합=1 정규화

| λ | 초반 (epoch=0) | 후반 (epoch=30) | 역할 |
|---|--------------|----------------|------|
| λ1 | 0.3 → 0.3 | (고정) | L_CLIP |
| λ2_dfn | 0.35 → 0.07 | (감소) | L_MSE (DFN 이미지·텍스트) |
| λ2_sig | 0.15 → 0.03 | (감소) | L_cosine (SigLIP2 이미지·텍스트) |
| λ3 | 0.2 → 0.6 | (증가) | L_KL (평균 유사도 분포 증류) |

```
L_stage2 = λ1·L_CLIP + λ2_dfn·L_MSE_DFN + λ2_sig·L_cosine_SIG + λ3·L_KL
```

---

### Stage 3: QAT Fine-tuning (에폭 41–45)

> **목표**: INT8 양자화 보정 및 배포 환경 정밀도 유지

- PyTorch QAT(Quantization-Aware Training) 적용
  - `qat_image_encoder: true` — 기본 활성
  - `qat_text_encoder: false` — 선택적 (config.yaml 에서 변경 가능)
- L_CLIP 비중 상향, 증류 강도 하향

```
L_stage3 = 0.7·L_CLIP + 0.08·L_MSE_DFN + 0.02·L_cosine_SIG + 0.2·L_KL
```

---

## 손실 함수 상세

```
L_total = λ1 × L_CLIP
        + λ2_dfn × L_MSE_DFN        ← DFN 공간 특징 모방 (Head_DFN)
        + λ2_sig × L_cosine_SIG     ← SigLIP2 공간 특징 모방 (Head_Sig)
        + λ3 × L_KL                 ← 두 스승 유사도 분포 평균 증류
```

| 손실 | 수식 | 목적 |
|------|------|------|
| L_CLIP | 대칭 InfoNCE | 자체 대조 학습 |
| L_MSE_DFN | MSE(norm(Head_DFN(e)), dfn_embed) | DFN 공간 특징 모방 |
| L_cosine_SIG | MSE(norm(Head_Sig(e)), sig_embed) | SigLIP2 공간 특징 모방 |
| L_KL | KL(student_dist ‖ avg_teacher_dist) | 유사도 분포 증류 |

> **Projection Head 설계 의도**: Head_Sig, Head_DFN 각각 L_cosine 및 L_MSE loss에 직접 연결됩니다. 두 헤드 모두 실제 loss에 참여하는 활성 파라미터입니다.

---

## 프로젝트 구조

```
lpcvc2026_track1/
├── config.yaml              ← 모든 하이퍼파라미터 (여기서만 수정)
├── train.py                 ← 학습 진입점
├── evaluate.py              ← 평가 스크립트
├── models/
│   ├── student.py           ← StudentCLIP (MNv4 + DistilBERT + Projection Heads)
│   ├── teacher.py           ← SigLIP2Teacher, DFNTeacher, TeacherManager
│   └── projection.py        ← DualProjectionHeads (Head_Sig, Head_DFN)
├── training/
│   ├── loss.py              ← L_CLIP, L_MSE_DFN, L_cosine_SIG, L_KL, TotalLoss
│   └── trainer.py           ← Stage1/2/3Trainer (cfg 기반, 동적 λ)
├── data/
│   └── dataset.py           ← StudentTokenizer(CLIP), ImageCaptionDataset, transforms
├── utils/
│   ├── config.py            ← YAML → ConfigNode (dot notation)
│   ├── metrics.py           ← Recall@K 평가
│   └── export.py            ← ONNX 내보내기 유틸
└── scripts/
    └── export_onnx.py       ← ONNX + Qualcomm AI Hub 컴파일/양자화/프로파일링
```

---

## 빠른 시작

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

`data/train/captions.csv` 형식:

```csv
image_path,caption
images/img001.jpg,"A dog running in the park"
images/img002.jpg,"A sunset over the mountains"
```

### 3. 설정 파일 수정

학습 전에 `config.yaml` 에서 경로만 변경하세요:

```yaml
data:
  train_path: "data/train"   # ← 실제 경로로 변경
  val_path: "data/val"
```

### 4. 학습 실행

```bash
# 전체 3단계 순차 학습 (config.yaml 의 경로 사용)
python train.py

# 개별 단계 실행
python train.py --stage 1
python train.py --stage 2 --resume checkpoints/stage1_epoch010.pt
python train.py --stage 3 --resume checkpoints/stage2_epoch030.pt

# 커스텀 설정 파일 사용
python train.py --config my_experiment.yaml

# 스승 없이 디버깅 (빠른 파이프라인 테스트)
python train.py --no_teachers --max_samples 500
```

> ❌ 구버전 문서에 있던 `--data_path`, `--val_data` 인자는 존재하지 않습니다.  
> 모든 경로는 `config.yaml → data.train_path / data.val_path` 에서만 설정합니다.

### 5. 평가

```bash
python evaluate.py --checkpoint checkpoints/stage3_epoch005.pt
python evaluate.py --checkpoint checkpoints/stage3_epoch005.pt --config config.yaml
```

### 6. ONNX 내보내기 및 AI Hub 제출

```bash
# ONNX 내보내기만 (로컬 검증)
python scripts/export_onnx.py --checkpoint checkpoints/stage3_epoch005.pt --onnx_only

# ONNX + AI Hub 컴파일
python scripts/export_onnx.py --checkpoint checkpoints/stage3_epoch005.pt

# ONNX + INT8 양자화 + 컴파일 + 지연시간 측정
python scripts/export_onnx.py \
    --checkpoint checkpoints/stage3_epoch005.pt \
    --quantize \
    --profile
```

AI Hub 컴파일 완료 후 출력되는 **Job ID** 를 LPCVC 제출 포털에 입력하면 됩니다.

---

## 대회 규격 체크리스트

| 항목 | 값 | 코드 위치 |
|------|----|---------|
| 입력 이미지 크기 | 1×3×224×224 | `config.yaml → model.student_image_input_size` |
| 텍스트 토크나이저 | `openai/clip-vit-base-patch32` | `config.yaml → model.clip_tokenizer_name` |
| 텍스트 최대 길이 | 77 토큰 | `config.yaml → model.max_text_length` |
| DistilBERT vocab 크기 | 49408 (CLIP) | `config.yaml → model.clip_vocab_size` |
| 임베딩 정규화 | `F.normalize()` (L2) | `models/student.py` |
| 평가 지표 | Recall@10 | `utils/metrics.py` |
| 지연시간 목표 | < 35ms 합산 | AI Hub 프로파일링으로 실측 |
| 제출 형식 | Qualcomm AI Hub Compile Job ID | `scripts/export_onnx.py` |

---

## 주요 하이퍼파라미터

모든 값은 `config.yaml` 에서 수정합니다. 코드를 직접 수정할 필요가 없습니다.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `model.embed_dim` | 512 | 공유 임베딩 차원 |
| `training.batch_size` | 256 | 배치 크기 |
| `training.temperature` | 0.07 | 대조 학습 초기 온도 |
| `training.kl_temperature` | 4.0 | KL soft-label 온도 |
| `training.stage1.epochs` | 10 | Stage 1 에폭 수 |
| `training.stage2.epochs` | 30 | Stage 2 에폭 수 |
| `training.stage3.epochs` | 5 | Stage 3 에폭 수 |
| `training.stage3.qat_image_encoder` | true | 이미지 인코더 QAT |
| `training.stage3.qat_text_encoder` | false | 텍스트 인코더 QAT |

---

## 알려진 설계 상충

| 항목 | 현재 결정 | 근거 |
|------|---------|------|
| DistilBERT + CLIP vocab | word_embeddings 레이어 교체 (Xavier 초기화) | 대회 규격 CLIP 토큰 수용 필수 |
| SigLIP2 text teacher | AutoProcessor 내부 처리 (별도 tokenizer) | SigLIP2 자체 BPE vocab 사용 |
| Stage 3 text encoder QAT | 기본 비활성 | DistilBERT QAT 시 수렴 불안정 가능 |
| Stage 1 L_CLIP | λ1=0.1 (소량 유지) | 완전 제거 시 이미지-텍스트 정렬 지연 |

---

## requirements

```
torch>=2.1.0, timm>=1.0.0, transformers>=4.39.0,
open_clip_torch>=2.24.0, onnx>=1.16.0, onnxruntime>=1.18.0,
qai_hub (Qualcomm AI Hub SDK)
```
