# LPCVC 2026 Track 1 — Image-to-Text Retrieval

**Dual-Teacher Knowledge Distillation for Efficient VLP on Edge Devices**

> Competition: [LPCVC 2026](https://lpcv.ai) · Track 1: Image-to-Text Retrieval  
> Submission Window: March 1 – April 30, 2026  
> Target Platform: Qualcomm Snapdragon 8 Elite (via Qualcomm AI Hub)  
> Primary Metric: Recall@10  
> Latency Budget: < 35ms (image + text encoder combined)

---

## 프로젝트 개요

본 프로젝트는 **Dual-Teacher Knowledge Distillation** 전략을 사용하여  
대형 VLP(Vision-Language Pretraining) 모델의 지식을 경량 학생 모델에 전수합니다.

| 역할 | 모델 | 파라미터 | 비고 |
|------|------|----------|------|
| 학생 이미지 인코더 | MobileNetV4-Small | ~3.7M | 224×224 입력, 대회 규격 |
| 학생 텍스트 인코더 | DistilBERT-Base | ~67M | CLS 토큰 추출, max_len=77 |
| 스승 모델 1 | SigLIP2-SO400M | ~400M | 훈련 중에만 사용 |
| 스승 모델 2 | DFN ViT-H-14-378 | ~632M | 훈련 중에만 사용 |

---

## 모델 아키텍처

```
                    ┌─────────────────────────────────┐
입력 이미지          │         Student Model           │
(1×3×224×224)  ──►  │  MobileNetV4-Small (ImageEncoder)│  ──► L2-norm ──► image_embed (D=512)
                    │  + Linear Projection (→ D=512)  │
                    └───────────┬─────────────────────┘
                                │ (학습 시에만 활성화)
                         ┌──────┴───────┐
                         │ Dual Proj.   │
                         │ Head_Sig ────┼──► SigLIP2 공간 투영 (D=1152)
                         │ Head_DFN ────┼──► DFN 공간 투영 (D=1024)
                         └─────────────┘

입력 텍스트          ┌─────────────────────────────────┐
(1×77 token IDs) ──►  │  DistilBERT-Base (TextEncoder)  │  ──► L2-norm ──► text_embed (D=512)
                    │  CLS token → Linear (→ D=512)   │
                    └─────────────────────────────────┘
```

**Projection Heads**: 각 인코더 출력에 2개의 MLP 레이어(Linear → LayerNorm → GELU → Linear)를 부착하여 두 스승 모델의 임베딩 공간으로 투영합니다.

---

## 3단계 학습 전략

### Stage 1: DFN Warm-up (에폭 1~10)

> **목표**: DFN의 고품질 시각 지식으로 이미지 인코더 초기화

- DistilBERT **동결(Frozen)**
- MobileNetV4가 DFN 이미지 임베딩을 MSE Loss로 모방
- 손실 가중치: λ1=0.2 (CLIP), λ2=0.8 (MSE-DFN), λ3=0.0 (KL 비활성)

```
L_stage1 = 0.2 × L_CLIP + 0.8 × L_MSE(DFN)
```

**이유**: 파라미터가 적은 MobileNetV4는 초기에 노이즈가 많은 데이터를 학습하면 수렴이 불안정합니다. DFN의 깨끗한 특징을 먼저 학습하면 안정적인 초기화가 가능합니다.

---

### Stage 2: Dual Teacher KD (에폭 11~40)

> **목표**: 두 스승의 유사도 분포를 학생이 흡수

- DistilBERT **해제(Unfrozen)**
- SigLIP2와 DFN이 생성한 유사도 행렬의 평균을 Soft Label로 사용
- **Dynamic Weighting**: 초반은 λ2(MSE) 높게, 후반은 λ3(KL) 높게

```
epoch_progress = current_epoch / total_epochs

λ2 = 0.5 → 0.1  (linear decay)
λ3 = 0.2 → 0.6  (linear increase)

L_stage2 = λ1 × L_CLIP + λ2 × L_MSE + λ3 × L_KL(avg_teacher)
```

**KL Loss 상세**:
```
dist_teacher = (dist_SigLIP2 + dist_DFN) / 2    # 두 스승 분포의 평균
L_KL = KL(dist_student || dist_teacher)
```

---

### Stage 3: QAT Fine-tuning (에폭 41~45)

> **목표**: INT8 양자화 보정 및 엣지 배포 최적화

- PyTorch QAT(Quantization-Aware Training) 적용
- 자체 Contrastive Loss 비중 상향, 증류 강도 하향
- 손실 가중치: λ1=0.7 (CLIP), λ2=0.1 (MSE), λ3=0.2 (KL)

```
L_stage3 = 0.7 × L_CLIP + 0.1 × L_MSE + 0.2 × L_KL
```

---

## 손실 함수 상세

### L_CLIP — 대조 학습 손실
```
L_CLIP = (CE(logit_scale × E_img × E_txt^T) + CE(logit_scale × E_txt × E_img^T)) / 2
```

### L_MSE — DFN 특징 모방 손실
```
L_MSE = MSE(normalize(Student_proj_DFN), normalize(DFN_embed))
```

### L_KL — 확률 분포 증류 손실
```
Teacher_dist_i2t = softmax((E_sig_img × E_sig_txt^T) / τ)   # SigLIP2
Teacher_dist_i2t += softmax((E_dfn_img × E_dfn_txt^T) / τ)  # DFN
Teacher_dist_i2t /= 2                                        # 평균

L_KL = KL(Student_dist_i2t || Teacher_dist_i2t)
      + KL(Student_dist_t2i || Teacher_dist_t2i)
```

---

## 프로젝트 구조

```
lpcvc2026_track1/
├── models/
│   ├── student.py          # StudentCLIP: MNv4 + DistilBERT + 투영 헤드
│   ├── teacher.py          # SigLIP2Teacher, DFNTeacher, TeacherManager
│   └── projection.py       # DualProjectionHeads (Head_Sig, Head_DFN)
├── training/
│   ├── loss.py             # L_CLIP, L_MSE, L_KL, TotalLoss
│   └── trainer.py          # Stage1/2/3Trainer, 공통 학습 루프
├── data/
│   └── dataset.py          # ImageCaptionDataset, CLIPTextTokenizer, transforms
├── utils/
│   ├── config.py           # ModelConfig, TrainingConfig, DataConfig, ExportConfig
│   ├── metrics.py          # Recall@K 평가 함수
│   └── export.py           # ONNX 내보내기 (이미지/텍스트 인코더 분리)
├── scripts/
│   └── export_onnx.py      # ONNX 내보내기 실행 스크립트
├── train.py                # 메인 학습 진입점
├── evaluate.py             # 평가 스크립트
├── requirements.txt        # 의존성 목록
└── README.md               # 이 파일
```

---

## 빠른 시작

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

데이터 디렉토리에 `captions.csv` 파일을 생성합니다:

```csv
image_path,caption
images/img001.jpg,"A dog running in the park"
images/img002.jpg,"A sunset over the mountains"
...
```

### 3. 학습 실행

```bash
# 전체 3단계 순차 학습
python train.py --stage all --data_path data/train --val_data data/val

# 개별 단계 실행
python train.py --stage 1 --data_path data/train
python train.py --stage 2 --data_path data/train --resume checkpoints/stage1_epoch010.pt
python train.py --stage 3 --data_path data/train --resume checkpoints/stage2_epoch030.pt
```

### 4. 평가

```bash
python evaluate.py --checkpoint checkpoints/stage3_epoch005.pt --val_data data/val
```

### 5. ONNX 내보내기 및 제출

```bash
python scripts/export_onnx.py --checkpoint checkpoints/stage3_epoch005.pt --output_dir export/
```

출력 파일을 Qualcomm AI Hub에 업로드합니다:
1. `export/image_encoder.onnx` → Snapdragon 8 Elite 컴파일
2. `export/text_encoder.onnx` → Snapdragon 8 Elite 컴파일
3. 컴파일된 Job ID를 LPCVC 제출 포털에 제출

---

## 대회 규격 체크리스트

| 항목 | 값 | 상태 |
|------|----|------|
| 입력 이미지 크기 | 1×3×224×224 | ✅ |
| 텍스트 토크나이저 | openai/clip-vit-base-patch32 | ✅ |
| 최대 텍스트 길이 | 77 토큰 | ✅ |
| 임베딩 정규화 | F.normalize() (L2) | ✅ |
| 평가 메트릭 | Recall@10 | ✅ |
| 지연시간 목표 | < 35ms (이미지+텍스트 합산) | 🔄 AI Hub 실측 필요 |
| 제출 형식 | Qualcomm AI Hub Compiled Job | ✅ |

---

## 주요 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|----|------|
| embed_dim | 512 | 공유 임베딩 차원 |
| batch_size | 256 | 배치 크기 |
| temperature | 0.07 (learnable) | 대조 학습 온도 |
| kl_temperature | 4.0 | KL 증류 소프트 레이블 온도 |
| stage1 lr | 1e-4 | Stage 1 학습률 |
| stage2 lr | 5e-5 | Stage 2 학습률 |
| stage3 lr | 1e-5 | Stage 3 (QAT) 학습률 |

---

## 성능 기대치

| 모델 | Recall@10 (COCO 5K) | 지연시간 |
|------|--------------------|---------| 
| CLIP-ViT-B/32 (baseline) | ~75% | ~120ms |
| **본 학생 모델 (목표)** | **>70%** | **<35ms** |

> Recall@10이 핵심 지표입니다. 10개 검색 결과 안에 정답이 포함되기만 하면 됩니다.
> Temperature 튜닝과 임베딩 정규화가 이 지표에 가장 큰 영향을 미칩니다.

---

## 라이선스

본 코드는 LPCVC 2026 참가 목적으로 작성되었습니다.
