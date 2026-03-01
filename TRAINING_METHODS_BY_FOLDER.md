# 폴더별 학습 방식 정리

이 문서는 프로젝트 내 각 모델 폴더(`vit_s_16`, `mobileclip2_s4_model`, `siglip2_model`)에서 채택한 학습 전략을 **기술 선택 배경**, **기대 효과**, **소스 코드 근거** 중심으로 정리합니다.

---

## 1) `vit_s_16` — 제한된 GPU 환경을 위한 단일 교사 KD + Gradient Accumulation

### 핵심 전략 요약
- **학생 모델:** `ViT-S-16` (경량)
- **교사 모델:** `EVA02-E-14-plus` (대형, 고성능)
- **학습 방식:**
  - CLIP 스타일 양방향 대조학습(이미지→텍스트, 텍스트→이미지)
  - 교사 임베딩을 정답 신호로 활용하는 **Knowledge Distillation (KD)**
  - 작은 micro-batch를 여러 번 누적해 큰 배치처럼 학습하는 **gradient accumulation**
  - Warmup + Cosine 스케줄

### 기술 선택 배경
1. **연산 자원 제약(예: 12GB급 GPU) 대응**
   - ViT-S는 파라미터/메모리 관점에서 모바일·엣지 지향이며 학습 비용이 상대적으로 낮습니다.
   - 반면 교사는 대형 모델이므로 직접 파인튜닝보다, 교사 특성을 학생으로 전달(KD)하는 접근이 효율적입니다.
2. **배치 사이즈 확보 필요성**
   - 대조학습 계열은 batch 내 음성 샘플 수가 중요하기 때문에, 작은 batch만으로는 안정성이 떨어질 수 있습니다.
   - `accumulation_steps`로 유효 배치를 키워 학습을 안정화하려는 의도입니다.
3. **초반 발산 방지**
   - Warmup 이후 Cosine decay를 통해 초기 과도한 업데이트를 줄이고 후반에 부드럽게 수렴시킵니다.

### 기대 효과
- 제한된 메모리에서도 **유효 배치 증가 효과**를 통해 대조학습 안정성 향상
- 대형 교사의 표현력을 학생에 주입하여 **Recall 성능 개선** 가능성 증가
- Warmup + Cosine으로 학습 초반 불안정 완화 및 후반 일반화 개선 기대

### 코드 근거 (어디를 보면 되는가)
- 설정값(배치 누적, 교사 사용, distill weight):
  - `vit_s_16/config.yaml`
    - `training.accumulation_steps`, `training.use_teacher`, `training.distill_weight`
- 대조학습 손실 + KD 손실 정의:
  - `vit_s_16/train.py`
    - `contrastive_loss(...)`
    - `distillation_loss(...)`
- Gradient Accumulation 실제 적용:
  - `vit_s_16/train.py`
    - `loss_scaled = loss / accum`
    - 누적 step마다 optimizer/scheduler step 수행 (`is_update_step` 블록)
- 스케줄러(Warmup → Cosine):
  - `vit_s_16/train.py`
    - `LinearLR`, `CosineAnnealingLR`, `SequentialLR`
- KD 경로(교사 출력 계산 + 학생 투영 후 MSE 결합):
  - `vit_s_16/train.py`
    - `teacher` 로드/고정, `proj` 레이어, distill loss 가산 로직

---

## 2) `mobileclip2_s4_model` — 듀얼 교사 KD(멀티-티처) + 대규모 누적 배치

### 핵심 전략 요약
- **학생 모델:** `MobileCLIP2-S4`
- **교사 모델:** 2개 동시 사용
  - `ViT-SO400M-14-SigLIP2`
  - `ViT-L-14`
- **학습 방식:**
  - CLIP 대조학습 + **멀티-티처 KD**
  - 교사별 distill weight 분리
  - 교사 차원 불일치 해결을 위한 teacher별 projection head
  - 매우 큰 유효 배치를 위한 높은 accumulation_steps

### 기술 선택 배경
1. **서로 다른 교사의 장점 결합**
   - 단일 교사보다 멀티-티처는 표현 다양성을 전달할 수 있습니다.
   - SigLIP 계열 교사와 ViT-L 교사를 함께 두어, 한쪽으로 치우치지 않은 정렬 신호를 주려는 설계입니다.
2. **임베딩 차원 불일치 문제 해결 필요**
   - 학생/교사 임베딩 차원이 다르므로, 단순 MSE가 불가능하거나 비효율적입니다.
   - teacher별 projection layer를 따로 두어 정합성을 맞춘 뒤 distillation합니다.
3. **작은 GPU에서도 큰 배치 효과 확보**
   - `batch_size=4`와 `accumulation_steps=32` 조합으로 유효 배치 128을 구성하여 대조학습 효율을 확보하려는 전략입니다.

### 기대 효과
- 듀얼 교사로부터 보완적 신호를 받아 학생의 임베딩 품질 향상 기대
- projection을 통한 차원 정렬로 KD 학습 안정성 향상
- 큰 유효 배치로 대조학습에서 hard negative 활용 효과 증대

### 코드 근거 (어디를 보면 되는가)
- 설정값(듀얼 교사, 가중치, 누적 배치):
  - `mobileclip2_s4_model/config.yaml`
    - `model.teacher_names`, `training.distill_weights`, `training.accumulation_steps`
- 교사 다중 로딩 및 고정:
  - `mobileclip2_s4_model/train.py`
    - `teacher_models`/`teacher_tokenizers` 구성 루프
- teacher별 projection:
  - `mobileclip2_s4_model/train.py`
    - `proj_t1 = Linear(768, 1152)`
    - `proj_t2 = Linear(768, 768)`
- 멀티-티처 KD 가산:
  - `mobileclip2_s4_model/train.py`
    - `for i, (weight, teacher, tok) in enumerate(...)`
    - teacher별 토크나이즈/특징 추출 후 `loss += weight * (...)`
- Gradient Accumulation 적용:
  - `mobileclip2_s4_model/train.py`
    - `loss = loss / accum`
    - `(batch_idx + 1) % accum == 0` 조건에서 optimizer/scheduler step

---

## 3) `siglip2_model` — 베이스라인형 단일 교사 KD + 표준 대조학습

### 핵심 전략 요약
- **학생 모델:** `siglip2_base`
- **교사 모델:** `siglip2_so400m` (옵션)
- **학습 방식:**
  - CLIP/NT-Xent 계열 대조학습
  - 단일 교사 KD (이미지/텍스트 임베딩 각각 MSE)
  - Cosine LR scheduler + AMP + grad clipping

### 기술 선택 배경
1. **안정적인 베이스라인 확보**
   - SigLIP2 계열의 student/teacher 조합은 아키텍처/토크나이저 측면에서 일관성이 높아 초기 실험 베이스라인 구축이 쉽습니다.
2. **구현 복잡도 대비 성능 확보**
   - 멀티-티처, 복잡 projection, 고급 스케줄 없이도 KD + contrastive만으로 경쟁력 있는 출발점을 만들 수 있습니다.
3. **확장 가능한 최소 구조**
   - 현재 구조는 단순하지만, 이후 accumulation/teacher 확장/EMA 활용 등의 실험을 붙이기 쉬운 형태입니다.

### 기대 효과
- 비교적 단순한 코드 경로로 재현성 높은 초기 성능 확보
- KD로 student 단독 학습 대비 수렴 속도/최종 리콜 개선 가능
- AMP + clipping으로 학습 안정성/속도 균형 확보

### 코드 근거 (어디를 보면 되는가)
- 설정값(학생/교사, distill weight):
  - `siglip2_model/config.yaml`
    - `model.student_name`, `model.teacher_name`, `training.distill_weight`
- 손실 정의:
  - `siglip2_model/train.py`
    - `contrastive_loss(...)`
    - `distillation_loss(...)`
- KD 결합 로직:
  - `siglip2_model/train.py`
    - `if teacher_model is not None:` 블록에서 teacher embedding 계산 후 distill loss 가산
- 학습 안정화 요소:
  - `siglip2_model/train.py`
    - `torch.amp.autocast(...)`, `GradScaler`, `clip_grad_norm_`, `CosineAnnealingLR`

---

## 폴더 간 비교 요약

| 폴더 | Student | Teacher 전략 | 배치 전략 | 특징 |
|---|---|---|---|---|
| `vit_s_16` | ViT-S/16 | 단일 대형 교사(EVA02-E-14-plus) | accumulation 적극 사용 | 자원 제약 환경 최적화 + Warmup/Cosine |
| `mobileclip2_s4_model` | MobileCLIP2-S4 | 듀얼 교사(SigLIP2 + ViT-L) | accumulation 매우 크게 사용 | teacher별 projection 포함한 멀티-티처 KD |
| `siglip2_model` | SigLIP2 Base | 단일 교사(SigLIP2 SO400M) | 일반 배치(누적 없음) | 단순·안정 베이스라인 |

---

## 실무 적용 시 권장 해석
- **리소스가 가장 부족**하면: `vit_s_16` 방식(작은 student + accumulation + 단일 강교사)
- **최고 성능 지향**이면: `mobileclip2_s4_model` 방식(멀티-티처 + projection + 큰 유효배치)
- **빠른 실험/재현성 우선**이면: `siglip2_model` 방식(단순 베이스라인)

필요하면 다음 단계로, 이 문서를 기반으로 각 폴더별 **학습 플로우 다이어그램**(데이터→학생/교사→손실→업데이트)까지 추가할 수 있습니다.
