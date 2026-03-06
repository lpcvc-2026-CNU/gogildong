# ViT-S/16 Training

`vit_s_16` 폴더는 ViT-S/16 학생 모델을 단일 teacher distillation으로 학습합니다.

## 현재 코드 기준 핵심 설정

- Student: `ViT-S-16`
- Teacher: `EVA02-E-14-plus` (`pretrained_tag: laion2b_s9b_b144k`)
- Distillation: teacher 임베딩과 student 임베딩(MSE) + contrastive loss
- Projection: student 출력 차원 -> teacher 출력 차원을 `train.py`에서 자동 탐지 후 `Linear` projection 생성
- Dataset type: `coco` / `json` / `csv`
- 기본 `config.yaml`:
- `batch_size: 4`
- `accumulation_steps: 8` (유효 배치 32)
- `epochs: 15`
- `lr: 5e-4`
- `warmup_ratio: 0.1`
- `distill_weight: 0.5`

## 실행

프로젝트 루트에서:

```bash
python vit_s_16/train.py --config vit_s_16/config.yaml
```

또는 폴더 내부에서:

```bash
cd vit_s_16
python train.py --config config.yaml
```

## 결과물

- 체크포인트: `runs/vit_s_16/best.pt`
- ONNX 내보내기:

```bash
python vit_s_16/export_onnx.py --ckpt runs/vit_s_16/best.pt --out_dir vit_s_16/exported_onnx
```

- AI Hub 컴파일/프로파일:

```bash
python vit_s_16/compile_and_profile.py --onnx_dir vit_s_16/exported_onnx --device "XR2 Gen 2 (Proxy)" --target_runtime qnn_context_binary
```

- 추론:

```bash
python vit_s_16/inference.py --image_job <IMAGE_JOB_ID> --text_job <TEXT_JOB_ID> --dataset_id <DATASET_ID>
```

## 참고

- `use_teacher: false`로 두면 순수 contrastive 학습만 수행합니다.
- 데이터 경로 형식은 루트 [DATASET.md](../DATASET.md)를 따릅니다.
- 상세 학습 동작은 루트 [TRAINING_GUIDE.md](../TRAINING_GUIDE.md)를 참고하세요.
