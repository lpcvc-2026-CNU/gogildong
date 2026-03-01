# SigLIP2 Base Training

`siglip2_model` 폴더는 SigLIP2 Base 학생 모델의 단순 baseline 학습 파이프라인입니다.

## 현재 코드 기준 핵심 설정

- Student: `siglip2_base`
- Teacher(선택): `siglip2_so400m`
- Distillation: contrastive loss + 단일 teacher distillation(MSE)
- 기본 `config.yaml`:
- `batch_size: 128`
- `epochs: 10`
- `lr: 5e-4`
- `temperature: 0.07`
- `use_teacher: true`
- `distill_weight: 0.5`

## 실행

프로젝트 루트에서:

```bash
python siglip2_model/train.py --config siglip2_model/config.yaml
```

또는 폴더 내부에서:

```bash
cd siglip2_model
python train.py --config config.yaml
```

## 결과물

- 체크포인트: `runs/siglip2/best.pt`
- ONNX 내보내기:

```bash
python siglip2_model/export_onnx.py --ckpt runs/siglip2/best.pt --out_dir siglip2_model/exported_onnx
```

- AI Hub 컴파일/프로파일:

```bash
python siglip2_model/compile_and_profile.py --onnx_dir siglip2_model/exported_onnx --device "XR2 Gen 2 (Proxy)"
```

- 추론:

```bash
python siglip2_model/inference.py --image_job <IMAGE_JOB_ID> --text_job <TEXT_JOB_ID> --dataset_id <DATASET_ID>
```

## 참고

- 이 폴더의 `train.py`는 가장 단순한 형태라 토크나이저 처리/teacher 로딩이 모델별로 다를 수 있습니다.
- 데이터 경로 형식은 루트 [DATASET.md](../DATASET.md)를 따릅니다.
- 상세 학습 동작은 루트 [TRAINING_GUIDE.md](../TRAINING_GUIDE.md)를 참고하세요.
