# MobileCLIP2-S4 Training

`mobileclip2_s4_model` 폴더는 MobileCLIP2-S4 학생 모델을 2개 teacher로 distillation 학습합니다.

## 현재 코드 기준 핵심 설정

- Student: `MobileCLIP2-S4`
- Teachers:
- `ViT-SO400M-14-SigLIP2` (`webli`)
- `ViT-L-14` (`dfn2b`)
- Distillation: contrastive loss + teacher별 distillation loss 가중합
- Projection:
- teacher 1용 `Linear(768 -> 1152)`
- teacher 2용 `Linear(768 -> 768)`
- 기본 `config.yaml`:
- `batch_size: 4`
- `accumulation_steps: 32` (유효 배치 128)
- `epochs: 12`
- `lr: 5e-4`
- `distill_weights: [0.4, 0.4]`

## 실행

프로젝트 루트에서:

```bash
python mobileclip2_s4_model/train.py --config mobileclip2_s4_model/config.yaml
```

또는 폴더 내부에서:

```bash
cd mobileclip2_s4_model
python train.py --config config.yaml
```

## 결과물

- 체크포인트: `runs/mobileclip2_s4/best.pt`
- ONNX 내보내기:

```bash
python mobileclip2_s4_model/export_onnx.py --ckpt runs/mobileclip2_s4/best.pt --out_dir mobileclip2_s4_model/exported_onnx
```

- AI Hub 컴파일/프로파일:

```bash
python mobileclip2_s4_model/compile_and_profile.py --onnx_dir mobileclip2_s4_model/exported_onnx --device "XR2 Gen 2 (Proxy)"
```

- 추론:

```bash
python mobileclip2_s4_model/inference.py --image_job <IMAGE_JOB_ID> --text_job <TEXT_JOB_ID> --dataset_id <DATASET_ID>
```

## 참고

- `distill_weights` 길이는 `teacher_names` 길이와 동일해야 합니다.
- 데이터 경로 형식은 루트 [DATASET.md](../DATASET.md)를 따릅니다.
- 상세 학습 동작은 루트 [TRAINING_GUIDE.md](../TRAINING_GUIDE.md)를 참고하세요.
