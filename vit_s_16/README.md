# ViT-S/16 Student Model

This project fine-tunes a **ViT-S/16** (Vision Transformer Small, patch 16) for the
**LPCVC 2026 Track 1 Image-Text Retrieval** task using single-teacher knowledge
distillation from **EVA-02-CLIP-E-14-plus**.

## Model choices

| Role | Model | Params | Precision | Notes |
|------|-------|--------|-----------|-------|
| **Student** | `vit_s_16` (open_clip) | ~22 M | fp32 | Lightweight ViT; well within 50 MB limit |
| **Teacher** | `EVA02-E-14-plus` (`laion2b_s9b_b144k`) | ~4.4 B | fp16 (frozen) | State-of-the-art EVA-CLIP; fits on RTX 4070 alongside the student |

EVA-02-E-14-plus is used as a single teacher rather than a dual-teacher setup to
keep VRAM usage within the 12 GB budget of an RTX 4070.  The student learns to
mimic the teacher's normalised embeddings via MSE distillation loss in addition
to the standard InfoNCE contrastive loss.

A lightweight **linear projection head** (student_dim → teacher_dim) is inserted
between the student output and the distillation loss.  Its target dimension is
detected automatically at startup with a dummy forward pass through the teacher,
so no hardcoded sizes are needed.

## Directory structure

```
vit_s_16_model/
├── config.yaml            — hyper-parameters, data paths, teacher config
├── train.py               — training loop with live progress display
├── export_onnx.py         — export image/text encoders to ONNX
├── compile_and_profile.py — submit ONNX to Qualcomm AI Hub
├── inference.py           — compute Recall@1/5/10 on compiled models
└── README.md              — this file
```

## Quick start

### 1. Prepare dataset

Ensure the dataset folder is set up as described in [DATASET.md](../DATASET.md).
The default `config.yaml` points to the MS-COCO layout:

```
dataset/coco/train2017/                              ← images
dataset/coco/annotations/captions_train2017.json     ← captions
```

Switch to `dataset_type: json` or `dataset_type: csv` by editing `config.yaml`.

### 2. Train

```bash
# From project root (recommended)
python vit_s_16_model/train.py --config vit_s_16_model/config.yaml

# Or from the model subdirectory
cd vit_s_16_model
python train.py --config config.yaml
```

### 3. Export to ONNX

```bash
python vit_s_16_model/export_onnx.py \
    --ckpt    runs/vit_s_16/best.pt \
    --out_dir exported_onnx
```

Produces `exported_onnx/image_encoder.onnx` and `exported_onnx/text_encoder.onnx`.

### 4. Compile & profile on AI Hub

```bash
qai-hub configure --api_token <YOUR_TOKEN>

python vit_s_16_model/compile_and_profile.py \
    --onnx_dir exported_onnx \
    --device   "XR2 Gen 2 (Proxy)"
```

Use `--skip_profile` to compile without profiling.

### 5. Inference

```bash
python vit_s_16_model/inference.py \
    --image_job  <IMAGE_COMPILE_JOB_ID> \
    --text_job   <TEXT_COMPILE_JOB_ID> \
    --dataset_id <DATASET_ID>
```

## Training progress display

Each batch line:
```
  Epoch [ 1/15] [████████░░░░░░░░░░░░░░░░░░░░░░] 26.7%  batch   333/1250  |  loss  2.3451  |  lr 4.98e-04  |  0.62s/batch  |  ETA 6.0m  |  GPU 5240/8100MB
```

After each epoch:
```
  Validation  (took 18s):
    R@1  : 0.3120  (31.20%)
    R@5  : 0.6240  (62.40%)
    R@10 : 0.7580  (75.80%)
  ✓ New best model saved!  R@10 = 0.7580  (+0.7580)
```

## Configuration reference

| Key | Default | Description |
|-----|---------|-------------|
| `model.student_name` | `vit_s_16` | open_clip model name |
| `model.teacher_name` | `EVA02-E-14-plus` | Single teacher model |
| `model.pretrained_tag` | `laion2b_s9b_b144k` | Teacher pretrained weights tag |
| `model.embed_dim` | `512` | Student output dimension |
| `model.temperature` | `0.05` | Contrastive loss temperature |
| `training.batch_size` | `16` | Per-step batch size (RTX 4070 target) |
| `training.epochs` | `15` | Total epochs |
| `training.lr` | `5e-4` | Peak learning rate |
| `training.distill_weight` | `0.5` | Weight of distillation loss vs contrastive |
| `training.amp` | `true` | fp16 AMP — essential with the large teacher on 12 GB |
| `training.log_interval` | `10` | Batches between progress prints |

## VRAM tuning (RTX 4070, 12 GB)

The EVA-02-E-14-plus teacher is large (~4.4 B parameters) but is loaded in **fp16
and fully frozen**, so it does not consume gradient memory.  At `batch_size: 16`
expect roughly 8–10 GB used.

| batch_size | approx. VRAM | notes |
|------------|-------------|-------|
| 8 | ~5–6 GB | safe fallback if OOM |
| 16 | ~8–10 GB | **default — recommended for RTX 4070** |
| 32 | ~11–12 GB | try if GPU is free; may OOM |

## Gradient accumulation (not yet applied)

`accumulation_steps` is intentionally absent from `config.yaml` for now.
Once you have confirmed a comfortable per-epoch train time, you can re-enable it
by adding the following lines to `config.yaml` and then updating `train.py` to
use them:

```yaml
training:
  accumulation_steps: 8   # effective batch = batch_size × accumulation_steps
```

## Notes

- `force_image_size=224` is applied to both student and teacher to ensure
  compatible input resolution.
- The projection head (`student_dim → teacher_dim`) is trained jointly with the
  student and its weights are included in `all_params` for the optimiser.
- Set `use_teacher: false` in `config.yaml` to train with contrastive loss only
  (much faster per batch, but lower quality without distillation).
- For CSV datasets with mismatched image/caption counts, `filter_missing: true`
  (default) silently skips pairs where the image file does not exist.