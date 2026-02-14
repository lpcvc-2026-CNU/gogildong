# MobileCLIP2‑S4 Student Model

This project implements a **mobile‑friendly image–text retrieval model** using
Apple’s **MobileCLIP 2 S4** architecture.  MobileCLIP 2 replaces the
heavy vision transformer in CLIP with a low‑latency, low‑memory
backbone and couples it with a distilled text encoder to achieve
state‑of‑the‑art zero‑shot accuracy in a 50 MB package【691468416182057†L24-L56】.

## Model choices

* **Student** – `mobileclip2_s4`.  This tiny model achieves roughly the
  same zero‑shot ImageNet accuracy as a SigLIP 2 SO400M/14 model while
  being **half the size** and **2.5× faster**【691468416182057†L50-L56】.
* **Teachers** – by default the project uses **two teachers**:
  * **SigLIP 2 SO400M/14** – a 400 M parameter model trained with SigLIP’s
    multi‑task loss on multilingual web data【625023311643555†L54-L69】.
  * **DFN ViT‑L/14 CLIP** (or alternatively **MetaCLIP 2 H/14**).  Apple’s
    MobileCLIP 2 experiments show that combining a SigLIP teacher with
    a CLIP‑style teacher provides complementary semantic signals【691468416182057†L34-L56】.

## Training procedure

1. **Dataset** – use the [LPCV 2026 Image-Text Retrieval](https://lpcv.ai/2026LPCVC/image-text-retrieval/) dataset or e.g. MS‑COCO.  See [DATASET.md](../DATASET.md) for the expected layout.  Align images and captions in a JSON file and set `image_root` and `captions_json` in `config.yaml`.
2. **Load student** – `train.py` uses [open‑clip](https://github.com/mlfoundations/open_clip)
   to load the `mobileclip2_s4` student.  The model’s vision encoder is
   a hybrid convolution–transformer architecture optimised for mobile.
3. **Load teachers** – if `use_teacher=true`, the script loads both
   teachers and produces their embeddings in parallel.  Distillation
   losses are computed separately for each teacher and summed.
4. **Losses** – the student is trained with a mixture of contrastive
   loss (InfoNCE) and distillation losses from each teacher.  You may
   adjust the weights of each distillation component in `config.yaml`.
5. **Optimisation** – MobileCLIP 2 uses temperature tuning, mixed
   precision (AMP), exponential moving average (EMA) and gradient
   clipping【691468416182057†L24-L56】.  These features are exposed via
   configuration options.
6. **Evaluation & checkpointing** – as in the SigLIP model, the script
   measures Recall@1/5/10 on a validation split and saves the best
   checkpoint to `runs/best.pt`.

## Exporting and deployment

MobileCLIP2 encoders are exported using `export_onnx.py` to separate
ONNX files (image and text).  You can compile and profile them using
`compile_and_profile.py`.  Finally, `inference.py` loads the compiled
models to compute retrieval metrics on a dataset.

## Notes

* You may choose a different teacher combination (e.g., SigLIP 2 Giant
  + MetaCLIP 2 H/14) by editing `config.yaml`.
* MobileCLIP 2 expects input images of size 224×224; no patching is
  necessary.
* The default configuration uses a small number of epochs.  Increase
  the training duration for better performance.
