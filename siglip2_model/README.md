# SigLIP 2 Student Model

This project implements an **image–text retrieval model** based on **SigLIP 2 Base** (≈86 M parameters) for the LPCVC 2026 Track 1 competition.  SigLIP 2 uses a sigmoid‑based contrastive loss, combines image–text matching with captioning and masked‑language‑model tasks, and outperforms much larger vision–language encoders【625023311643555†L54-L69】.  Here we fine‑tune the base model on the competition dataset and optionally distil knowledge from a larger teacher model.

## Model choices

* **Student** – `SigLIP2_Base`.  This 86 M parameter model provides a good balance of accuracy and latency and fits comfortably within the LPCVC 50 MB model size limit after ONNX export.
* **Teacher (optional)** – `SigLIP2_SO400M` or `SigLIP2_Giant`.  These larger models (400 M or 1 B parameters) capture richer semantics and serve as a distillation target.  Distillation helps the student match the teacher’s embedding distribution.

## Training procedure

1. **Dataset preparation** – download and organise the image–text pairs from the [LPCV 2026 Image-Text Retrieval](https://lpcv.ai/2026LPCVC/image-text-retrieval/) dataset.  See the project root’s [DATASET.md](../DATASET.md) for the expected layout.  The default script expects `images/` and `captions.json`; update `image_root` and `captions_json` in `config.yaml` accordingly.
2. **Load models** – `train.py` uses [open‑clip](https://github.com/mlfoundations/open_clip) to load the SigLIP 2 Base model and, if `use_teacher=true`, a larger SigLIP 2 model for distillation.  Both models are loaded in evaluation mode on a single GPU.
3. **Losses** – the student is optimised with a combination of:
   * **Contrastive loss** – cross‑entropy between cosine similarities of image/text embeddings and a diagonal target matrix (InfoNCE/NT‑Xent).
   * **Sigmoid alignment loss** – SigLIP’s original formulation uses a symmetric binary‑cross‑entropy loss with logits scaled by a temperature parameter【625023311643555†L54-L69】.
   * **Distillation loss (optional)** – mean‑squared error between L2‑normalised student embeddings and teacher embeddings.  Temperature scaling can be applied to soften the targets.
4. **Optimiser & scheduler** – the template uses the AdamW optimiser with a cosine learning‑rate schedule.  Adjust hyper‑parameters in `config.yaml` for your hardware.
5. **Evaluation** – after each epoch the script computes Recall@1/5/10 using the validation split.  Metrics help monitor training progress.
6. **Checkpointing** – the best model (highest Recall@10) is saved to `runs/best.pt`.  You can adjust the save directory in the config.

## Exporting and deployment

After training, run `export_onnx.py` to export the student’s image and text encoders to ONNX.  The script splits the model into separate ONNX files and stores them in `exported_onnx/`.  Then run `compile_and_profile.py` to submit the ONNX models to Qualcomm AI Hub for compilation and profiling.  Finally, `inference.py` can be used to run the compiled models on a dataset and compute retrieval metrics.

## Notes

* SigLIP 2 is trained on multilingual data and supports 300+ languages【625023311643555†L54-L69】.  If your competition dataset is multilingual, consider enabling multilingual tokenisation via open‑clip’s tokenizer.
* The provided training loop is a simplified example.  For high performance, add data augmentation, gradient accumulation, automatic mixed precision, and exponential moving average (EMA).
