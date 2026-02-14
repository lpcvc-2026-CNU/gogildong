# Proposed Model Implementations for LPCVC 2026 Track 1

This directory contains three example implementations for the **LPCVC 2026 Track 1 Image‑to‑Text Retrieval** task.  Each subdirectory holds a self‑contained project that follows the conventions of the official sample solution (export, compile/profile, upload, inference) but swaps the encoders for more advanced vision‑language models described in the research summary.

- **Track & dataset:** [LPCV 2026 Image-Text Retrieval](https://lpcv.ai/2026LPCVC/image-text-retrieval/)
- **Dataset preparation and expected layout:** see [DATASET.md](DATASET.md).

The projects are intended as starting points.  You should review the included `README.md` files and adjust hyper‑parameters, data paths, and training settings according to your needs and the LPCVC competition rules.

## Projects

| Directory              | Student model | Teacher model(s)       | Notes                            |
|-----------------------|---------------|------------------------|----------------------------------|
| `siglip2_model`       | SigLIP 2 Base (≈86 M) | SigLIP 2 SO400M (400 M) or Giant (1 B) | Uses the sigmoid based contrastive loss of SigLIP.  Fine‑tunes the student with cross‑entropy and distillation losses. |
| `mobileclip2_s4_model` | MobileCLIP2‑S4 (~50 M) | SigLIP 2 SO400M/14 + DFN ViT‑L/14 or MetaCLIP 2 H/14 | Follows Apple’s MobileCLIP2 recipe.  Employs dual‑teacher distillation and heavy data augmentation. |
| `eva_clip_l_model`    | EVA‑02‑CLIP‑L (≈428 M) | EVA‑02‑CLIP‑E (4.4 B) or CLIP‑ViT‑bigG‑14 | Adapts the EVA‑CLIP architecture; optionally uses a large EVA or bigG model as teacher. |

Each subproject contains:

* **`README.md`** — explains the model choice, training procedure, and how to run the scripts.
* **`config.yaml`** — defines hyper‑parameters such as learning rate, batch size, number of epochs, and whether to use distillation.
* **`train.py`** — high‑level training loop using [open‑clip](https://github.com/mlfoundations/open_clip) for loading models.  The code is intentionally simple and should be extended with proper data handling and evaluation.
* **`export_onnx.py`** — exports the trained student encoders to ONNX, following the official sample solution.
* **`compile_and_profile.py`** — wrapper script that submits ONNX models to Qualcomm AI Hub for compilation and profiling.  You may need to install `qai_hub` and configure your token.
* **`inference.py`** — runs the compiled models on a dataset and computes retrieval metrics such as Recall@10.

These templates do **not** download or preprocess data.  You must provide a dataset (e.g., MS‑COCO or the competition dataset) and ensure that the `image_root` and `captions_json` paths in each model’s `config.yaml` are correct.  See [DATASET.md](DATASET.md) for the expected data layout and official track/dataset links.
