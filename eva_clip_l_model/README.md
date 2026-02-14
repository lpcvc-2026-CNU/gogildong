# EVA‑02‑CLIP‑L Student Model

This project fine‑tunes the **EVA‑02‑CLIP‑L** (428 M parameters) vision–language model for image–text retrieval in LPCVC 2026.  EVA‑CLIP is an open‑source CLIP series that achieves state‑of‑the‑art performance on ImageNet zero‑shot classification and COCO retrieval【412239981389457†L80-L83】.  The L‑size model provides a balance between accuracy and size and serves as the student network.

## Model choices

* **Student** – `EVA02_CLIP_L`.  This model is built on the EVA ViT architecture with a 14‑patch vision transformer and 428 M parameters.  It is considerably larger than the mobile models but fits within the competition’s latency and size constraints when distilled and quantised.
* **Teacher** – You may choose one of:
  * **EVA‑02‑CLIP‑E** (≈4.4 B parameters).  According to the EVA‑CLIP model card, the E‑size model is used as a teacher for smaller EVA models【412239981389457†L69-L74】.
  * **CLIP‑ViT‑bigG‑14 (LAION)**.  Another strong publicly available CLIP variant that can provide a complementary representation to EVA【412239981389457†L69-L74】.

## Training procedure

1. **Dataset preparation** – use the [LPCV 2026 Image-Text Retrieval](https://lpcv.ai/2026LPCVC/image-text-retrieval/) dataset; see [DATASET.md](../DATASET.md) for the expected layout.  Organise image files and captions as in the other projects.  Large models benefit from more data; consider MS‑COCO or additional pairs if allowed by competition rules.
2. **Load models** – `train.py` loads the EVA‑02‑CLIP‑L student with `open_clip` and the chosen teacher model.  Because the student is large, consider using gradient accumulation or reducing batch size to fit on GPU memory.
3. **Losses** – training follows the standard CLIP contrastive loss plus optional distillation loss from the teacher.  EVA‑CLIP uses the same cross‑entropy loss as OpenAI‑CLIP.
4. **Optimisation** – due to the model’s size, learning rate and weight decay must be tuned carefully.  The default configuration uses a smaller batch size and learning rate.
5. **Evaluation** – compute Recall@K on the validation set after each epoch and save the best checkpoint.

## Exporting and deployment

The export, compile, and inference scripts mirror those of the other projects.  After training, run `export_onnx.py` to produce ONNX files, compile them via `compile_and_profile.py`, and evaluate performance with `inference.py`.

## Notes

* EVA‑CLIP models are open‑sourced under a permissive license, making them suitable for competition submissions.
* Because EVA‑02‑CLIP‑L is comparatively large, the ONNX files may need to be quantised (e.g., dynamic INT8) to fit within 50 MB.
