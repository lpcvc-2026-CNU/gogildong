"""Run inference with compiled ViT-S/16 models and compute retrieval metrics.

Usage:
    python vit_s_16/inference.py \
        --image_job <IMAGE_COMPILE_JOB_ID> \
        --text_job  <TEXT_COMPILE_JOB_ID>  \
        --dataset_id <DATASET_ID>

The compile job IDs are printed by compile_and_profile.py.
The dataset_id is the ID of a dataset uploaded to Qualcomm AI Hub
(e.g., via the official sample solution's upload_dataset.py script).

Outputs R@1, R@5, R@10 to stdout.
"""

import argparse

import numpy as np
import qai_hub as hub


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_recall(sim: np.ndarray, k: int) -> float:
    """Recall@k: fraction of queries whose ground-truth is in top-k results."""
    ranks = np.argsort(-sim, axis=1)          # descending similarity
    gt = np.arange(sim.shape[0])              # diagonal ground-truth
    correct = [1 if gt[i] in ranks[i, :k] else 0 for i in range(len(gt))]
    return float(np.mean(correct))


def compute_all_recalls(sim: np.ndarray) -> dict:
    return {
        "R@1":  compute_recall(sim, 1),
        "R@5":  compute_recall(sim, 5),
        "R@10": compute_recall(sim, 10),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(image_job_id: str, text_job_id: str, dataset_id: str, batch_size: int) -> None:
    device = hub.Device("XR2 Gen 2 (Proxy)")

    print(f"Fetching compiled image encoder (job: {image_job_id}) ...")
    img_model = hub.get_job(image_job_id).get_target_model()
    print(f"Fetching compiled text  encoder (job: {text_job_id}) ...")
    txt_model = hub.get_job(text_job_id).get_target_model()

    dataset = hub.Dataset(dataset_id)
    print(f"Dataset loaded: {dataset_id}")

    img_embeds = []
    txt_embeds = []

    print("Running inference...")
    for i, batch in enumerate(dataset.iter_batches(max_batch_size=batch_size)):
        imgs = batch["image"]
        txts = batch["txt"]
        img_embeds.append(img_model(imgs)["embedding"])
        txt_embeds.append(txt_model(txts)["embedding"])
        if (i + 1) % 10 == 0:
            print(f"  Processed {(i + 1) * batch_size} samples...")

    img_matrix = np.concatenate(img_embeds, axis=0)
    txt_matrix = np.concatenate(txt_embeds, axis=0)

    # L2 normalise
    img_norm = img_matrix / np.linalg.norm(img_matrix, axis=1, keepdims=True)
    txt_norm = txt_matrix / np.linalg.norm(txt_matrix, axis=1, keepdims=True)

    # Cosine similarity matrix  (N x N)
    sim = np.matmul(img_norm, txt_norm.T)

    recalls = compute_all_recalls(sim)
    print("\n=== Retrieval Metrics ===")
    for k, v in recalls.items():
        print(f"  {k}:  {v:.4f}")
    print("========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ViT-S/16 inference on Qualcomm AI Hub compiled models"
    )
    parser.add_argument(
        "--image_job",
        required=True,
        help="Compile job ID for the image encoder (from compile_and_profile.py)",
    )
    parser.add_argument(
        "--text_job",
        required=True,
        help="Compile job ID for the text encoder (from compile_and_profile.py)",
    )
    parser.add_argument(
        "--dataset_id",
        required=True,
        help="AI Hub dataset ID (from upload_dataset.py)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    args = parser.parse_args()
    main(args.image_job, args.text_job, args.dataset_id, args.batch_size)