"""Run inference on compiled models and compute Recall@10.

This script downloads compiled image and text encoders from AI Hub using
their compile job IDs, runs them on a dataset of image–text pairs,
and computes retrieval metrics.  It is based on the official LPCVC
sample solution.  You must supply the compile job IDs obtained after
running `compile_and_profile.py`.

Usage:
    python inference.py --image_job <IMAGE_COMPILE_JOB_ID> --text_job <TEXT_COMPILE_JOB_ID> --dataset_id <DATASET_ID>

The `dataset_id` should correspond to an uploaded dataset created
with `upload_dataset.py`.
"""

import argparse
import qai_hub as hub
import torch
import numpy as np


def compute_recall(sim: np.ndarray, k: int) -> float:
    ranks = np.argsort(-sim, axis=1)
    gt = np.arange(sim.shape[0])
    correct = [1 if gt[i] in ranks[i, :k] else 0 for i in range(len(gt))]
    return np.mean(correct)


def main(image_job_id: str, text_job_id: str, dataset_id: str):
    # Fetch compiled models
    device = hub.Device('XR2 Gen 2 (Proxy)')
    img_model = hub.get_job(image_job_id).get_target_model()
    txt_model = hub.get_job(text_job_id).get_target_model()
    # Load dataset
    dataset = hub.Dataset(dataset_id)
    # Run inference in batches
    img_embeds = []
    txt_embeds = []
    for batch in dataset.iter_batches(max_batch_size=32):
        imgs = batch['image']
        txts = batch['txt']
        # Models return embeddings as numpy arrays
        img_embeds.append(img_model(imgs)['embedding'])
        txt_embeds.append(txt_model(txts)['embedding'])
    img_matrix = np.concatenate(img_embeds, axis=0)
    txt_matrix = np.concatenate(txt_embeds, axis=0)
    # Normalize and compute similarity
    img_norm = img_matrix / np.linalg.norm(img_matrix, axis=1, keepdims=True)
    txt_norm = txt_matrix / np.linalg.norm(txt_matrix, axis=1, keepdims=True)
    sim = np.matmul(img_norm, txt_norm.T)
    recall10 = compute_recall(sim, 10)
    recall5 = compute_recall(sim, 5)
    recall1 = compute_recall(sim, 1)
    print(f"Recall@1: {recall1:.3f}\nRecall@5: {recall5:.3f}\nRecall@10: {recall10:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on compiled SigLIP 2 models')
    parser.add_argument('--image_job', type=str, required=True)
    parser.add_argument('--text_job', type=str, required=True)
    parser.add_argument('--dataset_id', type=str, required=True)
    args = parser.parse_args()
    main(args.image_job, args.text_job, args.dataset_id)
