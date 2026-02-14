"""Run inference with compiled EVA‑02‑CLIP‑L models.

This script mirrors the inference scripts used for SigLIP and MobileCLIP
projects.  Pass the job IDs of the compiled image and text encoders
along with the dataset ID produced by `upload_dataset.py`.
"""

import argparse
import numpy as np
import qai_hub as hub

def compute_recall(sim: np.ndarray, k: int) -> float:
    ranks = np.argsort(-sim, axis=1)
    gt = np.arange(sim.shape[0])
    correct = [1 if gt[i] in ranks[i, :k] else 0 for i in range(len(gt))]
    return np.mean(correct)

def main(image_job: str, text_job: str, dataset_id: str):
    device = hub.Device('XR2 Gen 2 (Proxy)')
    img_model = hub.get_job(image_job).get_target_model()
    txt_model = hub.get_job(text_job).get_target_model()
    dataset = hub.Dataset(dataset_id)
    img_embeds = []
    txt_embeds = []
    for batch in dataset.iter_batches(max_batch_size=16):
        imgs = batch['image']
        txts = batch['txt']
        img_embeds.append(img_model(imgs)['embedding'])
        txt_embeds.append(txt_model(txts)['embedding'])
    img_matrix = np.concatenate(img_embeds, axis=0)
    txt_matrix = np.concatenate(txt_embeds, axis=0)
    img_norm = img_matrix / np.linalg.norm(img_matrix, axis=1, keepdims=True)
    txt_norm = txt_matrix / np.linalg.norm(txt_matrix, axis=1, keepdims=True)
    sim = np.matmul(img_norm, txt_norm.T)
    print(f"R@1: {compute_recall(sim, 1):.3f}\nR@5: {compute_recall(sim, 5):.3f}\nR@10: {compute_recall(sim, 10):.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVA‑02‑CLIP‑L inference')
    parser.add_argument('--image_job', required=True)
    parser.add_argument('--text_job', required=True)
    parser.add_argument('--dataset_id', required=True)
    args = parser.parse_args()
    main(args.image_job, args.text_job, args.dataset_id)
