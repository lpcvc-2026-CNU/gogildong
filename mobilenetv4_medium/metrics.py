"""
Evaluation metrics for LPCVC 2026 Track 1.

Primary metric: Recall@10 (image→text retrieval and text→image retrieval).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


def compute_recall_at_k(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> dict:
    """
    Compute Recall@K for image-to-text and text-to-image retrieval.

    Assumes ground truth: i-th image matches i-th text.

    Args:
        image_embeds: (N, D) – L2-normalized image embeddings
        text_embeds:  (N, D) – L2-normalized text embeddings
        k_values:     List of K values to evaluate

    Returns:
        Dict with 'i2t_R@K' and 't2i_R@K' for each K.
    """
    # Ensure normalized
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    n = image_embeds.shape[0]
    labels = torch.arange(n)

    # Cosine similarity matrix (N x N)
    sim_matrix = image_embeds @ text_embeds.T  # (N, N)

    results = {}

    # Image → Text Retrieval (i2t)
    for k in k_values:
        topk_indices = sim_matrix.topk(k, dim=-1).indices  # (N, k)
        hits = sum(
            int(labels[i] in topk_indices[i])
            for i in range(n)
        )
        results[f"i2t_R@{k}"] = hits / n

    # Text → Image Retrieval (t2i)
    sim_matrix_t = sim_matrix.T  # (N, N) — text queries vs images
    for k in k_values:
        topk_indices = sim_matrix_t.topk(k, dim=-1).indices  # (N, k)
        hits = sum(
            int(labels[i] in topk_indices[i])
            for i in range(n)
        )
        results[f"t2i_R@{k}"] = hits / n

    # Mean R@10 (competition primary metric)
    results["mean_R@10"] = (results.get("i2t_R@10", 0) + results.get("t2i_R@10", 0)) / 2.0

    return results


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device: str = "cuda",
    k_values: List[int] = [1, 5, 10],
) -> dict:
    """
    Run full evaluation on a dataloader.

    Args:
        model:      StudentCLIP instance.
        dataloader: Evaluation DataLoader.
        device:     Compute device.
        k_values:   K values for Recall@K.

    Returns:
        Dict of Recall@K metrics.
    """
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    for batch in dataloader:
        student_images = batch["student_image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        image_embeds = model.encode_image(student_images)
        text_embeds = model.encode_text(input_ids, attention_mask)

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())

    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    return compute_recall_at_k(all_image_embeds, all_text_embeds, k_values)
