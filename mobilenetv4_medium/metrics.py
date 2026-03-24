"""
Evaluation metrics for LPCVC 2026 Track 1.
Primary metric: Recall@10 (image→text and text→image retrieval).
"""

import torch
import torch.nn.functional as F
from typing import List


def compute_recall_at_k(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> dict:
    """
    Recall@K 계산. i-th 이미지 ↔ i-th 텍스트가 정답 쌍이라고 가정.

    Args:
        image_embeds: (N, D) — L2-normalized
        text_embeds:  (N, D) — L2-normalized
    """
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds  = F.normalize(text_embeds,  dim=-1)

    n      = image_embeds.shape[0]
    labels = torch.arange(n)
    sim    = image_embeds @ text_embeds.T   # (N, N)

    results = {}

    # Image → Text
    for k in k_values:
        topk = sim.topk(k, dim=-1).indices
        hits = sum(int(labels[i] in topk[i]) for i in range(n))
        results[f"i2t_R@{k}"] = hits / n

    # Text → Image
    for k in k_values:
        topk = sim.T.topk(k, dim=-1).indices
        hits = sum(int(labels[i] in topk[i]) for i in range(n))
        results[f"t2i_R@{k}"] = hits / n

    results["mean_R@10"] = (
        results.get("i2t_R@10", 0) + results.get("t2i_R@10", 0)
    ) / 2.0

    return results


@torch.no_grad()
def evaluate_model(model, dataloader, device: str = "cuda",
                   k_values: List[int] = [1, 5, 10]) -> dict:
    model.eval()
    all_img, all_txt = [], []

    for batch in dataloader:
        images = batch["student_image"].to(device)
        ids    = batch["student_input_ids"].to(device)       # ← 수정된 키
        mask   = batch["student_attention_mask"].to(device)  # ← 수정된 키

        all_img.append(model.encode_image(images).cpu())
        all_txt.append(model.encode_text(ids, mask).cpu())

    return compute_recall_at_k(
        torch.cat(all_img), torch.cat(all_txt), k_values
    )
