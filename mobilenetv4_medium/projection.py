"""
Projection head for single-teacher (SigLIP2) knowledge distillation.

StudentCLIP 의 shared embedding → SigLIP2 공간으로 투영합니다.
학습 시에만 사용되며, ONNX 추론 경로에는 포함되지 않습니다.
"""

import torch
import torch.nn as nn

from config import ConfigNode


class SigLIPProjectionHead(nn.Module):
    """
    Linear → GELU → Linear → LayerNorm 구조의 단일 투영 헤드.

    Input : (B, embed_dim)          — student shared embedding
    Output: (B, siglip_teacher_dim) — un-normalized projection
    """

    def __init__(self, cfg: ConfigNode):
        super().__init__()
        in_dim     = cfg.model.embed_dim
        hidden_dim = cfg.model.proj_sig_hidden_dim
        out_dim    = cfg.model.siglip_teacher_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
            nn.LayerNorm(out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
