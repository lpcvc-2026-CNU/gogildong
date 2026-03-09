"""
ONNX export utilities for LPCVC 2026 Track 1.

Exports separate ONNX models for image and text encoders.
Target: Qualcomm AI Hub (Snapdragon 8 Elite / X Elite NPU)

Output spec:
  image_encoder.onnx  : Input (1, 3, 224, 224) → Output (1, embed_dim)
  text_encoder.onnx   : Input input_ids (1, 77), attention_mask (1, 77) → Output (1, embed_dim)
"""

import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.student import StudentCLIP
from utils.config import ModelConfig, ExportConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wrapper modules for clean ONNX export
# ---------------------------------------------------------------------------

class ImageEncoderONNX(nn.Module):
    """
    Thin wrapper: forward returns L2-normalized image embeddings.
    No projection heads needed at inference time.
    """

    def __init__(self, model: StudentCLIP):
        super().__init__()
        self.image_encoder = model.image_encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            embeddings: (B, embed_dim) – L2-normalized
        """
        feats = self.image_encoder(pixel_values)
        return F.normalize(feats, dim=-1)


class TextEncoderONNX(nn.Module):
    """
    Thin wrapper: forward returns L2-normalized text embeddings.
    """

    def __init__(self, model: StudentCLIP):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, 77)
            attention_mask: (B, 77)
        Returns:
            embeddings: (B, embed_dim) – L2-normalized
        """
        feats = self.text_encoder(input_ids, attention_mask)
        return F.normalize(feats, dim=-1)


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_image_encoder(
    model: StudentCLIP,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
) -> None:
    """Export image encoder to ONNX."""
    model.eval()
    wrapper = ImageEncoderONNX(model).to(device)
    wrapper.eval()

    # Dummy input: (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
        do_constant_folding=True,
        export_params=True,
    )
    logger.info(f"[Export] Image encoder saved to: {output_path}")


def export_text_encoder(
    model: StudentCLIP,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
) -> None:
    """Export text encoder to ONNX."""
    model.eval()
    wrapper = TextEncoderONNX(model).to(device)
    wrapper.eval()

    # Dummy inputs: (1, 77) int64
    dummy_input_ids = torch.zeros(1, 77, dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(1, 77, dtype=torch.long, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"},
        },
        do_constant_folding=True,
        export_params=True,
    )
    logger.info(f"[Export] Text encoder saved to: {output_path}")


def verify_onnx_outputs(
    model: StudentCLIP,
    image_onnx_path: str,
    text_onnx_path: str,
    device: str = "cpu",
) -> bool:
    """
    Verify ONNX model outputs match PyTorch model outputs.
    Returns True if outputs are close enough.
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        logger.warning("onnxruntime not installed, skipping verification.")
        return False

    model.eval()

    # Test inputs
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_ids = torch.zeros(1, 77, dtype=torch.long)
    dummy_mask = torch.ones(1, 77, dtype=torch.long)

    with torch.no_grad():
        pt_img_emb = model.encode_image(dummy_image).numpy()
        pt_txt_emb = model.encode_text(dummy_ids, dummy_mask).numpy()

    # ONNX Runtime inference
    img_session = ort.InferenceSession(image_onnx_path)
    txt_session = ort.InferenceSession(text_onnx_path)

    ort_img_emb = img_session.run(
        None, {"pixel_values": dummy_image.numpy()}
    )[0]
    ort_txt_emb = txt_session.run(
        None, {
            "input_ids": dummy_ids.numpy(),
            "attention_mask": dummy_mask.numpy(),
        }
    )[0]

    img_close = np.allclose(pt_img_emb, ort_img_emb, atol=1e-4)
    txt_close = np.allclose(pt_txt_emb, ort_txt_emb, atol=1e-4)

    logger.info(f"[Verify] Image encoder match: {img_close}")
    logger.info(f"[Verify] Text encoder match:  {txt_close}")

    return img_close and txt_close


def export_all(
    model: StudentCLIP,
    export_cfg: ExportConfig,
    device: str = "cpu",
) -> dict:
    """
    Full export pipeline: save both ONNX models.

    Args:
        model:      Trained StudentCLIP.
        export_cfg: Export configuration.
        device:     CPU recommended for ONNX export.

    Returns:
        Dict with paths to exported files.
    """
    output_dir = Path(export_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = str(output_dir / "image_encoder.onnx")
    text_path = str(output_dir / "text_encoder.onnx")

    export_image_encoder(model, image_path, export_cfg.onnx_opset, device)
    export_text_encoder(model, text_path, export_cfg.onnx_opset, device)

    # Verify
    verify_onnx_outputs(model, image_path, text_path, device)

    return {"image_encoder": image_path, "text_encoder": text_path}
