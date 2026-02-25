"""Export ViT-S/16 student encoders to ONNX (single self-contained file).

Fix history
───────────
v3  disable fused MHA, opset 18
v4  - ONNX 외부 가중치 파일(*.onnx_data) 생성 방지:
      export 후 onnx.save(..., all_tensors_to_one_file=True) 로
      가중치를 .onnx 안에 인라인 임베드.
      → AI Hub "missing external weights" 오류 해결.
v5  - Text encoder dummy tokens cast to int32 before export.
      QAI Hub requires int32; exporting as int64 causes compile failure:
      "Provided input_shapes does not match shapes inferred from the model".

Usage (project root):
    python vit_s_16/export_onnx.py
    python vit_s_16/export_onnx.py --ckpt runs/vit_s_16/best.pt --out_dir vit_s_16/exported_onnx
"""

import argparse
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import open_clip
import onnx


# ─────────────────────────────────────────────────────────────────────────────
# Disable fused / flash attention so ONNX export can trace individual ops.
# aten::_native_multi_head_attention is not supported in opset ≤ 17.
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def disable_fused_attention():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            yield
    except ImportError:
        prev_flash   = torch.backends.cuda.flash_sdp_enabled()
        prev_mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
        prev_math    = torch.backends.cuda.math_sdp_enabled()
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            yield
        finally:
            torch.backends.cuda.enable_flash_sdp(prev_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(prev_mem_eff)
            torch.backends.cuda.enable_math_sdp(prev_math)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper modules
# ─────────────────────────────────────────────────────────────────────────────

class ImageEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._model.encode_image(image)


class TextEncoderWrapper(nn.Module):
    """encode_text(): token IDs → embedding (full pipeline).

    model.transformer alone only accepts embedded hidden states, NOT raw
    token IDs — wrapping encode_text() avoids the shape mismatch.
    """
    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._model.encode_text(input_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: export to temp dir, then inline all external weights into one file
# ─────────────────────────────────────────────────────────────────────────────

def export_inline(module, dummy_input, final_path: str, input_names, output_names,
                  dynamic_axes, opset: int) -> None:
    """Export ONNX and ensure all weights are inlined (no external data files).

    PyTorch may split large models into <name>.onnx + <name>.onnx_data.
    AI Hub requires a single self-contained .onnx file.
    We export to a temp directory then use onnx.save() with
    size_threshold=0 to force all tensors into the protobuf.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_onnx = os.path.join(tmp, "model.onnx")

        with torch.no_grad(), disable_fused_attention():
            torch.onnx.export(
                module,
                dummy_input,
                tmp_onnx,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
            )

        # Load and re-save with all tensors inlined
        proto = onnx.load(tmp_onnx, load_external_data=True)
        onnx.save(
            proto,
            final_path,
            save_as_external_data=False,   # ← inline everything
        )

    size_mb = os.path.getsize(final_path) / 1024 ** 2
    print(f"  saved: {final_path}  ({size_mb:.1f} MB, single file)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def export_models(ckpt_path: str, out_dir: str) -> None:
    print(f"Loading checkpoint : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg          = ckpt["config"]
    student_name = cfg["student_name"]
    print(f"Student model name : {student_name}")

    model, _, _ = open_clip.create_model_and_transforms(
        student_name, pretrained=None, precision="fp32"
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:3]} ...")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:3]} ...")
    model.eval()

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Dummy inputs
    dummy_image  = torch.randn(1, 3, 224, 224)
    tokenizer    = open_clip.get_tokenizer(student_name)
    dummy_tokens = tokenizer(["a dummy caption"])
    if isinstance(dummy_tokens, dict):
        dummy_tokens = dummy_tokens["input_ids"]
    if dummy_tokens.dim() == 3:
        dummy_tokens = dummy_tokens.squeeze(1)
    # Cast to int32: QAI Hub NPU backend requires int32 for token inputs.
    # Exporting as int64 (PyTorch default) causes a dtype mismatch error
    # at compile time: "Provided input_shapes does not match shapes inferred
    # from the model".
    dummy_tokens = dummy_tokens.to(torch.int32)

    print(f"  image  shape : {dummy_image.shape}")
    print(f"  tokens shape : {dummy_tokens.shape}  dtype: {dummy_tokens.dtype}")

    OPSET = 18  # first opset with native MultiHeadAttention op

    # ── Image encoder ─────────────────────────────────────────────────────────
    image_out = os.path.join(out_dir, "image_encoder.onnx")
    print(f"\n[1/2] Exporting image encoder → opset {OPSET}")
    export_inline(
        ImageEncoderWrapper(model).eval(),
        (dummy_image,),
        image_out,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset=OPSET,
    )
    print("  ✓ image encoder done")

    # ── Text encoder ──────────────────────────────────────────────────────────
    text_out = os.path.join(out_dir, "text_encoder.onnx")
    print(f"\n[2/2] Exporting text encoder  → opset {OPSET}")
    export_inline(
        TextEncoderWrapper(model).eval(),
        (dummy_tokens,),
        text_out,
        input_names=["input_ids"],
        output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset=OPSET,
    )
    print("  ✓ text encoder done")

    # Warn if stray external data files exist (shouldn't after inline save)
    stray = list(Path(out_dir).glob("*.onnx_data"))
    if stray:
        print(f"\n  [warn] Unexpected external data files found — removing: {stray}")
        for f in stray:
            f.unlink()

    # ── Sanity check ──────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        print("\nSanity-checking with onnxruntime...")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        sess = ort.InferenceSession(image_out, opts, providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"image": dummy_image.numpy()})
        print(f"  image_encoder output : {out[0].shape}  ✓")

        sess = ort.InferenceSession(text_out, opts, providers=["CPUExecutionProvider"])
        out  = sess.run(None, {"input_ids": dummy_tokens.numpy()})
        print(f"  text_encoder  output : {out[0].shape}  ✓")
    except ImportError:
        print("  (onnxruntime not installed — skipping sanity check)")
    except Exception as e:
        print(f"  [warn] Sanity check failed: {e}")

    print(f"\n✅ ONNX files saved to: {out_dir}")
    print("   Next → python vit_s_16/compile_and_profile.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ViT-S/16 ONNX models")
    parser.add_argument("--ckpt",    default="runs/vit_s_16/best.pt")
    parser.add_argument("--out_dir", default="vit_s_16/exported_onnx")
    args = parser.parse_args()
    export_models(args.ckpt, args.out_dir)