"""Compile and profile ViT-S/16 ONNX encoders on Qualcomm AI Hub.

Prerequisite:
    qai-hub configure --api_token <YOUR_TOKEN>

Typical workflow (project root):
    python vit_s_16/export_onnx.py          # export first
    python vit_s_16/compile_and_profile.py  # compile + profile
    python vit_s_16/compile_and_profile.py --skip_profile  # compile only

Fix history
───────────
v1  initial
v2  None-check on compiled model before profiling
v3  (이번) AI Hub는 dynamic shape 불가 → submit_compile_job에
    input_specs={name: (shape, dtype)} 로 정적 shape 명시.
    image  : (1, 3, 224, 224)  float32
    tokens : (1, 77)           int32   ← open_clip context_length default
    context_length / image_size 는 --context_length / --image_size 옵션으로 변경 가능.
"""

import argparse
import sys
from pathlib import Path

import qai_hub as hub


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compile(
    onnx_path: Path,
    device: hub.Device,
    label: str,
    input_specs: dict,
    target_runtime: str,
) -> tuple:
    """Submit compile job with static input_specs, wait, return (model, job_id)."""
    print(f"  Uploading {onnx_path.name} → {device.name} ...")
    print(f"  Input specs: { {k: v for k, v in input_specs.items()} }")

    job = hub.submit_compile_job(
        model=onnx_path,
        device=device,
        input_specs=input_specs,   # ← fixes "dynamic shapes" error
        options=f"--target_runtime {target_runtime}",
    )
    print(f"  {label} compile job ID : {job.job_id}  (waiting…)")
    job.wait()

    compiled = job.get_target_model()
    if compiled is None:
        print(
            f"  ❌ {label} compile FAILED\n"
            f"     → https://workbench.aihub.qualcomm.com/jobs/{job.job_id}/"
        )
    else:
        print(f"  ✓ {label} compile succeeded")
    return compiled, job.job_id


def _profile(compiled_model, device: hub.Device, label: str) -> None:
    if compiled_model is None:
        print(f"  ⚠ Skipping {label} profile (compile failed)")
        return
    prof = hub.submit_profile_job(model=compiled_model, device=device)
    print(f"  {label} profile job ID  : {prof.job_id}  (waiting…)")
    prof.wait()
    print(f"  ✓ {label} profile done")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def compile_and_profile(
    onnx_dir: str,
    img_name: str,
    txt_name: str,
    device_name: str,
    skip_profile: bool,
    image_size: int,
    context_length: int,
    target_runtime: str,
) -> None:
    onnx_dir   = Path(onnx_dir)
    image_path = onnx_dir / img_name
    text_path  = onnx_dir / txt_name

    if not image_path.exists() or not text_path.exists():
        raise FileNotFoundError(
            f"ONNX files not found in {onnx_dir}.\n"
            "Run export_onnx.py first."
        )

    device = hub.Device(device_name)

    # Static input shapes required by AI Hub
    # dtype strings accepted by qai_hub: "float32", "int32", "int64" …
    img_specs = {
        "image": ((1, 3, image_size, image_size), "float32"),
    }
    txt_specs = {
        # open_clip tokenizer output is int64 but most QNN backends expect int32
        "input_ids": ((1, context_length), "int32"),
    }

    # ── Image encoder ─────────────────────────────────────────────────────────
    print("\n[1/2] Image encoder")
    compiled_image, img_job_id = _compile(
        image_path, device, "Image", img_specs, target_runtime
    )
    if not skip_profile:
        _profile(compiled_image, device, "Image")

    # ── Text encoder ──────────────────────────────────────────────────────────
    print("\n[2/2] Text encoder")
    compiled_text, txt_job_id = _compile(
        text_path, device, "Text", txt_specs, target_runtime
    )
    if not skip_profile:
        _profile(compiled_text, device, "Text")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE — pass these IDs to inference.py:")
    if compiled_image:
        print(f"  --image_job {img_job_id}")
    else:
        print(f"  --image_job  ← FAILED  (job {img_job_id})")
    if compiled_text:
        print(f"  --text_job  {txt_job_id}")
    else:
        print(f"  --text_job   ← FAILED  (job {txt_job_id})")
    print("=" * 60)

    if compiled_image is None or compiled_text is None:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and profile ViT-S/16 models on Qualcomm AI Hub"
    )
    parser.add_argument("--onnx_dir",       default="vit_s_16/exported_onnx")
    parser.add_argument("--img_name",       default="image_encoder.onnx")
    parser.add_argument("--txt_name",       default="text_encoder.onnx")
    parser.add_argument("--device",         default="XR2 Gen 2 (Proxy)")
    parser.add_argument("--skip_profile",   action="store_true",
                        help="Compile only, skip profiling")
    parser.add_argument("--image_size",     type=int, default=224,
                        help="Image resolution (default: 224)")
    parser.add_argument("--context_length", type=int, default=77,
                        help="Token sequence length (default: 77, open_clip ViT-S-16)")
    parser.add_argument(
        "--target_runtime",
        default="qnn_context_binary",
        help=(
            "QAI Hub target runtime passed to compile options "
            "(default: qnn_context_binary)"
        ),
    )
    args = parser.parse_args()
    compile_and_profile(
        args.onnx_dir, args.img_name, args.txt_name,
        args.device, args.skip_profile,
        args.image_size, args.context_length,
        args.target_runtime,
    )
