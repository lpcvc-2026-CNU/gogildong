"""Compile and profile ONNX encoders on Qualcomm AI Hub.

This script wraps the `qai_hub` Python API to upload the exported
ONNX models, compile them for a specified device, and optionally
profile their performance.  Before running this script you must
authenticate by running `qai-hub configure --api_token <YOUR_TOKEN>`.

Usage:
    python compile_and_profile.py --onnx_dir exported_onnx --img_name image_encoder.onnx --txt_name text_encoder.onnx --device "XR2 Gen 2 (Proxy)"

Set `--skip_profile` to upload and compile without profiling.
"""

import argparse
from pathlib import Path
import qai_hub as hub


def compile_and_profile(onnx_dir: str, img_name: str, txt_name: str, device: str, skip_profile: bool):
    onnx_dir = Path(onnx_dir)
    image_path = onnx_dir / img_name
    text_path = onnx_dir / txt_name
    if not image_path.exists() or not text_path.exists():
        raise FileNotFoundError(f"Missing ONNX files in {onnx_dir}")
    # Upload and compile image encoder
    d = hub.Device(device)
    print(f"Uploading {image_path} for compilation on {device}…")
    job = hub.submit_compile_job(model=image_path, device=d)
    print(f"Image compile job submitted: {job.job_id}")
    job.wait()
    compiled_image_model = job.get_target_model()
    if not skip_profile:
        prof_job = hub.submit_profile_job(model=compiled_image_model, device=d)
        print(f"Image profile job submitted: {prof_job.job_id}")
        prof_job.wait()
        print("Image profiling completed")
    # Upload and compile text encoder
    print(f"Uploading {text_path} for compilation on {device}…")
    job = hub.submit_compile_job(model=text_path, device=d)
    print(f"Text compile job submitted: {job.job_id}")
    job.wait()
    compiled_text_model = job.get_target_model()
    if not skip_profile:
        prof_job = hub.submit_profile_job(model=compiled_text_model, device=d)
        print(f"Text profile job submitted: {prof_job.job_id}")
        prof_job.wait()
        print("Text profiling completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile and profile ONNX models on AI Hub')
    parser.add_argument('--onnx_dir', type=str, default='exported_onnx')
    parser.add_argument('--img_name', type=str, default='image_encoder.onnx')
    parser.add_argument('--txt_name', type=str, default='text_encoder.onnx')
    parser.add_argument('--device', type=str, default='XR2 Gen 2 (Proxy)')
    parser.add_argument('--skip_profile', action='store_true', help='Skip profiling step')
    args = parser.parse_args()
    compile_and_profile(args.onnx_dir, args.img_name, args.txt_name, args.device, args.skip_profile)
