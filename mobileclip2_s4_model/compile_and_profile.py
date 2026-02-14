"""Compile and profile MobileCLIP2 ONNX encoders on Qualcomm AI Hub.

Identical to the SigLIP variant but placed here for convenience.
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
    d = hub.Device(device)
    print(f"Uploading {image_path} for compilation on {device}…")
    job = hub.submit_compile_job(model=image_path, device=d)
    print(f"Image compile job submitted: {job.job_id}")
    job.wait()
    compiled_image_model = job.get_target_model()
    if not skip_profile:
        p = hub.submit_profile_job(model=compiled_image_model, device=d)
        print(f"Image profile job submitted: {p.job_id}")
        p.wait()
    print(f"Uploading {text_path} for compilation on {device}…")
    job = hub.submit_compile_job(model=text_path, device=d)
    print(f"Text compile job submitted: {job.job_id}")
    job.wait()
    compiled_text_model = job.get_target_model()
    if not skip_profile:
        p = hub.submit_profile_job(model=compiled_text_model, device=d)
        print(f"Text profile job submitted: {p.job_id}")
        p.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile and profile MobileCLIP2 models on AI Hub')
    parser.add_argument('--onnx_dir', type=str, default='exported_onnx')
    parser.add_argument('--img_name', type=str, default='image_encoder.onnx')
    parser.add_argument('--txt_name', type=str, default='text_encoder.onnx')
    parser.add_argument('--device', type=str, default='XR2 Gen 2 (Proxy)')
    parser.add_argument('--skip_profile', action='store_true')
    args = parser.parse_args()
    compile_and_profile(args.onnx_dir, args.img_name, args.txt_name, args.device, args.skip_profile)
