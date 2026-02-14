"""Export MobileCLIP2‑S4 student encoders to ONNX.

This script loads the best checkpoint produced by `train.py` and exports
the MobileCLIP2‑S4 student's vision and text encoders separately to
ONNX format.
"""

import argparse
import os
from pathlib import Path
import torch
import open_clip

def export_models(ckpt_path: str, out_dir: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    student_name = cfg['student_name']
    model, _, _ = open_clip.create_model_and_transforms(student_name, pretrained=None, precision='fp32')
    model.load_state_dict(ckpt['model'])
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    dummy_image = torch.randn(1, 3, 224, 224)
    tokenizer = open_clip.get_tokenizer(student_name)
    dummy_text = tokenizer(["a dummy caption"])
    # Export vision encoder
    torch.onnx.export(
        model.visual,
        dummy_image,
        os.path.join(out_dir, 'image_encoder.onnx'),
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={'image': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        opset_version=17,
    )
    # Export text encoder
    dummy_tokens = dummy_text['input_ids']
    torch.onnx.export(
        model.transformer,
        dummy_tokens,
        os.path.join(out_dir, 'text_encoder.onnx'),
        input_names=['input_ids'],
        output_names=['embedding'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        opset_version=17,
    )
    print(f"Exported ONNX models to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export MobileCLIP2 ONNX')
    parser.add_argument('--ckpt', type=str, default='runs/mobileclip2_s4/best.pt')
    parser.add_argument('--out_dir', type=str, default='exported_onnx')
    args = parser.parse_args()
    export_models(args.ckpt, args.out_dir)
