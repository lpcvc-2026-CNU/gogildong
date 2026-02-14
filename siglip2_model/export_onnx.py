"""Export SigLIP 2 student encoders to ONNX.

This script loads the best checkpoint produced by `train.py` and exports
the image and text encoders separately to ONNX format.  The resulting
`.onnx` files can be compiled and profiled using Qualcomm AI Hub.

Usage:
    python export_onnx.py --ckpt runs/siglip2/best.pt --out_dir exported_onnx
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
    state_dict = ckpt['model']
    model.load_state_dict(state_dict)
    model.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Dummy inputs
    dummy_image = torch.randn(1, 3, 224, 224)
    tokenizer = open_clip.get_tokenizer(student_name)
    dummy_text = tokenizer(["a dummy caption"])
    for k in dummy_text:
        dummy_text[k] = dummy_text[k]
    # Export image encoder
    image_out_path = os.path.join(out_dir, 'image_encoder.onnx')
    torch.onnx.export(
        model.visual,
        dummy_image,
        image_out_path,
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={'image': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        opset_version=17,
    )
    # Export text encoder
    text_out_path = os.path.join(out_dir, 'text_encoder.onnx')
    # Flatten dummy text dict to tensor input; open_clip’s transformer accepts token ids
    dummy_tokens = dummy_text['input_ids']
    torch.onnx.export(
        model.transformer,
        dummy_tokens,
        text_out_path,
        input_names=['input_ids'],
        output_names=['embedding'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}},
        opset_version=17,
    )
    print(f"Exported image encoder to {image_out_path}\nExported text encoder to {text_out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export SigLIP 2 ONNX')
    parser.add_argument('--ckpt', type=str, default='runs/siglip2/best.pt', help='Path to student checkpoint')
    parser.add_argument('--out_dir', type=str, default='exported_onnx', help='Output directory')
    args = parser.parse_args()
    export_models(args.ckpt, args.out_dir)
