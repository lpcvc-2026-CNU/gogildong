"""
LPCVC 2026 Track 1 — ONNX 내보내기 및 Qualcomm AI Hub 제출 스크립트.

26LPCVC_Track1_Sample_Solution (lpcvai/26LPCVC_Track1_Sample_Solution) 의
공식 제출 방식을 기반으로 작성되었습니다.

전체 워크플로우:
  1. 학습된 PyTorch 모델 로드
  2. 이미지/텍스트 인코더를 분리하여 ONNX 내보내기
  3. qai_hub.submit_compile_job() 으로 각 모델 컴파일 (Qualcomm XR2 Gen 2)
  4. 컴파일 완료 대기 및 결과 확인
  5. (선택) 양자화: qai_hub.submit_quantize_job()
  6. (선택) 프로파일링: qai_hub.submit_profile_job() 으로 지연시간 측정

Usage:
    # ONNX 내보내기만
    python scripts/export_onnx.py --checkpoint checkpoints/stage3_epoch005.pt --onnx_only

    # ONNX 내보내기 + AI Hub 컴파일
    python scripts/export_onnx.py --checkpoint checkpoints/stage3_epoch005.pt

    # ONNX + 컴파일 + 양자화(INT8) + 프로파일링
    python scripts/export_onnx.py --checkpoint checkpoints/stage3_epoch005.pt --quantize --profile

    # 커스텀 설정 파일
    python scripts/export_onnx.py --checkpoint ... --config config.yaml
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from utils.config import load_config, ConfigNode
from models.student import StudentCLIP, build_student_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX 래퍼 모듈 (추론 전용, projection head 없음)
# ---------------------------------------------------------------------------

class ImageEncoderONNX(nn.Module):
    """
    ONNX 내보내기 전용 이미지 인코더 래퍼.
    입력  : pixel_values (1, 3, 224, 224)
    출력  : image_embeddings (1, embed_dim) — L2-normalized
    """

    def __init__(self, model: StudentCLIP):
        super().__init__()
        self.image_encoder = model.image_encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(pixel_values)
        return F.normalize(feats, dim=-1)


class TextEncoderONNX(nn.Module):
    """
    ONNX 내보내기 전용 텍스트 인코더 래퍼.
    입력  : input_ids (1, 77), attention_mask (1, 77)
    출력  : text_embeddings (1, embed_dim) — L2-normalized
    """

    def __init__(self, model: StudentCLIP):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        feats = self.text_encoder(input_ids, attention_mask)
        return F.normalize(feats, dim=-1)


# ---------------------------------------------------------------------------
# ONNX 내보내기
# ---------------------------------------------------------------------------

def export_to_onnx(model: StudentCLIP, cfg: ConfigNode, output_dir: Path) -> dict:
    """
    이미지 인코더와 텍스트 인코더를 각각 ONNX 파일로 내보냅니다.

    Returns:
        {'image': Path, 'text': Path}
    """
    model.eval()
    opset   = cfg.export.onnx_opset
    img_size = cfg.model.student_image_input_size
    seq_len  = cfg.model.max_text_length

    image_onnx_path = output_dir / "image_encoder.onnx"
    text_onnx_path  = output_dir / "text_encoder.onnx"

    # --- 이미지 인코더 ---
    logger.info(f"이미지 인코더 ONNX 내보내기 → {image_onnx_path}")
    img_wrapper   = ImageEncoderONNX(model).cpu().eval()
    dummy_image   = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        img_wrapper,
        dummy_image,
        str(image_onnx_path),
        opset_version=opset,
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        dynamic_axes={
            "pixel_values":     {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        },
        do_constant_folding=True,
        export_params=True,
    )

    # --- 텍스트 인코더 ---
    logger.info(f"텍스트 인코더 ONNX 내보내기 → {text_onnx_path}")
    txt_wrapper        = TextEncoderONNX(model).cpu().eval()
    dummy_input_ids    = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_attn_mask    = torch.ones(1, seq_len, dtype=torch.long)

    torch.onnx.export(
        txt_wrapper,
        (dummy_input_ids, dummy_attn_mask),
        str(text_onnx_path),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeddings"],
        dynamic_axes={
            "input_ids":       {0: "batch_size"},
            "attention_mask":  {0: "batch_size"},
            "text_embeddings": {0: "batch_size"},
        },
        do_constant_folding=True,
        export_params=True,
    )

    return {"image": image_onnx_path, "text": text_onnx_path}


def verify_onnx(model: StudentCLIP, paths: dict, cfg: ConfigNode) -> bool:
    """PyTorch 출력과 ONNX Runtime 출력을 비교하여 검증합니다."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime 미설치 — ONNX 검증 건너뜀.")
        return True

    img_size = cfg.model.student_image_input_size
    seq_len  = cfg.model.max_text_length
    model.eval()

    dummy_image   = torch.randn(1, 3, img_size, img_size)
    dummy_ids     = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_mask    = torch.ones(1, seq_len, dtype=torch.long)

    with torch.no_grad():
        pt_img = model.encode_image(dummy_image).numpy()
        pt_txt = model.encode_text(dummy_ids, dummy_mask).numpy()

    img_sess = ort.InferenceSession(str(paths["image"]))
    txt_sess = ort.InferenceSession(str(paths["text"]))

    ort_img = img_sess.run(None, {"pixel_values": dummy_image.numpy()})[0]
    ort_txt = txt_sess.run(None, {
        "input_ids": dummy_ids.numpy(),
        "attention_mask": dummy_mask.numpy(),
    })[0]

    img_ok = np.allclose(pt_img, ort_img, atol=1e-4)
    txt_ok = np.allclose(pt_txt, ort_txt, atol=1e-4)

    logger.info(f"[검증] 이미지 인코더 일치: {img_ok}")
    logger.info(f"[검증] 텍스트 인코더 일치: {txt_ok}")
    return img_ok and txt_ok


# ---------------------------------------------------------------------------
# Qualcomm AI Hub 컴파일
# ---------------------------------------------------------------------------

def compile_on_aihub(paths: dict, cfg: ConfigNode) -> dict:
    """
    qai_hub.submit_compile_job() 으로 두 모델을 Qualcomm AI Hub 에 컴파일 제출.

    Returns:
        {'image': CompileJob, 'text': CompileJob}
    """
    try:
        import qai_hub as hub
    except ImportError:
        logger.error("qai_hub 미설치. `pip install qai_hub` 후 재실행하세요.")
        raise

    hub_cfg  = cfg.export.qai_hub
    device   = hub.Device(hub_cfg.device)
    options  = hub_cfg.compile_options or ""
    img_size = cfg.model.student_image_input_size
    seq_len  = cfg.model.max_text_length

    logger.info(f"[AI Hub] 타겟 디바이스: {hub_cfg.device}")
    logger.info(f"[AI Hub] target_runtime: {hub_cfg.target_runtime}")

    # 이미지 인코더 컴파일
    logger.info(f"[AI Hub] 이미지 인코더 컴파일 제출...")
    img_compile_job = hub.submit_compile_job(
        model=str(paths["image"]),
        device=device,
        name=cfg.export.image_model_name,
        input_specs={"pixel_values": ((1, 3, img_size, img_size), "float32")},
        options=f"--target_runtime {hub_cfg.target_runtime} {options}".strip(),
    )
    logger.info(f"[AI Hub] 이미지 인코더 Job ID: {img_compile_job.job_id}")

    # 텍스트 인코더 컴파일
    logger.info(f"[AI Hub] 텍스트 인코더 컴파일 제출...")
    txt_compile_job = hub.submit_compile_job(
        model=str(paths["text"]),
        device=device,
        name=cfg.export.text_model_name,
        input_specs={
            "input_ids":      ((1, seq_len), "int32"),
            "attention_mask": ((1, seq_len), "int32"),
        },
        options=f"--target_runtime {hub_cfg.target_runtime} {options}".strip(),
    )
    logger.info(f"[AI Hub] 텍스트 인코더 Job ID: {txt_compile_job.job_id}")

    return {"image": img_compile_job, "text": txt_compile_job}


def wait_for_compile(jobs: dict) -> dict:
    """
    컴파일 완료 대기 후 상태 반환.
    성공하면 각 job 의 target model 을 반환합니다.
    """
    results = {}
    for name, job in jobs.items():
        logger.info(f"[AI Hub] {name} 인코더 컴파일 완료 대기 중...")
        job.wait()
        status = job.get_status()
        logger.info(f"[AI Hub] {name} 인코더 컴파일 상태: {status.code} — {status.message}")
        if hasattr(status, "code") and str(status.code) == "DONE":
            results[name] = job.get_target_model()
        else:
            logger.error(f"[AI Hub] {name} 인코더 컴파일 실패: {status.message}")
            results[name] = None
    return results


# ---------------------------------------------------------------------------
# INT8 양자화 (선택)
# ---------------------------------------------------------------------------

def quantize_on_aihub(paths: dict, cfg: ConfigNode) -> dict:
    """
    qai_hub.submit_quantize_job() 으로 INT8 양자화 수행.
    양자화 보정 데이터는 랜덤 샘플을 사용합니다.

    Returns:
        양자화된 ONNX 모델 경로 dict {'image': ..., 'text': ...}
    """
    try:
        import qai_hub as hub
    except ImportError:
        logger.error("qai_hub 미설치.")
        raise

    hub_cfg = cfg.export.qai_hub
    img_size = cfg.model.student_image_input_size
    seq_len  = cfg.model.max_text_length
    n_calib  = hub_cfg.calibration_samples

    logger.info(f"[AI Hub] INT8 양자화 (보정 샘플 수: {n_calib})")

    # 이미지 인코더 양자화
    img_calib = {"pixel_values": [
        np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        for _ in range(n_calib)
    ]}
    img_q_job = hub.submit_quantize_job(
        model=str(paths["image"]),
        calibration_data=img_calib,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name=f"{cfg.export.image_model_name}_int8",
    )
    logger.info(f"[AI Hub] 이미지 인코더 양자화 Job ID: {img_q_job.job_id}")

    # 텍스트 인코더 양자화
    txt_calib = {
        "input_ids":      [np.zeros((1, seq_len), dtype=np.int32) for _ in range(n_calib)],
        "attention_mask": [np.ones((1, seq_len), dtype=np.int32)  for _ in range(n_calib)],
    }
    txt_q_job = hub.submit_quantize_job(
        model=str(paths["text"]),
        calibration_data=txt_calib,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name=f"{cfg.export.text_model_name}_int8",
    )
    logger.info(f"[AI Hub] 텍스트 인코더 양자화 Job ID: {txt_q_job.job_id}")

    # 완료 대기 및 다운로드
    output_dir = Path(cfg.export.output_dir)
    q_paths = {}
    for name, job, orig_path in [
        ("image", img_q_job, paths["image"]),
        ("text",  txt_q_job, paths["text"]),
    ]:
        job.wait()
        q_out = str(output_dir / orig_path.name.replace(".onnx", "_int8.onnx"))
        job.download_target_model(q_out)
        logger.info(f"[AI Hub] {name} 인코더 양자화 모델 저장: {q_out}")
        q_paths[name] = Path(q_out)

    return q_paths


# ---------------------------------------------------------------------------
# 프로파일링 (선택)
# ---------------------------------------------------------------------------

def profile_on_aihub(compiled_models: dict, cfg: ConfigNode):
    """
    컴파일된 모델을 실제 디바이스에서 프로파일링하여 지연시간을 측정합니다.
    이미지 + 텍스트 합산 35ms 미만인지 확인하세요.
    """
    try:
        import qai_hub as hub
    except ImportError:
        logger.error("qai_hub 미설치.")
        return

    hub_cfg = cfg.export.qai_hub
    device  = hub.Device(hub_cfg.device)

    for name, compiled_model in compiled_models.items():
        if compiled_model is None:
            continue
        logger.info(f"[AI Hub] {name} 인코더 프로파일링 제출...")
        profile_job = hub.submit_profile_job(
            model=compiled_model,
            device=device,
            name=f"{name}_encoder_profile",
        )
        logger.info(f"[AI Hub] {name} 인코더 프로파일 Job ID: {profile_job.job_id}")
        profile_job.wait()
        profile = profile_job.download_profile()
        # 추론 시간 (밀리초)
        inference_ms = profile.get("execution_summary", {}).get("inference_time_ms", "N/A")
        logger.info(f"[AI Hub] {name} 인코더 추론 시간: {inference_ms} ms")


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LPCVC 2026 Track 1 — ONNX Export & AI Hub 제출")
    parser.add_argument("--checkpoint", type=str, required=True, help="학습된 체크포인트 경로")
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--onnx_only",  action="store_true", help="ONNX 내보내기만 수행 (AI Hub 제출 생략)")
    parser.add_argument("--quantize",   action="store_true", help="AI Hub INT8 양자화 수행")
    parser.add_argument("--profile",    action="store_true", help="컴파일 후 AI Hub 프로파일링 수행")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    output_dir = Path(cfg.export.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 모델 로드
    logger.info(f"체크포인트 로드: {args.checkpoint}")
    model = build_student_model(cfg).cpu()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # 2. ONNX 내보내기
    onnx_paths = export_to_onnx(model, cfg, output_dir)

    # 3. ONNX 검증
    verify_onnx(model, onnx_paths, cfg)

    # 4. 파일 크기 출력
    print("\n" + "="*60)
    print("  ONNX 내보내기 완료")
    print("="*60)
    for name, path in onnx_paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {name:<8} : {path}  ({size_mb:.1f} MB)")
    print("="*60)

    if args.onnx_only:
        print("\n--onnx_only 플래그 설정 — AI Hub 제출 생략.")
        return

    # 5. (선택) INT8 양자화 먼저 수행
    compile_paths = onnx_paths
    if args.quantize:
        logger.info("INT8 양자화 수행 중...")
        compile_paths = quantize_on_aihub(onnx_paths, cfg)

    # 6. AI Hub 컴파일 제출
    logger.info("Qualcomm AI Hub 컴파일 제출 중...")
    compile_jobs = compile_on_aihub(compile_paths, cfg)

    # 7. 컴파일 완료 대기
    compiled_models = wait_for_compile(compile_jobs)

    # 8. (선택) 프로파일링
    if args.profile and compiled_models:
        profile_on_aihub(compiled_models, cfg)

    # 9. 제출 정보 출력
    print("\n" + "="*60)
    print("  Qualcomm AI Hub 컴파일 완료")
    print("="*60)
    for name, job in compile_jobs.items():
        print(f"  {name:<8} encoder Job ID: {job.job_id}")
    print("="*60)
    print("\nLPCVC 2026 제출 절차:")
    print("  1. 위 Job ID 를 LPCVC 제출 포털에 입력하세요.")
    print("  2. Stage 1 지연시간 기준: 이미지+텍스트 합산 35ms 미만")
    print("  3. Stage 2 품질 기준: Recall@10")
    print()


if __name__ == "__main__":
    main()
