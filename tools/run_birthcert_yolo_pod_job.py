"""Run YOLO training + OCR evaluation for birth certificates on a GPU pod.

This script assumes it is executed from the repo root on a RunPod GPU instance
with the dataset available at data/birthcert_yolo.

Example:
    python tools/run_birthcert_yolo_pod_job.py --epochs 100 --imgsz 960 --batch 8

For stronger OCR/VLM models:
    python tools/run_birthcert_yolo_pod_job.py --ocr-backend hf-vlm --ocr-model MODEL_ID
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("[pod-job]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO and evaluate YOLO+OCR birth-cert extraction.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="field_detector")
    parser.add_argument("--ocr-backend", choices=["easyocr", "hf-vlm"], default="easyocr")
    parser.add_argument("--ocr-model", default=None)
    parser.add_argument("--images", default="data/birthcert_yolo/images/val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    weights = ROOT / "outputs" / "birthcert_yolo" / args.name / "weights" / "best.pt"

    if not args.skip_train:
        run([
            sys.executable,
            "tools/train_birthcert_yolo.py",
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--name", args.name,
        ])

    if not weights.exists():
        raise SystemExit(f"missing YOLO weights: {weights}")

    extract_cmd = [
        sys.executable,
        "-m",
        "birthcert.yolo_ocr",
        "--weights", str(weights),
        "--images", args.images,
        "--ocr-backend", args.ocr_backend,
        "--conf", str(args.conf),
    ]
    if args.ocr_backend == "hf-vlm":
        if not args.ocr_model:
            raise SystemExit("--ocr-model is required with --ocr-backend hf-vlm")
        extract_cmd.extend(["--ocr-model", args.ocr_model])
    run(extract_cmd)

    run([
        sys.executable,
        "-m",
        "birthcert.evaluate",
        "--pred",
        "outputs/birthcert_yolo_ocr/records",
        "--labels",
        "data/birth_cert_labels",
    ])


if __name__ == "__main__":
    main()
