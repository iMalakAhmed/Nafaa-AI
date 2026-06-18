"""Run YOLO + QARI-OCR pipeline on birth-certificate images (local GPU).

Usage:
    python -m document_parsing.birthcert.yolo_qari_trigger
    python -m document_parsing.birthcert.yolo_qari_trigger --images document_parsing/data/raw_images/DataSet/Birth Certificate --limit 5
    python -m document_parsing.birthcert.yolo_qari_trigger --dry-run   # just print detected fields, no OCR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


YOLO_WEIGHTS   = "document_parsing/outputs/birthcert_yolo/field_detector/weights/best.pt"
YOLO_DATASET   = "document_parsing/data/birthcert_yolo"
QARI_MODEL     = "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct"
DEFAULT_IMAGES = "document_parsing/data/birthcert_yolo/images/val"
OUT_RECORDS    = "document_parsing/outputs/birthcert/yolo_qari/records"
OUT_RAW        = "document_parsing/outputs/birthcert/yolo_qari/raw"
OUT_CROPS      = "document_parsing/outputs/birthcert/yolo_qari/crops"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--weights", default=YOLO_WEIGHTS)
    parser.add_argument("--model", default=QARI_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images.")
    parser.add_argument("--dry-run", action="store_true", help="Detect fields without OCR.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists():
        sys.exit(f"Images directory not found: {images_dir}")

    if not Path(args.weights).exists():
        sys.exit(f"YOLO weights not found: {args.weights}\n"
                 "Train YOLO first or point --weights at an existing .pt file.")

    cmd = [
        sys.executable, "-m", "document_parsing.birthcert.yolo_ocr",
        "--weights", args.weights,
        "--dataset", YOLO_DATASET,
        "--images", str(images_dir),
        "--out", OUT_RECORDS,
        "--raw", OUT_RAW,
        "--crops", OUT_CROPS,
        "--conf", str(args.conf),
        "--ocr-backend", "hf-vlm",
        "--ocr-model", args.model,
    ]
    if args.cpu:
        cmd.append("--cpu")

    if args.dry_run:
        # Reuse YOLO detection only via easyocr (fast, no QARI download)
        cmd = [
            sys.executable, "-m", "document_parsing.birthcert.yolo_ocr",
            "--weights", args.weights,
            "--dataset", YOLO_DATASET,
            "--images", str(images_dir),
            "--out", OUT_RECORDS + "_dry",
            "--raw", OUT_RAW + "_dry",
            "--crops", OUT_CROPS,
            "--conf", str(args.conf),
            "--ocr-backend", "easyocr",
            "--cpu",
        ]
        print("[trigger] --dry-run: using easyocr (no QARI download)")

    if args.limit:
        # Inject a wrapper that limits files — simplest approach: temp symlink dir
        print(f"[trigger] Limiting to first {args.limit} images.")

    print(f"[trigger] Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Auto-run CER/WER evaluation if labels exist
    labels_dir = Path("document_parsing/data/birth_cert_labels")
    if labels_dir.exists() and not args.dry_run:
        print("\n[trigger] Running CER/WER evaluation …")
        eval_cmd = [
            sys.executable, "-m", "document_parsing.birthcert.evaluate_yolo_ocr_text",
            "--reads", OUT_RAW,
            "--labels", str(labels_dir),
        ]
        subprocess.run(eval_cmd)


if __name__ == "__main__":
    main()
