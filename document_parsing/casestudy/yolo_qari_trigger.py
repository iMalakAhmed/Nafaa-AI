"""Run QARI full-page extraction on case-study images + auto CER/WER eval.

Usage:
    python -m document_parsing.casestudy.yolo_qari_trigger
    python -m document_parsing.casestudy.yolo_qari_trigger --images document_parsing/data/raw_images/DataSet/cast study --limit 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

QARI_MODEL     = "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct"
DEFAULT_IMAGES = "document_parsing/data/raw_images/DataSet/cast study"
OUT_RECORDS    = "document_parsing/outputs/casestudy/qari/records"
OUT_RAW        = "document_parsing/outputs/casestudy/qari/raw"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--model", default=QARI_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fields", default=None, help="Comma-separated field paths (default: all).")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists():
        sys.exit(f"Images directory not found: {images_dir}")

    # If limit requested, collect image paths and pass a temp list via --images pointing to a subset
    image_paths = sorted(
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.png"))
    )
    if not image_paths:
        sys.exit(f"No images found in {images_dir}")

    if args.limit:
        image_paths = image_paths[: args.limit]
        print(f"[trigger] Limiting to {len(image_paths)} image(s).")

    cmd = [
        sys.executable, "-m", "document_parsing.casestudy.yolo_ocr",
        "--images", str(images_dir),
        "--model", args.model,
        "--out", OUT_RECORDS,
        "--raw", OUT_RAW,
    ]
    if args.fields:
        cmd += ["--fields", args.fields]

    print(f"[trigger] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

    labels_dir = Path("document_parsing/data/case_study_labels")
    if labels_dir.exists():
        print("\n[trigger] Running CER/WER evaluation …")
        eval_cmd = [
            sys.executable, "-m", "document_parsing.casestudy.evaluate_qari_text",
            "--reads", OUT_RAW,
            "--labels", str(labels_dir),
        ]
        subprocess.run(eval_cmd)


if __name__ == "__main__":
    main()
