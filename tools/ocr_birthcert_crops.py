from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from birthcert.yolo_ocr import _repair_mojibake, make_ocr_backend, normalize_field_text


def field_from_crop(path: Path, document_id: str) -> str:
    prefix = f"{document_id}_"
    stem = path.stem
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR directly on saved birth certificate crop images.")
    parser.add_argument("--crops", required=True)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--ocr-backend", choices=["easyocr", "paddleocr", "hf-vlm"], default="paddleocr")
    parser.add_argument("--ocr-model", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    crop_dir = Path(args.crops)
    document_id = args.document_id or crop_dir.name
    crop_paths = sorted(crop_dir.glob("*.png"))
    if not crop_paths:
        raise SystemExit(f"no PNG crops found in {crop_dir}")

    ocr = make_ocr_backend(args.ocr_backend, model_name=args.ocr_model, gpu=not args.cpu)
    rows = []
    for crop_path in crop_paths:
        field_name = field_from_crop(crop_path, document_id)
        raw_text = _repair_mojibake(ocr.read(Image.open(crop_path), field_name=field_name))
        rows.append({
            "crop": str(crop_path).replace("\\", "/"),
            "field": field_name,
            "ocr_raw": raw_text,
            "value": normalize_field_text(field_name, raw_text),
        })

    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
