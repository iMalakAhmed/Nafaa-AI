"""Generate YOLO bounding-box annotations for case-study form images.

Uses EasyOCR to detect all Arabic/English text regions, then fuzzy-matches
every ground-truth scalar field value to the nearest OCR hit to get its
bounding box. family_members rows and checkbox_answers are skipped (they
span table cells and need a separate annotation pass).

Install extra deps:
    pip install easyocr

Usage:
    python -m document_parsing.casestudy.annotate
    python -m document_parsing.casestudy.annotate --images "document_parsing/data/raw_images/DataSet/cast study" \\
                                  --labels document_parsing/data/case_study_labels \\
                                  --out    document_parsing/data/yolo/casestudy \\
                                  --conf   0.50
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .schema import SCALAR_FIELD_PATHS, get_path

CLASS_NAMES: list[str] = SCALAR_FIELD_PATHS  # 42 scalar classes

_TASHKEEL = re.compile(r"[ؗ-ًؚ-ْٰـ]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).translate(_AR_DIGITS)
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = (
        text.replace("أ", "ا")
            .replace("إ", "ا")
            .replace("آ", "ا")
            .replace("ى", "ي")
            .replace("ة", "ه")
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def _poly_to_yolo(poly: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return (
        (x1 + x2) / 2 / img_w,
        (y1 + y2) / 2 / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    )


def _merge_polys(p1: list, p2: list) -> list:
    all_pts = list(p1) + list(p2)
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    return [[min(xs), min(ys)], [max(xs), min(ys)],
            [max(xs), max(ys)], [min(xs), max(ys)]]


def _best_match(target: str, ocr: list) -> tuple[float, list | None]:
    target_n = _norm(target)
    if not target_n:
        return 0.0, None

    best_score, best_poly = 0.0, None

    for poly, text, _ in ocr:
        score = SequenceMatcher(None, target_n, _norm(text)).ratio()
        if score > best_score:
            best_score, best_poly = score, poly

    for i in range(len(ocr) - 1):
        p1, t1, _ = ocr[i]
        p2, t2, _ = ocr[i + 1]
        score = SequenceMatcher(None, target_n, _norm(t1 + " " + t2)).ratio()
        if score > best_score:
            best_score, best_poly = score, _merge_polys(p1, p2)

    return best_score, best_poly


def annotate_image(
    image_path: Path,
    label: dict,
    reader,
    conf_threshold: float,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    from PIL import Image as PILImage

    img_w, img_h = PILImage.open(image_path).size
    ocr = reader.readtext(str(image_path), detail=1, paragraph=False)

    yolo_lines: list[str] = []
    low_conf: list[tuple[str, str, str]] = []

    for class_id, field_path in enumerate(CLASS_NAMES):
        value = get_path(label, field_path)
        if value is None:
            continue

        score, poly = _best_match(str(value), ocr)

        if score >= conf_threshold and poly is not None:
            x_c, y_c, w, h = _poly_to_yolo(poly, img_w, img_h)
            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        else:
            low_conf.append((field_path, str(value)[:50], f"{score:.2f}"))

    return yolo_lines, low_conf


def _write_data_yaml(out_dir: Path) -> None:
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images",
        "val: images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for name in CLASS_NAMES:
        lines.append(f"  - {name}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate case-study images for YOLO.")
    parser.add_argument("--images", default="document_parsing/data/raw_images/DataSet/cast study")
    parser.add_argument("--labels", default="document_parsing/data/case_study_labels")
    parser.add_argument("--out",    default="document_parsing/data/yolo/casestudy")
    parser.add_argument("--conf",   type=float, default=0.50)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_dir    = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    import easyocr
    print("[annotate] Loading EasyOCR (ar + en) …", flush=True)
    reader = easyocr.Reader(["ar", "en"], gpu=not args.no_gpu)

    label_files = sorted(labels_dir.glob("*.json"))
    all_low_conf: list[str] = []
    n_images = 0

    for label_file in label_files:
        doc_id = label_file.stem

        image_path: Path | None = None
        for ext in (".jpeg", ".jpg", ".png"):
            c = images_dir / f"{doc_id}{ext}"
            if c.exists():
                image_path = c
                break

        if image_path is None:
            print(f"[annotate] {doc_id}: image not found — skipped")
            continue

        label = json.loads(label_file.read_text(encoding="utf-8"))
        print(f"[annotate] {doc_id} …", end=" ", flush=True)

        yolo_lines, low_conf = annotate_image(image_path, label, reader, args.conf)

        (out_dir / "labels" / f"{doc_id}.txt").write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )
        shutil.copy2(image_path, out_dir / "images" / image_path.name)
        n_images += 1

        print(f"{len(yolo_lines)} boxes  |  {len(low_conf)} low-conf")
        for field, val, score in low_conf:
            all_low_conf.append(f"{doc_id}\t{field}\t{score}\t{val}")

    _write_data_yaml(out_dir)

    if all_low_conf:
        report = out_dir / "low_confidence.tsv"
        header = "doc_id\tfield\tscore\tvalue\n"
        report.write_text(header + "\n".join(all_low_conf), encoding="utf-8")
        print(f"\n[annotate] {len(all_low_conf)} low-conf fields saved -> {report}")

    print(f"\n[annotate] Done — {n_images} images annotated")
    print(f"  {out_dir / 'data.yaml'}")
    print(f"  {out_dir / 'images'}  ({n_images} files)")
    print(f"  {out_dir / 'labels'}  ({n_images} files)")
    print(f"\nNext: review low_confidence.tsv in Label Studio, then train with:")
    print(f"  yolo train data={out_dir / 'data.yaml'} model=yolo11n.pt epochs=100 imgsz=1280")


if __name__ == "__main__":
    main()
