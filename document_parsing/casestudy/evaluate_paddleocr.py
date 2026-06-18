"""Evaluate PaddleOCR accuracy on case-study images vs ground-truth labels."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .schema import SCALAR_FIELD_PATHS, get_path

_TASHKEEL = re.compile(r"[ؗ-ًؚ-ْٰـ]")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).translate(_AR_DIGITS)
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = (
        text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ى", "ي").replace("ة", "ه")
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def _ocr_texts(image_path: Path, ocr) -> list[str]:
    texts = []
    for res in ocr.predict(str(image_path)):
        texts.extend(res["rec_texts"])
    return texts


def _best_match(target_n: str, texts: list[str]) -> float:
    best = 0.0
    for t in texts:
        s = SequenceMatcher(None, target_n, _norm(t)).ratio()
        if s > best:
            best = s
    for i in range(len(texts) - 1):
        s = SequenceMatcher(None, target_n, _norm(texts[i] + " " + texts[i+1])).ratio()
        if s > best:
            best = s
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="document_parsing/data/raw_images/DataSet/cast study")
    parser.add_argument("--labels", default="document_parsing/data/case_study_labels")
    args = parser.parse_args()

    from paddleocr import PaddleOCR
    print("[eval_paddle] Loading PaddleOCR (lang=ar) …", flush=True)
    ocr = PaddleOCR(
        lang="ar",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    tp = fp = fn = tn = 0
    per_field_correct: dict[str, int] = {p: 0 for p in SCALAR_FIELD_PATHS}
    per_field_present: dict[str, int] = {p: 0 for p in SCALAR_FIELD_PATHS}
    scored = []

    for label_file in sorted(labels_dir.glob("*.json")):
        doc_id = label_file.stem
        image_path = None
        for ext in (".jpeg", ".jpg", ".png"):
            c = images_dir / f"{doc_id}{ext}"
            if c.exists():
                image_path = c
                break
        if image_path is None:
            continue

        label = json.loads(label_file.read_text(encoding="utf-8"))
        print(f"[eval_paddle] {doc_id} …", end=" ", flush=True)

        texts = _ocr_texts(image_path, ocr)
        doc_tp = doc_fp = doc_fn = doc_tn = 0

        for field_path in SCALAR_FIELD_PATHS:
            gt_val = get_path(label, field_path)
            gt_n = _norm(gt_val)
            if not gt_n:
                tn += 1; doc_tn += 1
                continue
            per_field_present[field_path] += 1
            score = _best_match(gt_n, texts)
            if score >= 0.85:
                tp += 1; doc_tp += 1
                per_field_correct[field_path] += 1
            elif score >= 0.50:
                fp += 1; doc_fp += 1
            else:
                fn += 1; doc_fn += 1

        scored.append(doc_id)
        print(f"tp={doc_tp} fp={doc_fp} fn={doc_fn} tn={doc_tn}")

    produced = tp + fp
    present  = tp + fn
    precision = tp / produced * 100 if produced else 0
    recall    = tp / present  * 100 if present  else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\nDocuments scored: {len(scored)}")
    print("=" * 60)
    print(f"Precision (correct / produced): {precision:.1f}%   [{tp} / {produced}]")
    print(f"Recall    (correct / present):  {recall:.1f}%   [{tp} / {present}]")
    print(f"F1:                              {f1:.1f}%")
    print(f"True negatives (both blank):     {tn}")
    print("=" * 60)
    print("Per-field accuracy:")
    for field_path in SCALAR_FIELD_PATHS:
        n = per_field_present[field_path]
        if n == 0:
            continue
        c = per_field_correct[field_path]
        print(f"  {field_path:<52} {c}/{n}  ({100*c//n}%)")


if __name__ == "__main__":
    main()
