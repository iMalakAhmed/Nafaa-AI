"""Evaluate EasyOCR accuracy on birth-certificate images vs ground-truth labels.

Same normalization and scoring logic as evaluate.py so results are directly
comparable to the Qwen2.5-VL numbers.

Usage:
    python -m document_parsing.birthcert.evaluate_ocr
    python -m document_parsing.birthcert.evaluate_ocr --no-gpu
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .schema import SCALAR_FIELD_PATHS, get_path
from .validate import normalize_digits

_TASHKEEL = re.compile(r"[ؗ-ًؚ-ْٰـ]")


def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = normalize_digits(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = (
        text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
            .replace("ى", "ي").replace("ة", "ه")
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def _best_ocr_text(target_n: str, ocr: list) -> tuple[str, float]:
    """Return (best_matching_ocr_text, similarity_score)."""
    best_score, best_text = 0.0, ""
    for _, text, _ in ocr:
        score = SequenceMatcher(None, target_n, _norm(text)).ratio()
        if score > best_score:
            best_score, best_text = score, text
    # Try merged adjacent pairs
    for i in range(len(ocr) - 1):
        merged = ocr[i][1] + " " + ocr[i + 1][1]
        score = SequenceMatcher(None, target_n, _norm(merged)).ratio()
        if score > best_score:
            best_score, best_text = score, merged
    return best_text, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="document_parsing/data/raw_images/DataSet/Birth Certificate")
    parser.add_argument("--labels", default="document_parsing/data/birth_cert_labels")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import easyocr
    print("[eval_ocr] Loading EasyOCR …", flush=True)
    reader = easyocr.Reader(["ar", "en"], gpu=not args.no_gpu)

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)

    # Counters matching evaluate.py logic
    tp = 0   # model produced correct value
    fp = 0   # model produced wrong value
    fn = 0   # label has value, model produced nothing (score < 0.5)
    tn = 0   # both blank

    per_field_correct: dict[str, int] = {p: 0 for p in SCALAR_FIELD_PATHS}
    per_field_present: dict[str, int] = {p: 0 for p in SCALAR_FIELD_PATHS}

    label_files = sorted(labels_dir.glob("*.json"))
    scored = []

    for label_file in label_files:
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
        print(f"[eval_ocr] {doc_id} …", end=" ", flush=True)

        ocr = reader.readtext(str(image_path), detail=1, paragraph=False)
        doc_tp = doc_fp = doc_fn = doc_tn = 0

        for field_path in SCALAR_FIELD_PATHS:
            gt_val = get_path(label, field_path)
            gt_n = _norm(gt_val)

            if not gt_n:
                # Ground truth is blank
                tn += 1
                doc_tn += 1
                continue

            per_field_present[field_path] += 1
            _, score = _best_ocr_text(gt_n, ocr)

            # Use exact match (score == 1.0) and near-exact (score >= 0.85)
            # as "correct" — same threshold the Qwen evaluator uses (after
            # normalization, a score of 0.85+ typically means the value is right).
            if score >= 0.85:
                tp += 1
                doc_tp += 1
                per_field_correct[field_path] += 1
            elif score >= 0.50:
                fp += 1  # OCR found something but it's wrong / partial
                doc_fp += 1
                if args.verbose:
                    _, best_text = _best_ocr_text(gt_n, ocr)[0], _
                    print(f"\n  FP {field_path}: gt={gt_val!r} score={score:.2f}")
            else:
                fn += 1  # OCR missed the field entirely
                doc_fn += 1

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
    print("Per-field accuracy (correct / present-in-label):")
    for field_path in SCALAR_FIELD_PATHS:
        n = per_field_present[field_path]
        if n == 0:
            continue
        c = per_field_correct[field_path]
        print(f"  {field_path:<52} {c}/{n}  ({100*c//n}%)")


if __name__ == "__main__":
    main()
