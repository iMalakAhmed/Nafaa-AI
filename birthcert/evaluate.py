"""Score model predictions against hand-filled ground-truth labels.

Field-level metrics, computed only over the scalar fields in the schema:
  - precision: of the values the model produced, how many matched the label
  - recall:    of the values that exist in the label, how many the model produced
  - exact-match accuracy per field, plus a confusion of the common error types

Comparison is forgiving about formatting (Arabic digit normalization, whitespace,
diacritics) but strict about the actual content.

Usage (from repo root):
  python -m birthcert.evaluate
  python -m birthcert.evaluate --pred outputs/birthcert --labels data/birth_cert_labels
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .schema import SCALAR_FIELD_PATHS, get_path
from .validate import normalize_digits

_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = normalize_digits(text)
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip().strip(".،,").strip()
    return text.lower() or None


def _load(folder: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(folder.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] could not parse {f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate birth-certificate predictions vs labels.")
    parser.add_argument("--pred", type=str, default="outputs/birthcert")
    parser.add_argument("--labels", type=str, default="data/birth_cert_labels")
    parser.add_argument("--verbose", action="store_true", help="Print every mismatch.")
    args = parser.parse_args()

    preds = _load(Path(args.pred))
    labels = _load(Path(args.labels))
    common = sorted(set(preds) & set(labels))

    if not common:
        print("No overlapping documents between predictions and labels.")
        print(f"  predictions: {len(preds)} in {args.pred}")
        print(f"  labels:      {len(labels)} in {args.labels}")
        return

    tp = fp = fn = tn = 0
    per_field: dict[str, list[int]] = {p: [0, 0] for p in SCALAR_FIELD_PATHS}  # [correct, total_with_label]
    mismatches: list[str] = []

    for doc_id in common:
        pred, label = preds[doc_id], labels[doc_id]
        for path in SCALAR_FIELD_PATHS:
            p = _norm(get_path(pred, path))
            g = _norm(get_path(label, path))
            if g is not None:
                per_field[path][1] += 1
            if g is None and p is None:
                tn += 1
            elif g is not None and p is not None and p == g:
                tp += 1
                per_field[path][0] += 1
            elif g is not None and p is not None and p != g:
                fp += 1  # produced wrong value (a hallucination or misread)
                mismatches.append(f"{doc_id} {path}: pred={p!r} label={g!r}")
            elif g is None and p is not None:
                fp += 1  # produced a value where the label is blank (hallucination)
                mismatches.append(f"{doc_id} {path}: pred={p!r} label=∅ (extra)")
            else:  # g is not None and p is None
                fn += 1  # missed a value that exists
                mismatches.append(f"{doc_id} {path}: pred=∅ label={g!r} (missed)")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\nDocuments scored: {len(common)}  ({', '.join(common)})")
    print("=" * 60)
    print(f"Precision (correct / produced): {precision:.1%}   [{tp} / {tp + fp}]")
    print(f"Recall    (correct / present):  {recall:.1%}   [{tp} / {tp + fn}]")
    print(f"F1:                              {f1:.1%}")
    print(f"True negatives (both blank):     {tn}")
    print("=" * 60)
    print("Per-field accuracy (correct / present-in-label):")
    for path in SCALAR_FIELD_PATHS:
        correct, total = per_field[path]
        if total == 0:
            continue
        print(f"  {path:<48} {correct}/{total}  ({correct / total:.0%})")

    if args.verbose and mismatches:
        print("\nMismatches:")
        for m in mismatches:
            print("  -", m)
    elif mismatches:
        print(f"\n{len(mismatches)} field mismatch(es). Re-run with --verbose to list them.")


if __name__ == "__main__":
    main()
