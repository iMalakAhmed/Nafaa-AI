"""Compute field-level F1 + CER/WER for QARI full-page extraction vs ground-truth labels."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .schema import SCALAR_FIELD_PATHS, get_path
from .validate import normalize_digits

_TASHKEEL = re.compile(r"[ؗ-ًؚ-ْٰـ]")


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = normalize_digits(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", text).strip(" .,:;،").lower()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reads", default="document_parsing/outputs/casestudy/qari/raw",
                        help="Directory of per-image raw JSON files from document_parsing/casestudy/yolo_ocr.py")
    parser.add_argument("--labels", default="document_parsing/data/case_study_labels")
    args = parser.parse_args()

    reads_dir = Path(args.reads)
    labels_dir = Path(args.labels)
    label_files = {p.stem: p for p in labels_dir.glob("*.json")}

    char_err = char_total = 0
    word_err = word_total = 0
    tp = fp = fn = tn = 0
    per_field: dict[str, list[int]] = {f: [0, 0, 0, 0, 0, 0] for f in SCALAR_FIELD_PATHS}
    # per_field values: [char_err, char_total, word_err, word_total, correct, present]
    scored = 0

    for read_path in sorted(reads_dir.glob("*.json")):
        label_path = label_files.get(read_path.stem)
        if not label_path:
            continue
        label = json.loads(label_path.read_text(encoding="utf-8"))
        reads = json.loads(read_path.read_text(encoding="utf-8"))
        by_field = {item["field"]: item for item in reads if isinstance(item, dict)}

        for field_path in SCALAR_FIELD_PATHS:
            gold = norm_text(get_path(label, field_path))
            if not gold:
                tn += 1
                continue
            per_field[field_path][5] += 1

            pred = norm_text((by_field.get(field_path) or {}).get("value"))

            c_err = edit_distance(list(pred), list(gold))
            g_chars = len(gold)
            g_words = gold.split()
            p_words = pred.split()
            w_err = edit_distance(p_words, g_words)

            char_err += c_err
            char_total += g_chars
            word_err += w_err
            word_total += len(g_words)
            scored += 1

            pf = per_field[field_path]
            pf[0] += c_err
            pf[1] += g_chars
            pf[2] += w_err
            pf[3] += len(g_words)

            # field-level F1: exact match after normalization
            if pred == gold:
                tp += 1
                pf[4] += 1
            elif pred:
                fp += 1
            else:
                fn += 1

    cer = char_err / char_total if char_total else 0.0
    wer = word_err / word_total if word_total else 0.0

    produced = tp + fp
    present  = tp + fn
    precision = tp / produced if produced else 0.0
    recall    = tp / present  if present  else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Fields scored  : {scored}")
    print(f"CER            : {cer:.1%}   [{char_err}/{char_total}]")
    print(f"WER            : {wer:.1%}   [{word_err}/{word_total}]")
    print(f"Exact-match F1 : {f1:.1%}   (P={precision:.1%}, R={recall:.1%})")
    print()
    print("Per-field breakdown:")
    for field_path in SCALAR_FIELD_PATHS:
        ce, ct, we, wt, correct, present_n = per_field[field_path]
        if ct == 0:
            continue
        f_cer = ce / ct
        f_wer = we / wt if wt else 0.0
        f_exact = correct / present_n if present_n else 0.0
        print(f"  {field_path:<45} CER {f_cer:>6.1%}  WER {f_wer:>6.1%}  Exact {f_exact:>6.1%}  [{correct}/{present_n}]")


if __name__ == "__main__":
    main()
