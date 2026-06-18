"""Compute CER/WER for YOLO+OCR field reads against birth-certificate labels."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .schema import get_path
from .validate import normalize_digits


FIELD_TO_LABEL_PATH = {
    "child_national_id": "ids.child_national_id",
    "child_name": "personal_and_other.child.name",
    "date_of_birth": "personal_and_other.child.date_of_birth",
    "place_of_birth": "personal_and_other.child.place_of_birth",
    "father_name": "personal_and_other.father.name",
    "mother_name": "personal_and_other.mother.name",
    "registration_number": "birth_certificate.registration_number",
    "registration_date": "birth_certificate.registration_date",
    "issue_date": "birth_certificate.issue_date",
    "serial_number": "birth_certificate.serial_number",
}

_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = normalize_digits(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip(" .,:;،")
    return text


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR text CER/WER for YOLO field reads.")
    parser.add_argument("--reads", default="outputs/birthcert_yolo_ocr/field_reads")
    parser.add_argument("--labels", default="data/birth_cert_labels")
    args = parser.parse_args()

    reads_dir = Path(args.reads)
    labels_dir = Path(args.labels)
    label_files = {p.stem: p for p in labels_dir.glob("*.json")}

    char_err = char_total = 0
    word_err = word_total = 0
    scored = 0
    per_field: dict[str, list[int]] = {field: [0, 0, 0, 0] for field in FIELD_TO_LABEL_PATH}

    for read_path in sorted(reads_dir.glob("*.json")):
        label_path = label_files.get(read_path.stem)
        if not label_path:
            continue
        label = load_json(label_path)
        reads = load_json(read_path)
        by_field = {item.get("field"): item for item in reads if isinstance(item, dict)}
        for field, label_field_path in FIELD_TO_LABEL_PATH.items():
            gold = norm_text(get_path(label, label_field_path))
            if not gold:
                continue
            pred = norm_text((by_field.get(field) or {}).get("value"))

            c_err = edit_distance(list(pred), list(gold))
            g_chars = len(gold)
            p_words = pred.split()
            g_words = gold.split()
            w_err = edit_distance(p_words, g_words)

            char_err += c_err
            char_total += g_chars
            word_err += w_err
            word_total += len(g_words)
            scored += 1

            stats = per_field[field]
            stats[0] += c_err
            stats[1] += g_chars
            stats[2] += w_err
            stats[3] += len(g_words)

    cer = char_err / char_total if char_total else 0.0
    wer = word_err / word_total if word_total else 0.0
    print(f"Fields scored: {scored}")
    print(f"CER: {cer:.1%}   [{char_err} / {char_total}]")
    print(f"WER: {wer:.1%}   [{word_err} / {word_total}]")
    print("Per-field CER/WER:")
    for field, (ce, ct, we, wt) in per_field.items():
        if ct == 0 and wt == 0:
            continue
        f_cer = ce / ct if ct else 0.0
        f_wer = we / wt if wt else 0.0
        print(f"  {field:<22} CER {f_cer:>7.1%}   WER {f_wer:>7.1%}")


if __name__ == "__main__":
    main()
