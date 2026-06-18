"""Score case-study predictions against hand-filled labels."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .schema import SCALAR_FIELD_PATHS, get_path
from .validate import normalize_digits

_TASHKEEL = re.compile(r"[\u0617-\u061a\u064b-\u0652\u0670\u0640]")


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = normalize_digits(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = (
        text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        .replace("ى", "ي").replace("ة", "ه")
    )
    text = re.sub(r"\s+", " ", text).strip().strip(".,،")
    return text.lower() or None


def _load(folder: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(folder.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] could not parse {f}")
    return out


def _family_score(pred: dict[str, Any], label: dict[str, Any]) -> tuple[int, int]:
    p_rows = pred.get("family_members") if isinstance(pred.get("family_members"), list) else []
    g_rows = label.get("family_members") if isinstance(label.get("family_members"), list) else []
    correct = total = 0
    for i, g in enumerate(g_rows):
        if not isinstance(g, dict):
            continue
        p = p_rows[i] if i < len(p_rows) and isinstance(p_rows[i], dict) else {}
        for key in ("name", "relationship", "age", "national_id", "marital_status", "education_status", "employment_status", "health_status"):
            gv = _norm(g.get(key))
            if gv is None:
                continue
            total += 1
            if _norm(p.get(key)) == gv:
                correct += 1
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate case-study predictions vs labels.")
    parser.add_argument("--pred", type=str, default="outputs/casestudy/records")
    parser.add_argument("--labels", type=str, default="data/case_study_labels")
    parser.add_argument("--verbose", action="store_true")
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
    mismatches: list[str] = []
    per_field: dict[str, list[int]] = {p: [0, 0] for p in SCALAR_FIELD_PATHS}
    fam_correct = fam_total = 0
    for doc_id in common:
        pred, label = preds[doc_id], labels[doc_id]
        for path in SCALAR_FIELD_PATHS:
            p = _norm(get_path(pred, path))
            g = _norm(get_path(label, path))
            if g is not None:
                per_field[path][1] += 1
            if g is None and p is None:
                tn += 1
            elif g is not None and p == g:
                tp += 1
                per_field[path][0] += 1
            elif g is not None and p is not None:
                fp += 1
                mismatches.append(f"{doc_id} {path}: pred={p!r} label={g!r}")
            elif g is None and p is not None:
                fp += 1
                mismatches.append(f"{doc_id} {path}: pred={p!r} label=null")
            else:
                fn += 1
                mismatches.append(f"{doc_id} {path}: pred=null label={g!r}")
        c, t = _family_score(pred, label)
        fam_correct += c
        fam_total += t

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\nDocuments scored: {len(common)}  ({', '.join(common)})")
    print("=" * 60)
    print(f"Precision (correct / produced): {precision:.1%}   [{tp} / {tp + fp}]")
    print(f"Recall    (correct / present):  {recall:.1%}   [{tp} / {tp + fn}]")
    print(f"F1:                              {f1:.1%}")
    print(f"True negatives (both blank):     {tn}")
    if fam_total:
        print(f"Family table accuracy:           {fam_correct}/{fam_total} ({fam_correct / fam_total:.1%})")
    print("=" * 60)
    for path in SCALAR_FIELD_PATHS:
        correct, total = per_field[path]
        if total:
            print(f"  {path:<48} {correct}/{total} ({correct / total:.0%})")
    if args.verbose and mismatches:
        print("\nMismatches:")
        for item in mismatches:
            print("  -", item)
    elif mismatches:
        print(f"\n{len(mismatches)} scalar field mismatch(es). Re-run with --verbose to list them.")


if __name__ == "__main__":
    main()

