"""Turn hand-labeled case-study pages into a supervised fine-tuning dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ..prompt import SYSTEM_PROMPT, build_user_instruction
from ..schema import SCALAR_FIELD_PATHS, empty_record, get_path, set_path
from .reasoning import build_target_with_reasoning

IMAGES_DIR = Path("data/raw_images/DataSet/cast study")
LABELS_DIR = Path("data/case_study_labels")
OUT_DIR = Path("data/case_study_ft")


def _filled_count(record: dict[str, Any]) -> int:
    count = sum(1 for path in SCALAR_FIELD_PATHS if get_path(record, path) not in (None, ""))
    count += len(record.get("family_members") or [])
    count += len(record.get("checkbox_answers") or [])
    return count


def _canonical_target(label: dict[str, Any], doc_id: str, source: str) -> dict[str, Any]:
    record = empty_record(doc_id, [source])
    for path in SCALAR_FIELD_PATHS:
        value = get_path(label, path)
        if isinstance(value, str):
            value = value.strip() or None
        if value is not None:
            set_path(record, path, value)
    for key in ("family_members", "checkbox_answers", "uncertain_fields", "review_notes"):
        value = label.get(key)
        if isinstance(value, list):
            record[key] = value
    record["review_required"] = bool(label.get("review_required", False))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset from labeled case-study pages.")
    parser.add_argument("--labels", type=str, default=str(LABELS_DIR))
    parser.add_argument("--images", type=str, default=str(IMAGES_DIR))
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-review-required",
        action="store_true",
        help="Include labels still marked review_required. Default skips them to avoid training on bootstrap guesses.",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels)
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped_empty = skipped_noimage = skipped_review = 0

    for lf in sorted(labels_dir.glob("*.json")):
        doc_id = lf.stem
        try:
            label = json.loads(lf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] bad JSON, skipping: {lf.name}")
            continue
        if _filled_count(label) == 0:
            skipped_empty += 1
            continue
        if label.get("review_required") and not args.include_review_required:
            skipped_review += 1
            continue
        image = None
        for ext in (".jpeg", ".jpg", ".png"):
            cand = images_dir / f"{doc_id}{ext}"
            if cand.exists():
                image = cand
                break
        if image is None:
            skipped_noimage += 1
            print(f"[warn] no image for {doc_id}, skipping")
            continue
        source = str(image).replace("\\", "/")
        target = _canonical_target(label, doc_id, source)
        record_json = json.dumps(target, ensure_ascii=False)
        rows.append({
            "document_id": doc_id,
            "image": source,
            "system": SYSTEM_PROMPT,
            "instruction": build_user_instruction(doc_id, source),
            "target": build_target_with_reasoning(record_json, target),
        })

    if not rows:
        print("No usable labeled examples found.")
        print(f"  empty templates skipped: {skipped_empty}")
        print(f"  review-required skipped: {skipped_review}")
        print(f"  missing images skipped:  {skipped_noimage}")
        print("Correct data/case_study_labels/*.json, set review_required=false, then re-run.")
        return

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, round(len(rows) * args.val_frac)) if len(rows) >= 5 else 0
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    def _write(path: Path, data: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in data:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write(out_dir / "train.jsonl", train_rows)
    _write(out_dir / "val.jsonl", val_rows)
    print(f"Wrote {len(train_rows)} train + {len(val_rows)} val examples to {out_dir}/")
    if skipped_review:
        print(f"  (skipped {skipped_review} review-required label(s))")


if __name__ == "__main__":
    main()
