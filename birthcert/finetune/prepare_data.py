"""Turn hand-labeled certificates into a supervised fine-tuning dataset.

Reads the label JSONs you filled in (data/birth_cert_labels/*.json) plus their
images, and writes train/val JSONL where each row is:
  { "image": "<path>", "instruction": "<the layout prompt>", "target": "<gold JSON>" }

The instruction is the SAME prompt used at inference, so the model learns to map
this exact layout to the exact JSON we want. Only labels that actually contain
values are used (an all-null template teaches nothing).

Usage (from repo root):
  python -m birthcert.finetune.prepare_data
  python -m birthcert.finetune.prepare_data --labels data/birth_cert_labels --val-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from ..prompt import SYSTEM_PROMPT, build_user_instruction
from ..schema import SCALAR_FIELD_PATHS, empty_record, get_path, set_path
from .reasoning import build_target_with_reasoning

IMAGES_DIR = Path("data/raw_images/DataSet/Birth Certificate")
LABELS_DIR = Path("data/birth_cert_labels")
OUT_DIR = Path("data/birth_cert_ft")


def _filled_count(record: dict[str, Any]) -> int:
    return sum(1 for p in SCALAR_FIELD_PATHS if get_path(record, p) not in (None, ""))


def _canonical_target(label: dict[str, Any], doc_id: str, source: str) -> dict[str, Any]:
    """Rebuild a clean, full-schema record from the label so targets are consistent."""
    record = empty_record(document_id=doc_id, source_files=[source])
    for path in SCALAR_FIELD_PATHS:
        value = get_path(label, path)
        if isinstance(value, str):
            value = value.strip() or None
        if value is not None:
            set_path(record, path, value)
    # Carry over a few non-scalar fields if present.
    other = get_path(label, "ids.other_ids")
    if isinstance(other, list):
        record["ids"]["other_ids"] = other
    for key in ("uncertain_fields", "review_notes"):
        val = label.get(key)
        if isinstance(val, list):
            record[key] = val
    record["review_required"] = bool(label.get("review_required", False))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset from labeled certificates.")
    parser.add_argument("--labels", type=str, default=str(LABELS_DIR))
    parser.add_argument("--images", type=str, default=str(IMAGES_DIR))
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels_dir = Path(args.labels)
    images_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        print(f"No label files in {labels_dir}. Run: python -m birthcert.make_labels --count 10")
        return

    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    skipped_noimage = 0
    for lf in label_files:
        doc_id = lf.stem
        try:
            label = json.loads(lf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[warn] bad JSON, skipping: {lf.name}")
            continue
        if _filled_count(label) == 0:
            skipped_empty += 1
            continue
        # Find the image (jpeg/jpg/png).
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
        rows.append(
            {
                "document_id": doc_id,
                "image": source,
                "system": SYSTEM_PROMPT,
                "instruction": build_user_instruction(doc_id, source),
                "target": build_target_with_reasoning(record_json, target),
            }
        )

    if not rows:
        print(
            "No usable labeled examples found.\n"
            f"  empty templates skipped: {skipped_empty}\n"
            "Fill in the values in data/birth_cert_labels/*.json (leave truly blank fields null),\n"
            "then re-run this. Aim for at least ~20-30 filled certificates for a useful fine-tune."
        )
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
    if skipped_empty:
        print(f"  (skipped {skipped_empty} still-blank label template(s))")
    if skipped_noimage:
        print(f"  (skipped {skipped_noimage} label(s) with no matching image)")
    if len(rows) < 15:
        print(
            f"\nNote: only {len(rows)} labeled example(s). Fine-tuning works better with more —\n"
            "consider labeling 20-40 certificates for a meaningful accuracy gain."
        )


if __name__ == "__main__":
    main()
