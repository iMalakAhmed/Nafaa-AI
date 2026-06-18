"""Create blank ground-truth label templates for a few certificates.

You fill these in by reading the images yourself; `evaluate.py` then scores model
predictions against them. This is the path to a real accuracy number — and, later,
to actual fine-tuning data.

Usage (from repo root):
  python -m birthcert.make_labels --count 10
  python -m birthcert.make_labels --ids BC_001,BC_002,BC_003
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import empty_record

IMAGES_DIR = Path("data/raw_images/DataSet/Birth Certificate")
LABELS_DIR = Path("data/birth_cert_labels")


def _list_images() -> list[Path]:
    return sorted(IMAGES_DIR.glob("*.jpeg")) + sorted(IMAGES_DIR.glob("*.jpg"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blank birth-certificate label templates.")
    parser.add_argument("--count", type=int, default=10, help="How many (from the start) to template.")
    parser.add_argument("--ids", type=str, default="", help="Comma-separated ids, e.g. BC_001,BC_002.")
    parser.add_argument("--out", type=str, default=str(LABELS_DIR))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing label files.")
    parser.add_argument(
        "--from-pred",
        type=str,
        default="",
        help="Pre-fill templates from a predictions folder (e.g. outputs/birthcert/records_ain). "
        "You then just CORRECT the wrong fields instead of typing everything.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _list_images()
    if args.ids.strip():
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        images = [p for p in images if p.stem in wanted]
    else:
        images = images[: args.count]

    if not images:
        print("No matching images found under", IMAGES_DIR)
        return

    pred_dir = Path(args.from_pred) if args.from_pred.strip() else None

    created = 0
    seeded = 0
    for img in images:
        target = out_dir / f"{img.stem}.json"
        if target.exists() and not args.overwrite:
            print(f"skip (exists): {target.name}")
            continue
        source = str(img).replace("\\", "/")
        record = empty_record(document_id=img.stem, source_files=[source])
        if pred_dir is not None:
            pred_file = pred_dir / f"{img.stem}.json"
            if pred_file.exists():
                try:
                    pred = json.loads(pred_file.read_text(encoding="utf-8"))
                    pred["document_id"] = img.stem
                    pred["source_files"] = [source]
                    record = pred
                    seeded += 1
                except json.JSONDecodeError:
                    pass
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        created += 1
        print(f"created: {target.name}" + (" (pre-filled)" if pred_dir else ""))

    if pred_dir is not None:
        print(
            f"\n{created} template(s) in {out_dir}/  ({seeded} pre-filled from predictions).\n"
            "Open each image side-by-side and FIX the wrong fields (the model often\n"
            "mislabels child vs parent names and misreads handwritten IDs). Set anything\n"
            "you cannot verify to null. Then run:\n"
            "  python -m birthcert.finetune.prepare_data"
        )
    else:
        print(
            f"\n{created} template(s) in {out_dir}/.\n"
            "Open each image, type the correct values into the JSON (leave null if blank),\n"
            "then run:  python -m birthcert.evaluate"
        )


if __name__ == "__main__":
    main()
