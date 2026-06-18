"""Create blank ground-truth label templates for case-study pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import empty_record

IMAGES_DIR = Path("data/raw_images/DataSet/cast study")
LABELS_DIR = Path("data/case_study_labels")


def _list_images() -> list[Path]:
    return sorted(IMAGES_DIR.glob("*.jpeg")) + sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blank case-study label templates.")
    parser.add_argument("--count", type=int, default=0, help="0 means all images")
    parser.add_argument("--ids", type=str, default="")
    parser.add_argument("--out", type=str, default=str(LABELS_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--from-pred", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = _list_images()
    if args.ids.strip():
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        images = [p for p in images if p.stem in wanted]
    elif args.count > 0:
        images = images[: args.count]

    pred_dir = Path(args.from_pred) if args.from_pred.strip() else None
    created = seeded = 0
    for img in images:
        target = out_dir / f"{img.stem}.json"
        if target.exists() and not args.overwrite:
            print(f"skip (exists): {target.name}")
            continue
        source = str(img).replace("\\", "/")
        record = empty_record(img.stem, [source])
        if pred_dir:
            pred_file = pred_dir / f"{img.stem}.json"
            if pred_file.exists():
                try:
                    record = json.loads(pred_file.read_text(encoding="utf-8"))
                    record["document_id"] = img.stem
                    record["source_files"] = [source]
                    record["review_required"] = True
                    notes = record.get("review_notes")
                    if not isinstance(notes, list):
                        notes = []
                    notes.append("bootstrap prediction: human review required before fine-tuning")
                    record["review_notes"] = notes
                    seeded += 1
                except json.JSONDecodeError:
                    pass
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        created += 1
        print(f"created: {target.name}" + (" (pre-filled)" if pred_dir else ""))
    print(f"\n{created} template(s) in {out_dir}/ ({seeded} pre-filled).")


if __name__ == "__main__":
    main()
