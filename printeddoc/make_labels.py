"""Create blank ground-truth label templates for printed social-insurance documents.

Usage (from repo root):
  python -m printeddoc.make_labels --count 14
  python -m printeddoc.make_labels --ids PS_001,PS_002,PS_003
  python -m printeddoc.make_labels --from-pred outputs/printeddoc/records
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import empty_record

IMAGES_DIR = Path("data/raw_images/DataSet/printed_docuemnts")
LABELS_DIR = Path("data/printed_doc_labels")


def _list_images() -> list[Path]:
    return sorted(IMAGES_DIR.glob("*.jpeg")) + sorted(IMAGES_DIR.glob("*.jpg"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blank printed-document label templates.")
    parser.add_argument("--count", type=int, default=14)
    parser.add_argument("--ids", type=str, default="")
    parser.add_argument("--out", type=str, default=str(LABELS_DIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--from-pred", type=str, default="",
                        help="Pre-fill from a predictions folder so you only correct mistakes.")
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

    created = seeded = 0
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
                    pred["document_id"]   = img.stem
                    pred["source_files"]  = [source]
                    record = pred
                    seeded += 1
                except json.JSONDecodeError:
                    pass
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        created += 1
        print(f"created: {target.name}" + (" (pre-filled)" if pred_dir else ""))

    print(
        f"\n{created} template(s) in {out_dir}/  ({seeded} pre-filled).\n"
        "Open each image side-by-side, fill in the correct values (null if blank),\n"
        "then run:  python -m printeddoc.evaluate"
    )


if __name__ == "__main__":
    main()
