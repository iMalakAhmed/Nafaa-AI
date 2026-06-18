"""Quick fill-rate summary over a folder of extracted birth-certificate records.

Tells you, at a glance, how many documents produced a value for each field — a
fast sanity check that the pipeline is extracting data (not silently all-null).

Usage:
  python -m birthcert.summarize
  python -m birthcert.summarize --pred outputs/birthcert/records
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import SCALAR_FIELD_PATHS, get_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fill rates of extracted records.")
    parser.add_argument("--pred", type=str, default="outputs/birthcert/records")
    args = parser.parse_args()

    folder = Path(args.pred)
    files = sorted(folder.glob("*.json"))
    if not files:
        print(f"No records found in {folder}")
        return

    records = []
    for f in files:
        try:
            records.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"[warn] bad json: {f.name}")

    n = len(records)
    counts = {p: 0 for p in SCALAR_FIELD_PATHS}
    review = 0
    for rec in records:
        if rec.get("review_required"):
            review += 1
        for p in SCALAR_FIELD_PATHS:
            if get_path(rec, p) is not None:
                counts[p] += 1

    print(f"\nRecords: {n}    flagged for review: {review} ({review / n:.0%})")
    print("=" * 64)
    print(f"{'field':<48}{'filled':>8}{'rate':>8}")
    print("-" * 64)
    for p in SCALAR_FIELD_PATHS:
        c = counts[p]
        print(f"{p:<48}{c:>8}{c / n:>7.0%}")
    total_cells = n * len(SCALAR_FIELD_PATHS)
    total_filled = sum(counts.values())
    print("-" * 64)
    print(f"{'OVERALL':<48}{total_filled:>8}{total_filled / total_cells:>7.0%}")


if __name__ == "__main__":
    main()
