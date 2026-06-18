"""Run birth-certificate extraction locally using the Google Gemini API (free tier).

1. Get a free API key at: aistudio.google.com -> Get API key
2. Set it:  $env:GEMINI_API_KEY = "AIza..."
3. Run:     python -m document_parsing.birthcert.gemini_infer

Options:
  --ids BC_00001,BC_00002    only those documents
  --no-skip                  re-run even if output exists
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGES_DIR = Path("document_parsing/data/raw_images/DataSet/Birth Certificate")
OUT_DIR    = Path("document_parsing/outputs/birthcert/records_gemini")
RAW_DIR    = Path("document_parsing/outputs/birthcert/raw_gemini")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="gemini-2.5-flash")
    parser.add_argument("--ids",        default="")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--no-skip",    action="store_true")
    parser.add_argument("--out",        default=str(OUT_DIR))
    args = parser.parse_args()

    images = (
        sorted(IMAGES_DIR.glob("*.jpeg"))
        + sorted(IMAGES_DIR.glob("*.jpg"))
        + sorted(IMAGES_DIR.glob("*.png"))
    )
    if args.ids.strip():
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        images = [p for p in images if p.stem in wanted]

    if not images:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"[birthcert-gemini] {len(images)} image(s), model={args.model}")
    print(f"[birthcert-gemini] ~{len(images)} API call(s) (free tier: 1500/day)")

    from document_parsing.birthcert.extract_gemini import run_batch_gemini
    run_batch_gemini(
        images,
        args.out,
        model=args.model,
        max_tokens=args.max_tokens,
        skip_existing=not args.no_skip,
        raw_dir=str(RAW_DIR),
    )


if __name__ == "__main__":
    main()
