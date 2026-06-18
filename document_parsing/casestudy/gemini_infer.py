"""Run case-study extraction locally using the Google Gemini API (free tier).

1. Get a free API key at: aistudio.google.com -> Get API key
2. Set it:  $env:GEMINI_API_KEY = "AIza..."
3. Run:     python -m document_parsing.casestudy.gemini_infer

Options:
  --ids CS_001,CS_002    only those documents
  --no-regions           full-page pass only (4x fewer API calls)
  --no-skip              re-run even if output exists
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGES_DIR = Path("document_parsing/data/raw_images/DataSet/cast study")
OUT_DIR    = Path("document_parsing/outputs/casestudy/records_gemini")
RAW_DIR    = Path("document_parsing/outputs/casestudy/raw_gemini")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="gemini-2.5-flash")
    parser.add_argument("--ids",        default="")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--no-regions", action="store_true")
    parser.add_argument("--no-skip",    action="store_true")
    parser.add_argument("--out",        default=str(OUT_DIR))
    args = parser.parse_args()

    images = sorted(IMAGES_DIR.glob("*.jpeg")) + sorted(IMAGES_DIR.glob("*.jpg"))
    if args.ids.strip():
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        images = [p for p in images if p.stem in wanted]

    if not images:
        print(f"No images found in {IMAGES_DIR}")
        return

    use_regions = not args.no_regions
    calls = len(images) * (4 if use_regions else 1)
    print(f"[casestudy-gemini] {len(images)} image(s), model={args.model}, regions={use_regions}")
    print(f"[casestudy-gemini] ~{calls} API calls total (free tier: 1500/day)")

    from document_parsing.casestudy.extract_gemini import run_batch_gemini
    run_batch_gemini(
        images,
        args.out,
        model=args.model,
        max_tokens=args.max_tokens,
        use_regions=use_regions,
        skip_existing=not args.no_skip,
        raw_dir=str(RAW_DIR),
    )


if __name__ == "__main__":
    main()
