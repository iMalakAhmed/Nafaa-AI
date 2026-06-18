"""Run case-study extraction using the Groq API (free, no credit card needed).

1. Sign up at console.groq.com
2. Create an API key (starts with gsk_)
3. Run:
   $env:GROQ_API_KEY = "gsk_..."
   python -m document_parsing.casestudy.groq_infer
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGES_DIR = Path("document_parsing/data/raw_images/DataSet/cast study")
OUT_DIR    = Path("document_parsing/outputs/casestudy/records_groq")
RAW_DIR    = Path("document_parsing/outputs/casestudy/raw_groq")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="meta-llama/llama-4-scout-17b-16e-instruct")
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
    print(f"[casestudy-groq] {len(images)} image(s), model={args.model}, regions={use_regions}")
    print(f"[casestudy-groq] ~{calls} API calls (free tier: 14,400/day)")

    from document_parsing.casestudy.extract_groq import run_batch_groq
    run_batch_groq(
        images, args.out,
        model=args.model,
        max_tokens=args.max_tokens,
        use_regions=use_regions,
        skip_existing=not args.no_skip,
        raw_dir=str(RAW_DIR),
    )


if __name__ == "__main__":
    main()
