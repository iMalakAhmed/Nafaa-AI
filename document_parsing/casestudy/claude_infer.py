"""Run case-study extraction locally using the Claude API.

No GPU needed. Set your API key first:
  $env:ANTHROPIC_API_KEY = "sk-ant-..."

Then run:
  python -m document_parsing.casestudy.claude_infer
  python -m document_parsing.casestudy.claude_infer --model claude-sonnet-4-6   # higher quality
  python -m document_parsing.casestudy.claude_infer --ids CS_001,CS_002         # specific docs
  python -m document_parsing.casestudy.claude_infer --no-regions                # full-page only
"""

from __future__ import annotations

import argparse
from pathlib import Path

IMAGES_DIR = Path("document_parsing/data/raw_images/DataSet/cast study")
OUT_DIR    = Path("document_parsing/outputs/casestudy/records_claude")
RAW_DIR    = Path("document_parsing/outputs/casestudy/raw_claude")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="claude-haiku-4-5-20251001")
    parser.add_argument("--ids",         default="",   help="Comma-separated stems, e.g. CS_001,CS_002")
    parser.add_argument("--max-tokens",  type=int, default=4096)
    parser.add_argument("--no-regions",  action="store_true")
    parser.add_argument("--no-skip",     action="store_true", help="Re-run even if output exists")
    parser.add_argument("--out",         default=str(OUT_DIR))
    args = parser.parse_args()

    images = sorted(IMAGES_DIR.glob("*.jpeg")) + sorted(IMAGES_DIR.glob("*.jpg"))
    if args.ids.strip():
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        images = [p for p in images if p.stem in wanted]

    if not images:
        print(f"No images found in {IMAGES_DIR}")
        return

    print(f"[casestudy-claude] {len(images)} image(s), model={args.model}, regions={not args.no_regions}")

    from document_parsing.casestudy.extract_claude import run_batch_claude
    run_batch_claude(
        images,
        args.out,
        model=args.model,
        max_tokens=args.max_tokens,
        use_regions=not args.no_regions,
        skip_existing=not args.no_skip,
        raw_dir=str(RAW_DIR),
    )


if __name__ == "__main__":
    main()
