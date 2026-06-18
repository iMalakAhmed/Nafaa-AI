"""Run and score API-based case-study extractors without overwriting old outputs.

Examples:
  python -m document_parsing.casestudy.api_benchmark --provider gemini --ids CS_010 --no-skip
  python -m document_parsing.casestudy.api_benchmark --provider claude --ids CS_001,CS_010 --model claude-sonnet-4-6 --no-skip
  python -m document_parsing.casestudy.api_benchmark --provider groq --model meta-llama/llama-4-scout-17b-16e-instruct
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from document_parsing.casestudy.evaluate import main as evaluate_main

IMAGES_DIR = Path("document_parsing/data/raw_images/DataSet/cast study")
LABELS_DIR = Path("document_parsing/data/case_study_labels")
OUTPUT_ROOT = Path("document_parsing/outputs/casestudy")

DEFAULT_MODELS = {
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-haiku-4-5-20251001",
}


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return slug[:80] or "model"


def _select_images(ids: str) -> list[Path]:
    images = sorted(IMAGES_DIR.glob("*.jpeg")) + sorted(IMAGES_DIR.glob("*.jpg"))
    if ids.strip():
        wanted = {s.strip() for s in ids.split(",") if s.strip()}
        images = [p for p in images if p.stem in wanted]
    return images


def _score(output_dir: Path, *, verbose: bool) -> None:
    import sys

    pred_ids = {p.stem for p in output_dir.glob("*.json")}
    label_ids = {p.stem for p in LABELS_DIR.glob("*.json")}
    if not (pred_ids & label_ids):
        print(f"[benchmark] no labels found for {pred_ids} — skipping evaluation")
        return

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "document_parsing.casestudy.evaluate",
            "--pred",
            str(output_dir),
            "--labels",
            str(LABELS_DIR),
        ]
        if verbose:
            sys.argv.append("--verbose")
        evaluate_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an API extractor and score its case-study output.")
    parser.add_argument("--provider", choices=("groq", "gemini", "claude"), required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ids", default="", help="Comma-separated stems, e.g. CS_001,CS_010")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--no-regions", action="store_true")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    images = _select_images(args.ids)
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        return

    suffix = f"{args.provider}_{_safe_slug(model)}"
    if args.ids.strip():
        suffix += f"_{_safe_slug(args.ids)}"
    output_dir = OUTPUT_ROOT / f"records_{suffix}"
    raw_dir = OUTPUT_ROOT / f"raw_{suffix}"

    use_regions = not args.no_regions
    calls = len(images) * (4 if use_regions else 1)
    print(f"[benchmark] provider={args.provider}, model={model}, images={len(images)}, regions={use_regions}")
    print(f"[benchmark] ~{calls} API call(s)")
    print(f"[benchmark] records -> {output_dir}")
    print(f"[benchmark] raw     -> {raw_dir}")

    try:
        if args.provider == "groq":
            from document_parsing.casestudy.extract_groq import run_batch_groq

            run_batch_groq(
                images,
                output_dir,
                model=model,
                max_tokens=args.max_tokens,
                use_regions=use_regions,
                skip_existing=not args.no_skip,
                raw_dir=raw_dir,
            )
        elif args.provider == "gemini":
            from document_parsing.casestudy.extract_gemini import run_batch_gemini

            run_batch_gemini(
                images,
                output_dir,
                model=model,
                max_tokens=args.max_tokens,
                use_regions=use_regions,
                skip_existing=not args.no_skip,
                raw_dir=raw_dir,
            )
        else:
            from document_parsing.casestudy.extract_claude import run_batch_claude

            run_batch_claude(
                images,
                output_dir,
                model=model,
                max_tokens=args.max_tokens,
                use_regions=use_regions,
                skip_existing=not args.no_skip,
                raw_dir=raw_dir,
            )
    except EnvironmentError as exc:
        print(f"[benchmark] stopped: {exc}")
        raise SystemExit(2) from None

    _score(output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
