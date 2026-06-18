"""Evaluate Qwen2.5-VL via the live Modal API endpoint — case-study version.

Usage:
    python -m document_parsing.casestudy.api_eval_trigger
    python -m document_parsing.casestudy.api_eval_trigger --limit 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Install requests: pip install requests")


DEFAULT_API_URL = "https://malak2004shah--document-extraction-api-fastapi-app.modal.run"
DEFAULT_IMAGES  = "document_parsing/data/raw_images/DataSet/cast study"
DEFAULT_LABELS  = "document_parsing/data/case_study_labels"
DEFAULT_OUT     = "document_parsing/outputs/casestudy/api_eval"


def call_api(api_url: str, image_path: Path, *, timeout: int = 120) -> dict:
    endpoint = f"{api_url.rstrip('/')}/casestudy/extract"
    with open(image_path, "rb") as fh:
        resp = requests.post(
            endpoint,
            files={"file": (image_path.name, fh, "image/jpeg")},
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--images",  default=DEFAULT_IMAGES)
    parser.add_argument("--labels",  default=DEFAULT_LABELS)
    parser.add_argument("--out",     default=DEFAULT_OUT)
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    images_dir = Path(args.images)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.png"))
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        sys.exit(f"No images found in {images_dir}")

    print(f"[api-eval] Sending {len(image_paths)} image(s) to {args.api_url} …")
    failed = 0
    for i, image_path in enumerate(image_paths, 1):
        out_file = out_dir / f"{image_path.stem}.json"
        if out_file.exists():
            print(f"  [{i}/{len(image_paths)}] {image_path.name} — cached, skip")
            continue
        print(f"  [{i}/{len(image_paths)}] {image_path.name} …", end=" ", flush=True)
        try:
            t0 = time.time()
            record = call_api(args.api_url, image_path)
            elapsed = time.time() - t0
            out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"OK ({elapsed:.1f}s)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed += 1

    print(f"\n[api-eval] Done. {len(image_paths)-failed} saved, {failed} failed.")
    if failed == len(image_paths):
        sys.exit("All requests failed — is the API deployed?  modal deploy api_modal.py")

    labels_dir = Path(args.labels)
    if labels_dir.exists():
        print("\n[api-eval] Running field-level evaluation …\n")
        eval_cmd = [
            sys.executable, "-m", "document_parsing.casestudy.evaluate",
            "--pred", str(out_dir),
            "--labels", str(labels_dir),
        ]
        subprocess.run(eval_cmd)
    else:
        print(f"[api-eval] Labels not found at {labels_dir}")


if __name__ == "__main__":
    main()
