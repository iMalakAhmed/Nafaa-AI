"""Modal smoke test for Gemini national ID extraction.

Run:
  $env:GEMINI_API_KEY = "..."
  modal run document_parsing/nationalid_modal.py --ids NID_00001
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "national-id-gemini-smoke"
PROJECT_DIR = "/root/project"
IMAGES_SUBDIR = "document_parsing/data/raw_images/DataSet/National ID"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("google-genai>=1.0.0", "Pillow>=10.0.0")
    .add_local_dir("document_parsing/nationalid", remote_path=f"{PROJECT_DIR}/document_parsing/nationalid")
    .add_local_dir("document_parsing/casestudy", remote_path=f"{PROJECT_DIR}/document_parsing/casestudy")
    .add_local_dir("document_parsing/data/raw_images/DataSet/National ID", remote_path=f"{PROJECT_DIR}/{IMAGES_SUBDIR}")
)

app = modal.App(APP_NAME)


def _select_images(ids: list[str] | None, limit: int | None) -> list[Path]:
    root = Path(PROJECT_DIR) / IMAGES_SUBDIR
    images = sorted(root.glob("*.jpeg")) + sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    if ids:
        wanted = {item.strip() for item in ids if item.strip()}
        images = [path for path in images if path.stem in wanted]
    if limit and limit > 0:
        images = images[:limit]
    return images


@app.function(image=image, timeout=10 * 60)
def run_nationalid_gemini(
    ids: list[str] | None = None,
    limit: int | None = None,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
) -> list[dict]:
    import sys

    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from document_parsing.nationalid.extract_gemini import extract_one

    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set locally.")

    records: list[dict] = []
    for image_path in _select_images(ids, limit):
        print(f"[national-id-gemini] extracting {image_path.name}", flush=True)
        try:
            records.append(
                extract_one(
                    image_path,
                    api_key=key,
                    document_id=image_path.stem,
                    model=model,
                )
            )
        except Exception as exc:
            records.append({
                "document_id": image_path.stem,
                "document_type": "national_id",
                "error": str(exc),
                "review_required": True,
            })
    return records


@app.local_entrypoint()
def main(ids: str = "NID_00001", limit: int = 0, model: str = "gemini-2.5-flash-lite"):
    id_list = [item.strip() for item in ids.split(",") if item.strip()] or None
    cap = None if limit <= 0 else limit
    records = run_nationalid_gemini.remote(
        ids=id_list,
        limit=cap,
        model=model,
        api_key=os.environ.get("GEMINI_API_KEY", ""),
    )
    for record in records:
        print(record)
