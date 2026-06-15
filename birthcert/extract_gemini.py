"""Birth-certificate extractor using the Google Gemini API.

Free tier: 1500 requests/day, 15 req/min.
Model: gemini-2.5-flash (best free model, excellent Arabic OCR).

Get a free API key at: aistudio.google.com -> Get API key

Usage:
    $env:GEMINI_API_KEY = "AIza..."
    python birthcert_gemini_infer.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .jsonparse import extract_first_json_object
from .preprocess import prepare
from .prompt import SYSTEM_PROMPT, build_user_instruction
from .schema import empty_record
from .validate import validate_record

DEFAULT_MODEL = "gemini-2.5-flash"


def _json_only(instruction: str) -> str:
    return (
        instruction
        + "\n\nGEMINI OUTPUT OVERRIDE:\n"
        + "Return ONLY the final JSON object. Do not include Reasoning, markdown, code fences, "
        + "bullets, commentary, or the words FINAL JSON. The response must start with { and end with }."
    )


def _image_to_bytes(image) -> bytes:
    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _call_gemini(api_key: str, model: str, system: str, instruction: str, image, max_tokens: int) -> str:
    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    img_bytes = _image_to_bytes(image)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=instruction),
        ],
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


def extract_one(
    image_path: str | Path,
    *,
    api_key: str | None = None,
    document_id: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    enhance_image: bool = True,
    rpm_delay: float = 4.5,
) -> tuple[dict[str, Any], str]:
    """Extract one birth certificate image using Gemini. Returns (record, raw_text)."""
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    image_path = Path(image_path)
    doc_id = document_id or image_path.stem
    source = str(image_path).replace("\\", "/")
    image = prepare(image_path, enhance_image=enhance_image)

    instruction = _json_only(build_user_instruction(doc_id, source))
    try:
        raw_text = _call_gemini(key, model, SYSTEM_PROMPT, instruction, image, max_tokens)
        time.sleep(rpm_delay)
    except Exception as exc:
        record = empty_record(document_id=doc_id, source_files=[source])
        record["review_required"] = True
        record["review_notes"] = [f"Gemini API call failed: {exc!r}"]
        return record, ""

    try:
        raw_obj = extract_first_json_object(raw_text)
    except ValueError:
        record = empty_record(document_id=doc_id, source_files=[source])
        record["review_required"] = True
        record["review_notes"] = ["Gemini output was not valid JSON"]
        return record, raw_text

    return validate_record(raw_obj, document_id=doc_id, source_files=[source]), raw_text


def run_batch_gemini(
    image_paths: list[str | Path],
    output_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    enhance_image: bool = True,
    skip_existing: bool = True,
    raw_dir: str | Path | None = None,
) -> list[Path]:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Run:  $env:GEMINI_API_KEY = 'AIza...'"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_dir) if raw_dir else None
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, start=1):
        image_path = Path(image_path)
        out_file = output_dir / f"{image_path.stem}.json"
        if skip_existing and out_file.exists():
            print(f"[birthcert-gemini] skip existing {image_path.stem}", flush=True)
            continue
        print(f"[birthcert-gemini] ({idx}/{total}) {image_path.name}", flush=True)
        record, raw_text = extract_one(
            image_path,
            api_key=key,
            document_id=image_path.stem,
            model=model,
            max_tokens=max_tokens,
            enhance_image=enhance_image,
        )
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_file)
        if raw_path:
            (raw_path / f"{image_path.stem}.txt").write_text(raw_text, encoding="utf-8")
        print(f"[birthcert-gemini] wrote {out_file.name}", flush=True)

    print(f"[birthcert-gemini] done — {len(written)} record(s) in {output_dir}", flush=True)
    return written
