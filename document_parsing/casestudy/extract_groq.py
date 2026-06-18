"""Case-study extractor using the Groq API (free tier, no credit card needed).

Free tier: 14,400 requests/day, 30 req/min.
Model: meta-llama/llama-4-scout-17b-16e-instruct (vision, excellent Arabic).

Get a free API key at: console.groq.com -> API Keys -> Create key

Usage:
    $env:GROQ_API_KEY = "gsk_..."
    python -m document_parsing.casestudy.groq_infer
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from .extract import _merge_crop_raw
from .jsonparse import extract_first_json_object
from .preprocess import prepare
from .prompt import SYSTEM_PROMPT, build_region_instruction, build_user_instruction
from .regions import iter_crops
from .schema import empty_record
from .validate import validate_record

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _image_to_b64(image) -> str:
    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _call_groq(api_key: str, model: str, system: str, instruction: str, image, max_tokens: int) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_image_to_b64(image)}"},
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ],
    }
    resp = requests.post(_API_URL, headers=headers, json=payload, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Groq {resp.status_code}: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


def extract_one(
    image_path: str | Path,
    *,
    api_key: str | None = None,
    document_id: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    enhance_image: bool = True,
    use_regions: bool = True,
    rpm_delay: float = 2.5,
) -> tuple[dict[str, Any], str]:
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    image_path = Path(image_path)
    doc_id = document_id or image_path.stem
    source = str(image_path).replace("\\", "/")
    full_image = prepare(image_path, enhance_image=enhance_image)

    # Pass 1: full page
    instruction = build_user_instruction(doc_id, source)
    try:
        raw_text = _call_groq(key, model, SYSTEM_PROMPT, instruction, full_image, max_tokens)
        time.sleep(rpm_delay)
    except Exception as exc:
        record = empty_record(document_id=doc_id, source_files=[source])
        record["review_required"] = True
        record["review_notes"] = [f"Groq API call failed: {exc!r}"]
        return record, ""

    try:
        raw_obj = extract_first_json_object(raw_text)
    except ValueError:
        record = empty_record(document_id=doc_id, source_files=[source])
        record["review_required"] = True
        record["review_notes"] = ["Groq output was not valid JSON"]
        return record, raw_text

    if not use_regions:
        return validate_record(raw_obj, document_id=doc_id, source_files=[source]), raw_text

    # Pass 2: region crops
    page_side = (raw_obj.get("document") or {}).get("page_side")
    if isinstance(page_side, str):
        low = page_side.lower()
        if "front" in low or "أمام" in page_side:
            page_side = "front"
        elif "back" in low or "ظهر" in page_side:
            page_side = "back"
        else:
            page_side = None

    crop_tokens = max(1024, max_tokens // 2)
    raw_texts = [raw_text]

    for region, crop_image in iter_crops(full_image, page_side):
        region_instruction = build_region_instruction(
            doc_id, source, region.name, region.description
        )
        try:
            crop_text = _call_groq(key, model, SYSTEM_PROMPT, region_instruction, crop_image, crop_tokens)
            time.sleep(rpm_delay)
            crop_obj = extract_first_json_object(crop_text)
            raw_obj = _merge_crop_raw(raw_obj, crop_obj)
            raw_texts.append(f"\n--- region: {region.name} ---\n{crop_text}")
            print(f"[casestudy-groq] {doc_id}/{region.name}: merged", flush=True)
        except Exception as exc:
            print(f"[casestudy-groq] {doc_id}/{region.name}: skipped ({exc})", flush=True)
            time.sleep(rpm_delay)

    combined_raw = "\n".join(raw_texts)
    return validate_record(raw_obj, document_id=doc_id, source_files=[source]), combined_raw


def run_batch_groq(
    image_paths: list[str | Path],
    output_dir: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    enhance_image: bool = True,
    use_regions: bool = True,
    skip_existing: bool = True,
    raw_dir: str | Path | None = None,
) -> list[Path]:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError("GROQ_API_KEY is not set.\nRun: $env:GROQ_API_KEY = 'gsk_...'")

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
            print(f"[casestudy-groq] skip existing {image_path.stem}", flush=True)
            continue
        print(f"[casestudy-groq] ({idx}/{total}) {image_path.name}", flush=True)
        record, raw_text = extract_one(
            image_path, api_key=key, document_id=image_path.stem,
            model=model, max_tokens=max_tokens,
            enhance_image=enhance_image, use_regions=use_regions,
        )
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_file)
        if raw_path:
            (raw_path / f"{image_path.stem}.txt").write_text(raw_text, encoding="utf-8")
        print(f"[casestudy-groq] wrote {out_file.name}", flush=True)

    print(f"[casestudy-groq] done — {len(written)} record(s) in {output_dir}", flush=True)
    return written
