"""Case-study extractor using the Google Gemini API (REST, no SDK needed).

Free tier: 1500 requests/day, 15 req/min — plenty for 12 case-study images.
Model: gemini-2.0-flash (best free model, excellent Arabic OCR).

Get a free API key at: aistudio.google.com -> Get API key

Usage:
    $env:GEMINI_API_KEY = "AIza..."
    python casestudy_gemini_infer.py
"""

from __future__ import annotations
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

from .jsonparse import extract_first_json_object
from .preprocess import prepare
from .prompt import SYSTEM_PROMPT, build_region_instruction, build_user_instruction
from .regions import iter_crops
from .schema import FIELD_KINDS, empty_record, get_path, set_path
from .validate import validate_record

DEFAULT_MODEL = "gemini-2.5-flash"


def _merge_crop_raw(base: dict[str, Any], crop: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)

    for path in FIELD_KINDS:
        if get_path(result, path) is None:
            crop_val = get_path(crop, path)
            if crop_val is not None:
                set_path(result, path, crop_val)

    base_fam = result.get("family_members") or []
    crop_fam = crop.get("family_members") or []
    if isinstance(crop_fam, list) and len(crop_fam) > len(base_fam):
        result["family_members"] = crop_fam

    base_cb: list[dict] = result.get("checkbox_answers") or []
    crop_cb: list[dict] = crop.get("checkbox_answers") or []
    if isinstance(crop_cb, list) and crop_cb:
        seen = {
            (c.get("section", ""), c.get("question", ""), c.get("answer", ""))
            for c in base_cb
        }
        for cb in crop_cb:
            key = (cb.get("section", ""), cb.get("question", ""), cb.get("answer", ""))
            if key not in seen:
                base_cb.append(cb)
                seen.add(key)
        result["checkbox_answers"] = base_cb

    base_unc = set(result.get("uncertain_fields") or [])
    crop_unc = set(crop.get("uncertain_fields") or [])
    result["uncertain_fields"] = sorted(base_unc | crop_unc)

    return result


def _json_only(instruction: str) -> str:
    return (
        instruction
        + "\n\nGEMINI OUTPUT OVERRIDE:\n"
        + "Return ONLY the final JSON object. Do not include Reasoning, markdown, code fences, "
        + "bullets, commentary, or the words FINAL JSON. The response must start with { and end with }."
    )


def _image_to_pil_bytes(image) -> bytes:
    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _call_gemini(api_key: str, model: str, system: str, instruction: str, image, max_tokens: int) -> str:
    """Single Gemini call using the google-genai SDK (supports AQ. and AIza keys)."""
    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    img_bytes = _image_to_pil_bytes(image)
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
    use_regions: bool = True,
    rpm_delay: float = 4.5,   # stay under 15 req/min free limit
) -> tuple[dict[str, Any], str]:
    """Extract one case-study image using Gemini. Returns (record, raw_text)."""
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    image_path = Path(image_path)
    doc_id = document_id or image_path.stem
    source = str(image_path).replace("\\", "/")
    full_image = prepare(image_path, enhance_image=enhance_image)

    # ── Pass 1: full page ─────────────────────────────────────────────────
    instruction = _json_only(build_user_instruction(doc_id, source))
    try:
        raw_text = _call_gemini(key, model, SYSTEM_PROMPT, instruction, full_image, max_tokens)
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

    if not use_regions:
        return validate_record(raw_obj, document_id=doc_id, source_files=[source]), raw_text

    # ── Pass 2: region crops ──────────────────────────────────────────────
    page_side = (raw_obj.get("document") or {}).get("page_side")
    if isinstance(page_side, str):
        low = page_side.lower()
        if "front" in low or "أمام" in page_side or "وجه" in page_side:
            page_side = "front"
        elif "back" in low or "ظهر" in page_side or "خلف" in page_side:
            page_side = "back"
        else:
            page_side = None

    crop_tokens = max(1024, max_tokens // 2)
    raw_texts = [raw_text]

    for region, crop_image in iter_crops(full_image, page_side):
        region_instruction = _json_only(build_region_instruction(
            doc_id, source, region.name, region.description
        ))
        try:
            crop_text = _call_gemini(key, model, SYSTEM_PROMPT, region_instruction, crop_image, crop_tokens)
            time.sleep(rpm_delay)
            crop_obj = extract_first_json_object(crop_text)
            raw_obj = _merge_crop_raw(raw_obj, crop_obj)
            raw_texts.append(f"\n--- region: {region.name} ---\n{crop_text}")
            print(f"[casestudy-gemini] {doc_id}/{region.name}: merged", flush=True)
        except Exception as exc:
            print(f"[casestudy-gemini] {doc_id}/{region.name}: skipped ({exc})", flush=True)
            time.sleep(rpm_delay)

    combined_raw = "\n".join(raw_texts)
    return validate_record(raw_obj, document_id=doc_id, source_files=[source]), combined_raw


def run_batch_gemini(
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
            print(f"[casestudy-gemini] skip existing {image_path.stem}", flush=True)
            continue
        print(f"[casestudy-gemini] ({idx}/{total}) {image_path.name}", flush=True)
        record, raw_text = extract_one(
            image_path,
            api_key=key,
            document_id=image_path.stem,
            model=model,
            max_tokens=max_tokens,
            enhance_image=enhance_image,
            use_regions=use_regions,
        )
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_file)
        if raw_path:
            (raw_path / f"{image_path.stem}.txt").write_text(raw_text, encoding="utf-8")
        print(f"[casestudy-gemini] wrote {out_file.name}", flush=True)

    print(f"[casestudy-gemini] done — {len(written)} record(s) in {output_dir}", flush=True)
    return written
