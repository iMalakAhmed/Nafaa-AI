"""Case-study extractor using the Anthropic Claude API.

Claude handles Arabic cursive handwriting significantly better than a 3B local
model. Uses the same prompt, regions, merge, and validation as the Qwen pipeline
so output records are identical in shape.

Usage:
    from document_parsing.casestudy.extract_claude import run_batch_claude

    run_batch_claude(
        image_paths,
        output_dir="document_parsing/outputs/casestudy/records_claude",
        model="claude-haiku-4-5-20251001",   # or claude-sonnet-4-6 for best quality
        use_regions=True,
    )

Requires: ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from .extract import _merge_crop_raw
from .jsonparse import extract_first_json_object
from .preprocess import prepare
from .prompt import SYSTEM_PROMPT, build_region_instruction, build_user_instruction
from .regions import iter_crops
from .schema import empty_record
from .validate import validate_record

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _image_to_b64(image) -> tuple[str, str]:
    """Convert a PIL image to base64-encoded JPEG for the Claude API."""
    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return "image/jpeg", data


def _call_claude(
    client,
    image,
    instruction: str,
    model: str,
    max_tokens: int,
) -> str:
    """Send one image + instruction to Claude and return the raw text response."""
    media_type, data = _image_to_b64(image)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ],
    )
    return response.content[0].text if response.content else ""


def extract_one(
    image_path: str | Path,
    *,
    client=None,
    document_id: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    enhance_image: bool = True,
    use_regions: bool = True,
) -> tuple[dict[str, Any], str]:
    """Extract one case-study image using Claude. Returns (record, raw_text)."""
    import anthropic as _anthropic

    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        client = _anthropic.Anthropic(api_key=api_key)

    image_path = Path(image_path)
    doc_id = document_id or image_path.stem
    source = str(image_path).replace("\\", "/")
    full_image = prepare(image_path, enhance_image=enhance_image)

    # ── Pass 1: full page ────────────────────────────────────────────────────
    instruction = build_user_instruction(doc_id, source)
    try:
        raw_text = _call_claude(client, full_image, instruction, model, max_tokens)
    except Exception as exc:
        record = empty_record(document_id=doc_id, source_files=[source])
        record["review_required"] = True
        record["review_notes"] = [f"Claude API call failed: {exc!r}"]
        return record, ""

    try:
        raw_obj = extract_first_json_object(raw_text)
    except ValueError:
        record = empty_record(document_id=doc_id, source_files=[source])
        record["review_required"] = True
        record["review_notes"] = ["Claude output was not valid JSON"]
        return record, raw_text

    if not use_regions:
        return validate_record(raw_obj, document_id=doc_id, source_files=[source]), raw_text

    # ── Pass 2: region crops ─────────────────────────────────────────────────
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
        region_instruction = build_region_instruction(
            doc_id, source, region.name, region.description
        )
        try:
            crop_text = _call_claude(client, crop_image, region_instruction, model, crop_tokens)
            crop_obj = extract_first_json_object(crop_text)
            raw_obj = _merge_crop_raw(raw_obj, crop_obj)
            raw_texts.append(f"\n--- region: {region.name} ---\n{crop_text}")
            print(f"[casestudy-claude] {doc_id}/{region.name}: merged", flush=True)
        except Exception as exc:
            print(f"[casestudy-claude] {doc_id}/{region.name}: skipped ({exc})", flush=True)

    combined_raw = "\n".join(raw_texts)
    return validate_record(raw_obj, document_id=doc_id, source_files=[source]), combined_raw


def run_batch_claude(
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
    """Run Claude extraction on a list of images, writing one JSON per image."""
    import anthropic as _anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Run: $env:ANTHROPIC_API_KEY = 'sk-ant-...'"
        )
    client = _anthropic.Anthropic(api_key=api_key)

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
            print(f"[casestudy-claude] skip existing {image_path.stem}", flush=True)
            continue
        print(f"[casestudy-claude] ({idx}/{total}) {image_path.name} model={model}", flush=True)
        record, raw_text = extract_one(
            image_path,
            client=client,
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
        print(f"[casestudy-claude] wrote {out_file.name}", flush=True)

    print(f"[casestudy-claude] done — {len(written)} record(s) in {output_dir}", flush=True)
    return written
