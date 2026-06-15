"""Gemini extractor for Egyptian national ID cards."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "gemini-2.5-flash-lite"

_DIGIT_TRANS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

_GOV_CODES = {
    "01": "القاهرة",
    "02": "الإسكندرية",
    "03": "بورسعيد",
    "04": "السويس",
    "11": "دمياط",
    "12": "الدقهلية",
    "13": "الشرقية",
    "14": "القليوبية",
    "15": "كفر الشيخ",
    "16": "الغربية",
    "17": "المنوفية",
    "18": "البحيرة",
    "19": "الإسماعيلية",
    "21": "الجيزة",
    "22": "بني سويف",
    "23": "الفيوم",
    "24": "المنيا",
    "25": "أسيوط",
    "26": "سوهاج",
    "27": "قنا",
    "28": "أسوان",
    "29": "الأقصر",
    "31": "البحر الأحمر",
    "32": "الوادي الجديد",
    "33": "مطروح",
    "34": "شمال سيناء",
    "35": "جنوب سيناء",
    "88": "خارج مصر",
}


def _extract_first_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.DOTALL)
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in Gemini output")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(candidate)):
        char = candidate[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                obj = json.loads(candidate[start : index + 1])
                if isinstance(obj, dict):
                    return obj
    raise ValueError("Could not parse a complete JSON object from Gemini output")


def _image_to_bytes(image_path: str | Path) -> bytes:
    from PIL import Image
    import io

    image = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _image_crop_to_bytes(image_path: str | Path, box: tuple[float, float, float, float]) -> bytes:
    from PIL import Image
    from PIL import ImageEnhance
    import io

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    left, top, right, bottom = box
    crop = image.crop((
        int(width * left),
        int(height * top),
        int(width * right),
        int(height * bottom),
    ))
    crop = crop.resize((crop.width * 4, crop.height * 4))
    crop = ImageEnhance.Contrast(crop).enhance(1.8)
    crop = ImageEnhance.Sharpness(crop).enhance(2.0)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=98)
    return buf.getvalue()


def _normalize_digits(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).translate(_DIGIT_TRANS)
    return re.sub(r"\D+", "", text)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _governorate_from_text(*values: Any) -> str | None:
    text = " ".join(str(value) for value in values if value)
    for gov in _GOV_CODES.values():
        if gov and gov in text:
            return gov
    return None


def _valid_national_id(digits: str) -> bool:
    if len(digits) != 14 or digits[0] not in "23":
        return False
    try:
        month = int(digits[3:5])
        day = int(digits[5:7])
    except ValueError:
        return False
    return 1 <= month <= 12 and 1 <= day <= 31 and digits[7:9] in _GOV_CODES


def _best_national_id(value: Any) -> tuple[str | None, bool]:
    text = "" if value is None else str(value)
    digits = _normalize_digits(text)
    candidates = re.findall(r"[23]\d{13}", digits)
    if _valid_national_id(digits):
        return digits, True
    for candidate in candidates:
        if _valid_national_id(candidate):
            return candidate, True
    if len(digits) == 14:
        return digits, False
    return None, False


def _decode_national_id(digits: str | None) -> dict[str, str] | None:
    if not digits or not _valid_national_id(digits):
        return None
    century = 1900 if digits[0] == "2" else 2000
    year = century + int(digits[1:3])
    month = int(digits[3:5])
    day = int(digits[5:7])
    return {
        "birth_date": f"{day:02d}/{month:02d}/{year}",
        "governorate_code": digits[7:9],
        "governorate_from_id": _GOV_CODES.get(digits[7:9], ""),
        "gender": "ذكر" if int(digits[12]) % 2 == 1 else "أنثى",
    }


def _prompt(document_id: str, source: str) -> str:
    return f"""You are reading an Egyptian national ID card image.

Return ONLY one valid JSON object. No markdown, no explanation.

Read only visible text from the card. Do not guess.
The national ID number is the 14 large digits at the bottom of the card.
Arabic-Indic digits are allowed in the image, but return national_id as ASCII digits only.

Schema:
{{
  "document_id": "{document_id}",
  "document_type": "national_id",
  "full_name": null,
  "first_name": null,
  "remaining_name": null,
  "address": null,
  "street": null,
  "city": null,
  "governorate": null,
  "national_id": null,
  "serial_number": null,
  "review_required": false,
  "review_notes": []
}}

Source file: {source}
"""


def _model_candidates(model: str) -> list[str]:
    requested = [item.strip() for item in model.split(",") if item.strip()]
    defaults = ["gemini-2.5-flash-lite"]
    out: list[str] = []
    for item in requested + defaults:
        if item not in out:
            out.append(item)
    return out


def _call_gemini_bytes(api_key: str, model: str, image_bytes: bytes, instruction: str, max_tokens: int) -> str:
    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None
    for candidate in _model_candidates(model):
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=instruction),
                    ],
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                return response.text or ""
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if "503" not in message and "UNAVAILABLE" not in message and "high demand" not in message:
                    raise
                if attempt == 0:
                    time.sleep(2)
                    continue
                print(f"[national-id-gemini] {candidate} unavailable, trying fallback", flush=True)
                break
    raise RuntimeError(f"Gemini national ID extraction failed: {last_error}")


def _call_gemini(api_key: str, model: str, image_path: str | Path, instruction: str, max_tokens: int) -> str:
    return _call_gemini_bytes(api_key, model, _image_to_bytes(image_path), instruction, max_tokens)


def _read_id_from_crops(api_key: str, model: str, image_path: str | Path) -> str | None:
    instruction = """Read the Egyptian national ID number only.

The image is cropped around the large black ID digits at the bottom of an Egyptian ID card.
Ignore dates, the photo, the serial number, and any watermark.
Arabic zero is written as ٠ and may look like a small dot. Do not omit those zeros.
Read the 14-digit ID from left to right as printed in the digit line.
Return ONLY this JSON:
{"national_id": "14 ASCII digits or null"}

The value must contain exactly 14 digits. Count the digits before returning.
"""
    crop_boxes = [
        (0.34, 0.66, 0.98, 0.88),
        (0.28, 0.62, 0.98, 0.90),
        (0.40, 0.68, 0.96, 0.84),
    ]
    best: str | None = None
    for box in crop_boxes:
        raw_text = _call_gemini_bytes(
            api_key,
            model,
            _image_crop_to_bytes(image_path, box),
            instruction,
            max_tokens=256,
        )
        try:
            raw_obj = _extract_first_json_object(raw_text)
        except Exception:
            continue
        candidate, confident = _best_national_id(raw_obj.get("national_id"))
        if confident:
            return candidate
        candidate_digits = _normalize_digits(raw_obj.get("national_id"))
        if len(candidate_digits) == 14:
            best = candidate_digits
    return best


def _repair_invalid_id_with_context(
    api_key: str,
    model: str,
    image_path: str | Path,
    raw_obj: dict[str, Any],
) -> str | None:
    instruction = f"""Re-read the Egyptian national ID number from this card.

The previous OCR produced this JSON:
{json.dumps(raw_obj, ensure_ascii=False)}

That national_id is invalid. Egyptian national IDs have 14 digits:
- digit 1 is 2 or 3
- digits 2-3 are birth year
- digits 4-5 are birth month, 01-12
- digits 6-7 are birth day, 01-31
- digits 8-9 are governorate code, for example Minya is 24

The card address may mention the governorate/city. Use these rules only to reject impossible OCR,
then re-read the visible large black digit line at the bottom. Arabic zero ٠ may look like a dot.

Return ONLY:
{{"national_id": "14 ASCII digits or null"}}
"""
    raw_text = _call_gemini(api_key, model, image_path, instruction, max_tokens=512)
    raw = _extract_first_json_object(raw_text)
    candidate, confident = _best_national_id(raw.get("national_id"))
    return candidate if confident else None


def validate_record(raw: dict[str, Any], *, document_id: str, source_files: list[str]) -> dict[str, Any]:
    notes: list[str] = []
    model_notes = raw.get("review_notes")
    if isinstance(model_notes, list):
        notes.extend(str(item) for item in model_notes if item)

    raw_national_id = _normalize_digits(raw.get("national_id")) or None
    national_id, confident_id = _best_national_id(raw.get("national_id"))
    if not confident_id:
        national_id = None
    decoded = _decode_national_id(national_id)

    full_name = _clean_text(raw.get("full_name"))
    first_name = _clean_text(raw.get("first_name"))
    remaining_name = _clean_text(raw.get("remaining_name"))
    if not full_name:
        full_name = _clean_text(" ".join(x for x in (first_name, remaining_name) if x))
    if full_name and not first_name:
        first_name = full_name.split()[0]
    if full_name and not remaining_name and first_name and full_name.startswith(first_name):
        remaining_name = full_name[len(first_name):].strip() or None

    city = _clean_text(raw.get("city"))
    governorate = _clean_text(raw.get("governorate"))
    if not governorate:
        governorate = _governorate_from_text(raw.get("address"), raw.get("city"), raw.get("street"))
    if not governorate and decoded:
        governorate = decoded["governorate_from_id"]
        notes.append("governorate derived from national id")

    record = {
        "document_id": document_id,
        "document_type": "national_id",
        "source_files": source_files,
        "full_name": full_name,
        "first_name": first_name,
        "remaining_name": remaining_name,
        "address": _clean_text(raw.get("address")),
        "street": _clean_text(raw.get("street")),
        "city": city,
        "governorate": governorate,
        "national_id": national_id,
        "raw_national_id": raw_national_id,
        "serial_number": _clean_text(raw.get("serial_number")),
        "raw": raw,
        "review_required": False,
        "review_notes": notes,
    }
    if decoded:
        record.update(decoded)
    if not national_id or not confident_id:
        record["review_required"] = True
        record["review_notes"].append("National ID number was not confidently read as 14 valid digits")
    if not full_name:
        record["review_required"] = True
        record["review_notes"].append("Full name was not confidently read")
    return record


def extract_one(
    image_path: str | Path,
    *,
    api_key: str | None = None,
    document_id: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    use_extra_id_passes: bool | None = None,
) -> dict[str, Any]:
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    image_path = Path(image_path)
    doc_id = document_id or image_path.stem
    source = str(image_path).replace("\\", "/")
    raw_text = _call_gemini(key, model, image_path, _prompt(doc_id, source), max_tokens)
    raw_obj = _extract_first_json_object(raw_text)
    if use_extra_id_passes is None:
        use_extra_id_passes = os.environ.get("NATIONAL_ID_EXTRA_ID_PASSES", "").lower() in {
            "1", "true", "yes", "on"
        }
    national_id, confident = _best_national_id(raw_obj.get("national_id"))
    if use_extra_id_passes and not confident:
        crop_id = _read_id_from_crops(key, model, image_path)
        if crop_id:
            raw_obj["national_id"] = crop_id
            national_id, confident = _best_national_id(crop_id)
    if use_extra_id_passes and not confident:
        repaired_id = _repair_invalid_id_with_context(key, model, image_path, raw_obj)
        if repaired_id:
            raw_obj["national_id"] = repaired_id
    return validate_record(raw_obj, document_id=doc_id, source_files=[source])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    record = extract_one(args.image, model=args.model)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
