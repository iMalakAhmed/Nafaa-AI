"""Deterministic post-validation for printed social-insurance documents.

Same anti-hallucination approach as birthcert/validate.py:
  - Reject placeholder values, impossible IDs, bad enums.
  - Cross-check: national ID encodes DOB, gender, and governorate — use it
    to fill blanks and flag contradictions.
  - Normalise insurance_status to one of three canonical strings.
  - Enforce document-type discipline: null out the wrong section.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import (
    FIELD_KINDS,
    KIND_DATE,
    KIND_GENDER,
    KIND_NATIONAL_ID,
    empty_record,
    get_path,
    set_path,
)

# Arabic-Indic / Eastern-Arabic digits -> ASCII.
_DIGIT_MAP = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

_PLACEHOLDERS = {
    ".", "-", "--", "...", "....", "null", "none", "n/a", "na", "غير محدد",
    "لا يوجد", "غير معروف", "غير معروفة", "غير متوفر", "غير متوفرة",
    "غير معلوم", "غير مذكور", "غير مذكورة", "مجهول", "مجهولة",
    "xxxxxxxxxx", "string", "<value>",
}

_GENDER_MALE   = {"ذكر", "ذكر.", "male", "m", "ذ", "1"}
_GENDER_FEMALE = {"أنثى", "انثى", "أنثي", "انثي", "female", "f", "أ", "2"}

# Canonical insurance status strings (must match exactly in the schema description).
_STATUS_INSURED     = "مؤمن عليه حاليا"
_STATUS_NOT_INSURED = "غير مؤمن عليه حاليا"
_STATUS_NO_DATA     = "لا توجد بيانات تأمينية"

_STATUS_MAP: list[tuple[list[str], str]] = [
    (["مؤمن عليه حاليا", "مؤمن عليه", "مؤمن حاليا"],          _STATUS_INSURED),
    (["غير مؤمن عليه حاليا", "غير مؤمن عليه", "غير مؤمن"],   _STATUS_NOT_INSURED),
    (["لا توجد بيانات تأمينية", "لا توجد بيانات", "لا بيانات"], _STATUS_NO_DATA),
    (["عير مؤمن", "غيرمؤمن"],                                  _STATUS_NOT_INSURED),
]

_GOVERNORATES = {
    "01": "القاهرة",    "02": "الإسكندرية", "03": "بورسعيد",   "04": "السويس",
    "11": "دمياط",      "12": "الدقهلية",   "13": "الشرقية",   "14": "القليوبية",
    "15": "كفر الشيخ",  "16": "الغربية",    "17": "المنوفية",  "18": "البحيرة",
    "19": "الإسماعيلية","21": "الجيزة",     "22": "بني سويف",  "23": "الفيوم",
    "24": "المنيا",     "25": "أسيوط",      "26": "سوهاج",     "27": "قنا",
    "28": "أسوان",      "29": "الأقصر",     "31": "البحر الأحمر","32": "الوادي الجديد",
    "33": "مطروح",      "34": "شمال سيناء", "35": "جنوب سيناء","88": "خارج مصر",
}


def normalize_digits(text: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    v = value.strip().strip("‏‎").strip()
    return v or None


def is_placeholder(value: str) -> bool:
    v = value.strip()
    if not v or v.lower() in _PLACEHOLDERS:
        return True
    if set(v) <= {".", " ", "-", "_", "،", "*"}:
        return True
    digits = [c for c in normalize_digits(v) if c.isdigit()]
    if len(digits) >= 5:
        if len(set(digits)) == 1:
            return True
        joined = "".join(digits)
        if joined in "01234567890123456789" or joined in "98765432109876543210":
            return True
    return False


def _valid_national_id(digits: str) -> bool:
    if len(digits) != 14 or not digits.isdigit():
        return False
    if digits[0] not in "23":
        return False
    month = int(digits[3:5])
    day   = int(digits[5:7])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return False
    return digits[7:9] in _GOVERNORATES


def decode_national_id(digits: str) -> dict[str, Any] | None:
    if not _valid_national_id(digits):
        return None
    century = 1900 if digits[0] == "2" else 2000
    year    = century + int(digits[1:3])
    month   = int(digits[3:5])
    day     = int(digits[5:7])
    gender  = "ذكر" if int(digits[12]) % 2 == 1 else "أنثى"
    return {
        "date_of_birth": f"{day:02d}/{month:02d}/{year}",
        "gender":        gender,
        "governorate":   _GOVERNORATES.get(digits[7:9]),
        "gov_code":      digits[7:9],
    }


def clean_national_id(value: str) -> tuple[str | None, bool]:
    digits = re.sub(r"\D", "", normalize_digits(value))
    if not digits:
        return None, False
    if _valid_national_id(digits):
        return digits, True
    if is_placeholder(digits):
        return None, False
    if 10 <= len(digits) <= 16:
        return digits, False
    return None, False


def _normalize_gender(value: str) -> str | None:
    v = value.strip().lower()
    if v in {g.lower() for g in _GENDER_MALE}:
        return "ذكر"
    if v in {g.lower() for g in _GENDER_FEMALE}:
        return "أنثى"
    return None


def _normalize_insurance_status(value: str) -> str | None:
    v = value.strip()
    for keywords, canonical in _STATUS_MAP:
        for kw in keywords:
            if kw in v:
                return canonical
    return None


def _normalize_date(value: str) -> tuple[str | None, bool]:
    norm = normalize_digits(value).strip()
    if is_placeholder(norm):
        return None, False
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})", norm)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mo > 12 and d <= 12:
            d, mo = mo, d
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return norm, True
        return norm, False
    if re.fullmatch(r"\d{4}", norm) and 1900 <= int(norm) <= 2099:
        return norm, True
    return (norm or None), False


def _find_block(raw: dict[str, Any], key: str) -> dict[str, Any] | None:
    block = raw.get(key)
    if isinstance(block, dict):
        return block
    for value in raw.values():
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            return value[key]
    return None


def _arabic_score(text: str) -> int:
    return sum(1 for ch in text if "\u0600" <= ch <= "\u06ff")


def repair_mojibake_text(value: str) -> str:
    """Repair common UTF-8-as-Latin-1 mojibake from model output."""
    if not any(ch in value for ch in ("Ø", "Ù", "Û", "â")):
        return value
    for encoding in ("cp1252", "latin1"):
        try:
            fixed = value.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        if _arabic_score(fixed) > _arabic_score(value):
            return fixed
    return value


def repair_mojibake(value: Any) -> Any:
    if isinstance(value, str):
        return repair_mojibake_text(value)
    if isinstance(value, list):
        return [repair_mojibake(item) for item in value]
    if isinstance(value, dict):
        fixed: dict[Any, Any] = {}
        for key, item in value.items():
            fixed[repair_mojibake_text(key) if isinstance(key, str) else key] = repair_mojibake(item)
        return fixed
    return value


def normalize_structure(raw: dict[str, Any]) -> dict[str, Any]:
    """Hoist misplaced top-level blocks so downstream validation sees a clean shape."""
    fixed = dict(raw)
    for key in ("document_header", "ids", "personal", "social_insurance", "payroll"):
        block = _find_block(raw, key)
        if block is not None:
            fixed[key] = block
    return fixed


def validate_record(raw: dict[str, Any], *, document_id: str, source_files: list[str]) -> dict[str, Any]:
    """Coerce a raw model dict into a clean, schema-shaped, validated record."""
    raw    = repair_mojibake(raw)
    raw    = normalize_structure(raw)
    record = empty_record(document_id=document_id, source_files=source_files)
    uncertain: list[str] = []
    notes:     list[str] = []

    doc_type = _clean(get_path(raw, "document_header.document_type")) or ""

    for path, kind in FIELD_KINDS.items():
        # Skip type-specific fields for the wrong sub-type.
        if path.startswith("social_insurance.") and "payroll" in doc_type:
            continue
        if path.startswith("payroll.") and "inquiry" in doc_type:
            continue

        raw_value = _clean(get_path(raw, path))
        if raw_value is None:
            continue
        if is_placeholder(raw_value):
            uncertain.append(path)
            continue

        if kind == KIND_NATIONAL_ID:
            cleaned, confident = clean_national_id(raw_value)
            if cleaned is None:
                uncertain.append(path)
                continue
            set_path(record, path, cleaned)
            if not confident:
                uncertain.append(path)
                notes.append(f"{path}: national id is not 14 valid digits ({cleaned}).")

        elif kind == KIND_GENDER:
            norm = _normalize_gender(raw_value)
            if norm is None:
                uncertain.append(path)
                continue
            set_path(record, path, norm)

        elif kind == KIND_DATE:
            norm, confident = _normalize_date(raw_value)
            if norm is None:
                uncertain.append(path)
                continue
            set_path(record, path, norm)
            if not confident:
                uncertain.append(path)

        else:  # KIND_TEXT — including insurance_status, which we normalise specially.
            if path == "social_insurance.insurance_status":
                norm = _normalize_insurance_status(raw_value)
                set_path(record, path, norm if norm else raw_value)
                if norm is None:
                    uncertain.append(path)
            else:
                set_path(record, path, raw_value)

    # Cross-check: national ID deterministically encodes DOB, gender, governorate.
    nat_id  = record["ids"]["national_id"]
    decoded = decode_national_id(nat_id) if nat_id else None
    if decoded:
        if record["personal"]["date_of_birth"] is None:
            set_path(record, "personal.date_of_birth", decoded["date_of_birth"])
            notes.append("personal.date_of_birth derived from national id.")

        gender = record["personal"]["gender"]
        if gender is None:
            set_path(record, "personal.gender", decoded["gender"])
            notes.append("personal.gender derived from national id.")
        elif gender != decoded["gender"]:
            uncertain.append("personal.gender")
            notes.append(
                f"personal.gender '{gender}' contradicts national id "
                f"(implies '{decoded['gender']}')."
            )

        if record["personal"]["governorate_name"] is None and decoded["governorate"]:
            set_path(record, "personal.governorate_name", decoded["governorate"])
            set_path(record, "personal.governorate_code", decoded["gov_code"])
            notes.append("personal.governorate derived from national id.")

    # Null out the wrong section based on document type.
    if "payroll" in doc_type:
        record["social_insurance"] = None
    elif "inquiry" in doc_type:
        record["payroll"] = None

    # Pass through form_codes array from payroll (not a scalar, handled separately).
    raw_payroll = raw.get("payroll")
    if isinstance(raw_payroll, dict) and record.get("payroll") is not None:
        codes = raw_payroll.get("form_codes")
        if isinstance(codes, list):
            record["payroll"]["form_codes"] = [c for c in codes if isinstance(c, str)]

    # Merge model-declared uncertainty / notes.
    model_uncertain = raw.get("uncertain_fields")
    if isinstance(model_uncertain, list):
        for item in model_uncertain:
            if isinstance(item, str) and item not in uncertain:
                uncertain.append(item)
    model_notes = raw.get("review_notes")
    if isinstance(model_notes, list):
        notes.extend(n for n in model_notes if isinstance(n, str))

    record["uncertain_fields"] = sorted(set(uncertain))
    record["review_notes"]     = notes

    filled = sum(1 for p in FIELD_KINDS if get_path(record, p) is not None)
    no_identity = (
        record["ids"]["national_id"] is None
        and record["ids"]["insurance_number"] is None
        and record["personal"]["full_name_triple"] is None
    )
    record["review_required"] = (
        bool(raw.get("review_required")) or filled == 0 or len(uncertain) > 8 or no_identity
    )
    return record
