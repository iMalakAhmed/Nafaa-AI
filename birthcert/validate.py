"""Deterministic post-validation — the anti-hallucination layer.

The model proposes; this module disposes. Because the template is fixed, we know a
lot about what each field MUST look like, so we can mechanically reject values that
cannot be real (placeholder fillers, wrong-length IDs, impossible dates, bad enums).
Rejected values become null and the field path is recorded so a human can review it.

This runs with zero ML — pure rules — so it is cheap and fully predictable.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import (
    FIELD_KINDS,
    KIND_DATE,
    KIND_GENDER,
    KIND_NATIONAL_ID,
    KIND_RELIGION,
    empty_record,
    get_path,
    set_path,
)

# Arabic-Indic and Eastern-Arabic digits -> ASCII.
_DIGIT_MAP = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

_PLACEHOLDERS = {
    ".", "-", "--", "...", "....", "null", "none", "n/a", "na", "غير محدد",
    "لا يوجد", "غير معروف", "غير معروفة", "غير متوفر", "غير متوفرة", "غير معلوم",
    "غير مذكور", "غير مذكورة", "مجهول", "مجهولة", "xxxxxxxxxx", "string", "<value>",
}

_GENDER_MALE = {"ذكر", "ذكر.", "male", "m", "ذ"}
_GENDER_FEMALE = {"أنثى", "انثى", "أنثي", "انثي", "female", "f", "أ"}

_RELIGION_MUSLIM = {"مسلم", "مسلمة", "muslim", "islam", "الإسلام", "الاسلام"}
_RELIGION_CHRISTIAN = {"مسيحي", "مسيحية", "مسيحى", "christian", "مسيح", "نصراني"}


def normalize_digits(text: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def _find_block(raw: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Find a top-level block by name, even if the model nested it by mistake.

    Open VLMs frequently emit `ids` / `personal_and_other` *inside* the
    `birth_certificate` object. The data is correct, just misplaced — so we hunt
    for it (top level first, then one level deep) instead of dropping it.
    """
    block = raw.get(key)
    if isinstance(block, dict):
        return block
    for value in raw.values():
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            return value[key]
    return None


def normalize_structure(raw: dict[str, Any]) -> dict[str, Any]:
    """Hoist misplaced blocks so the rest of validation sees a canonical shape."""
    fixed: dict[str, Any] = dict(raw)
    for key in ("birth_certificate", "ids", "personal_and_other"):
        block = _find_block(raw, key)
        if block is not None:
            fixed[key] = block
    # If ids/personal_and_other were lifted out of birth_certificate, drop the
    # nested duplicates so they don't shadow anything downstream.
    bc = fixed.get("birth_certificate")
    if isinstance(bc, dict):
        bc = dict(bc)
        bc.pop("ids", None)
        bc.pop("personal_and_other", None)
        fixed["birth_certificate"] = bc
    return fixed


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    v = value.strip().strip("\u200f\u200e").strip()
    return v or None


def is_placeholder(value: str) -> bool:
    """True for obvious filler / fabricated values that are never real fields."""
    v = value.strip()
    if not v or v.lower() in _PLACEHOLDERS:
        return True
    if set(v) <= {".", " ", "-", "_", "،", "*"}:
        return True
    digits = [c for c in normalize_digits(v) if c.isdigit()]
    if len(digits) >= 5:
        if len(set(digits)) == 1:  # 7777777 / 0000000
            return True
        joined = "".join(digits)
        if joined in "01234567890123456789" or joined in "98765432109876543210":
            return True
    return False


# Egyptian governorate codes (digits 8-9 of the national ID).
_GOVERNORATES = {
    "01": "القاهرة", "02": "الإسكندرية", "03": "بورسعيد", "04": "السويس",
    "11": "دمياط", "12": "الدقهلية", "13": "الشرقية", "14": "القليوبية",
    "15": "كفر الشيخ", "16": "الغربية", "17": "المنوفية", "18": "البحيرة",
    "19": "الإسماعيلية", "21": "الجيزة", "22": "بني سويف", "23": "الفيوم",
    "24": "المنيا", "25": "أسيوط", "26": "سوهاج", "27": "قنا", "28": "أسوان",
    "29": "الأقصر", "31": "البحر الأحمر", "32": "الوادي الجديد", "33": "مطروح",
    "34": "شمال سيناء", "35": "جنوب سيناء", "88": "خارج جمهورية مصر العربية",
}


def _valid_national_id(digits: str) -> bool:
    """Egyptian national ID = 14 digits encoding birth date in positions 2-7."""
    if len(digits) != 14 or not digits.isdigit():
        return False
    if digits[0] not in "23":  # century marker (2 = 1900s, 3 = 2000s)
        return False
    month = int(digits[3:5])
    day = int(digits[5:7])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return False
    return digits[7:9] in _GOVERNORATES


def decode_national_id(digits: str) -> dict[str, Any] | None:
    """Deterministically derive birth date / gender / governorate from a valid ID.

    The Egyptian national ID is a structured, self-describing code, so when the
    model reads all 14 digits correctly we can compute these fields exactly instead
    of trusting the model's separate (often misread) transcription of them.
    """
    if not _valid_national_id(digits):
        return None
    century = 1900 if digits[0] == "2" else 2000
    year = century + int(digits[1:3])
    month = int(digits[3:5])
    day = int(digits[5:7])
    # 13th digit: odd => male, even => female.
    gender = "ذكر" if int(digits[12]) % 2 == 1 else "أنثى"
    return {
        "date_of_birth": f"{day:02d}/{month:02d}/{year}",
        "gender": gender,
        "governorate": _GOVERNORATES.get(digits[7:9]),
    }


def clean_national_id(value: str) -> tuple[str | None, bool]:
    """Return (cleaned_id_or_None, is_confident). Keeps readable-but-imperfect IDs."""
    digits = re.sub(r"\D", "", normalize_digits(value))
    if not digits:
        return None, False
    if _valid_national_id(digits):
        return digits, True
    if is_placeholder(digits):
        return None, False
    # Readable but wrong length / failing checksum-ish rules: keep it but flag.
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


def _normalize_religion(value: str) -> str | None:
    v = value.strip().lower()
    if v in {r.lower() for r in _RELIGION_MUSLIM}:
        return "مسلم"
    if v in {r.lower() for r in _RELIGION_CHRISTIAN}:
        return "مسيحي"
    return None


def _normalize_date(value: str) -> tuple[str | None, bool]:
    """Keep a date if it has a plausible day/month/year; else keep raw but flag."""
    norm = normalize_digits(value).strip()
    if is_placeholder(norm):
        return None, False
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})", norm)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # The form is little-endian (DD/MM/YYYY); swap if it looks reversed.
        if month > 12 and day <= 12:
            day, month = month, day
        if 1 <= month <= 12 and 1 <= day <= 31:
            return norm, True
        return norm, False
    if re.fullmatch(r"\d{4}", norm) and 1900 <= int(norm) <= 2099:
        return norm, True
    return (norm or None), False


def validate_record(raw: dict[str, Any], *, document_id: str, source_files: list[str]) -> dict[str, Any]:
    """Coerce a raw model dict into a clean, schema-shaped, validated record."""
    raw = normalize_structure(raw)
    record = empty_record(document_id=document_id, source_files=source_files)
    uncertain: list[str] = []
    notes: list[str] = []

    for path, kind in FIELD_KINDS.items():
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
        elif kind == KIND_RELIGION:
            norm = _normalize_religion(raw_value)
            if norm is None:
                # Unknown religion string is suspicious but may be legitimate; keep raw, flag.
                set_path(record, path, raw_value)
                uncertain.append(path)
            else:
                set_path(record, path, norm)
        elif kind == KIND_DATE:
            norm, confident = _normalize_date(raw_value)
            if norm is None:
                uncertain.append(path)
                continue
            set_path(record, path, norm)
            if not confident:
                uncertain.append(path)
        else:  # free text
            set_path(record, path, raw_value)

    # Cross-check: the child's national id deterministically encodes birth date,
    # gender, and governorate. Use it to fill blanks and flag contradictions — this
    # is exact arithmetic, so it never hallucinates.
    child_id = record["ids"]["child_national_id"]
    decoded = decode_national_id(child_id) if child_id else None
    if decoded:
        dob = record["personal_and_other"]["child"]["date_of_birth"]
        if dob is None:
            set_path(record, "personal_and_other.child.date_of_birth", decoded["date_of_birth"])
            notes.append("child.date_of_birth derived from national id.")

        gender = record["personal_and_other"]["child"]["gender"]
        if gender is None:
            set_path(record, "personal_and_other.child.gender", decoded["gender"])
            notes.append("child.gender derived from national id.")
        elif gender != decoded["gender"]:
            uncertain.append("personal_and_other.child.gender")
            notes.append(
                f"child.gender '{gender}' contradicts national id (which implies "
                f"'{decoded['gender']}')."
            )

        if decoded["governorate"] and record["birth_certificate"]["governorate_or_administration"] is None:
            set_path(
                record,
                "birth_certificate.governorate_or_administration",
                decoded["governorate"],
            )
            notes.append("governorate derived from national id.")

    # Pass through any extra other_ids the model found, cleaned.
    other_ids = get_path(raw, "ids.other_ids")
    cleaned_other: list[dict[str, Any]] = []
    if isinstance(other_ids, list):
        for item in other_ids:
            if not isinstance(item, dict):
                continue
            label = _clean(item.get("label_ar"))
            value = _clean(item.get("value"))
            if value and not is_placeholder(value):
                cleaned_other.append({"label_ar": label, "value": value})
    record["ids"]["other_ids"] = cleaned_other

    # Merge model-declared uncertainty/notes with ours.
    model_uncertain = raw.get("uncertain_fields")
    if isinstance(model_uncertain, list):
        for item in model_uncertain:
            if isinstance(item, str) and item not in uncertain:
                uncertain.append(item)
    model_notes = raw.get("review_notes")
    if isinstance(model_notes, list):
        notes.extend(n for n in model_notes if isinstance(n, str))

    record["uncertain_fields"] = sorted(set(uncertain))
    record["review_notes"] = notes
    # Require review if nothing usable came back, too many fields are shaky, or the
    # core identifying data (child name + every national id) is missing — in that
    # case only boilerplate was read and a human should look.
    filled = sum(1 for p in FIELD_KINDS if get_path(record, p) is not None)
    no_identity = (
        record["personal_and_other"]["child"]["name"] is None
        and record["ids"]["child_national_id"] is None
        and record["ids"]["father_national_id"] is None
        and record["ids"]["mother_national_id"] is None
    )
    record["review_required"] = (
        bool(raw.get("review_required")) or filled == 0 or len(uncertain) > 8 or no_identity
    )
    return record
