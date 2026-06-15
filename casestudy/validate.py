"""Validation and normalization for case-study form extraction."""

from __future__ import annotations

import re
from typing import Any

from .schema import FIELD_KINDS, KIND_BOOL, KIND_INT, KIND_NATIONAL_ID, empty_record, get_path, set_path

_DIGIT_MAP = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

_PLACEHOLDERS = {
    "", ".", "-", "--", "...", "null", "none", "n/a", "na",
    "غير محدد", "لا يوجد", "غير معروف", "غير متوفر", "مجهول",
}

_FRONT_HINTS = ("البيانات الأولية", "بيان بجميع أفراد الأسرة", "وصف المسكن الحالي")
_BACK_HINTS = ("الحالة الاجتماعية", "الحالة الصحية", "الحالة الاقتصادية", "احتياجات الأسرة", "توقيع الباحث")
_NEW_HINTS = ("نتائج بحث اجتماعي", "داخل", "خارج", "نعم", "لا", "ينطبق")
_OLD_HINTS = ("مديرية التضامن الاجتماعي", "قطاع الشئون الاجتماعية", "الوحدة الاجتماعية")


def _arabic_score(text: str) -> int:
    return sum(1 for ch in text if "\u0600" <= ch <= "\u06ff")


def repair_mojibake_text(value: str) -> str:
    if not any(ch in value for ch in ("\u00d8", "\u00d9", "\u00db", "\u00e2")):
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
        return {
            repair_mojibake_text(key) if isinstance(key, str) else key: repair_mojibake(item)
            for key, item in value.items()
        }
    return value


def normalize_digits(text: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = repair_mojibake(value)
    value = value.strip().strip("\u200e\u200f").strip()
    return value or None


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDERS


def _clean_national_id(value: str) -> tuple[str | None, bool]:
    digits = re.sub(r"\D", "", normalize_digits(value))
    if not digits:
        return None, False
    if len(digits) == 14 and digits[0] in "23":
        return digits, True
    if 10 <= len(digits) <= 16:
        return digits, False
    return None, False


def _clean_int(value: str) -> int | str | None:
    digits = re.sub(r"\D", "", normalize_digits(value))
    if digits and len(digits) <= 3:
        return int(digits)
    return value if value else None


def _clean_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _clean(value)
    if text is None:
        return None
    low = text.lower()
    if low in {"true", "yes", "1", "نعم", "موجود", "يوجد"}:
        return True
    if low in {"false", "no", "0", "لا", "غير موجود", "لا يوجد"}:
        return False
    return None


def _normalize_version(value: str | None, raw_text: str) -> str | None:
    if value:
        low = value.lower()
        if "old" in low or "قديم" in value:
            return "old"
        if "new" in low or "حديث" in value or "جديد" in value:
            return "new"
    new_hits = sum(1 for hint in _NEW_HINTS if hint in raw_text)
    old_hits = sum(1 for hint in _OLD_HINTS if hint in raw_text)
    if new_hits > old_hits:
        return "new"
    if old_hits > 0:
        return "old"
    return None


def _normalize_page_side(value: str | None, raw_text: str) -> str | None:
    if value:
        low = value.lower()
        if "front" in low or "وجه" in value or "أمام" in value:
            return "front"
        if "back" in low or "ظهر" in value or "خلف" in value:
            return "back"
    front_hits = sum(1 for hint in _FRONT_HINTS if hint in raw_text)
    back_hits = sum(1 for hint in _BACK_HINTS if hint in raw_text)
    if front_hits > back_hits:
        return "front"
    if back_hits > front_hits:
        return "back"
    return None


def _normalize_family_members(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {"row_index": item.get("row_index") or i}
        for key in (
            "name", "relationship", "marital_status", "education_status",
            "employment_status", "health_status", "notes",
        ):
            text = _clean(item.get(key))
            row[key] = None if text is None or _is_placeholder(text) else text
        age = _clean(item.get("age"))
        row["age"] = _clean_int(age) if age else None
        nat = _clean(item.get("national_id"))
        row["national_id"] = _clean_national_id(nat)[0] if nat else None
        if any(v is not None for k, v in row.items() if k != "row_index"):
            rows.append(row)
    return rows


def _normalize_checkboxes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        checked = _clean_bool(item.get("checked"))
        if checked is not True:
            continue
        out.append({
            "section": _clean(item.get("section")),
            "question": _clean(item.get("question")),
            "answer": _clean(item.get("answer")),
            "checked": True,
            "confidence": _clean(item.get("confidence")),
        })
    return out


def validate_record(raw: dict[str, Any], *, document_id: str, source_files: list[str]) -> dict[str, Any]:
    raw = repair_mojibake(raw)
    record = empty_record(document_id=document_id, source_files=source_files)
    uncertain: list[str] = []
    notes: list[str] = []
    raw_text = str(raw)

    for path, kind in FIELD_KINDS.items():
        raw_value = _clean(get_path(raw, path))
        if raw_value is None or _is_placeholder(raw_value):
            continue
        if kind == KIND_NATIONAL_ID:
            cleaned, confident = _clean_national_id(raw_value)
            if cleaned is None:
                uncertain.append(path)
                continue
            set_path(record, path, cleaned)
            if not confident:
                uncertain.append(path)
        elif kind == KIND_INT:
            set_path(record, path, _clean_int(raw_value))
        elif kind == KIND_BOOL:
            set_path(record, path, _clean_bool(raw_value))
        else:
            set_path(record, path, raw_value)

    record["document"]["document_type"] = "case_study"
    record["document"]["form_version"] = _normalize_version(
        get_path(record, "document.form_version"), raw_text
    )
    record["document"]["page_side"] = _normalize_page_side(
        get_path(record, "document.page_side"), raw_text
    )
    if record["document"]["form_version"] is None:
        uncertain.append("document.form_version")
    if record["document"]["page_side"] is None:
        uncertain.append("document.page_side")

    record["family_members"] = _normalize_family_members(raw.get("family_members"))
    record["checkbox_answers"] = _normalize_checkboxes(raw.get("checkbox_answers"))

    model_uncertain = raw.get("uncertain_fields")
    if isinstance(model_uncertain, list):
        for item in model_uncertain:
            if isinstance(item, str):
                uncertain.append(item)
    model_notes = raw.get("review_notes")
    if isinstance(model_notes, list):
        notes.extend(n for n in model_notes if isinstance(n, str))

    record["uncertain_fields"] = sorted(set(uncertain))
    record["review_notes"] = notes
    filled = sum(1 for path in FIELD_KINDS if get_path(record, path) is not None)
    record["review_required"] = bool(raw.get("review_required")) or filled < 4 or len(uncertain) > 10
    return record
