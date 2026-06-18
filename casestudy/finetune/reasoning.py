"""Generate compact reasoning traces for labeled case-study pages."""

from __future__ import annotations

from typing import Any

from ..schema import get_path

FINAL_MARKER = "FINAL JSON:"


def _line(label: str, value: Any) -> str:
    return f"- {label}: {value if value not in (None, '', []) else 'null'}."


def build_reasoning(record: dict[str, Any]) -> str:
    lines = ["Reasoning:"]
    version = get_path(record, "document.form_version")
    side = get_path(record, "document.page_side")
    lines.append(f"- Form classification: version = {version or 'null'}, page_side = {side or 'null'}.")
    lines.append(_line("Case/research number", get_path(record, "document.case_number")))
    lines.append(_line("Office directorate", get_path(record, "office.directorate")))
    lines.append(_line("Administration", get_path(record, "office.administration")))
    lines.append(_line("Social unit", get_path(record, "office.social_unit")))
    lines.append(_line("Researcher", get_path(record, "office.researcher_name")))
    lines.append(_line("Applicant", get_path(record, "applicant.full_name")))
    lines.append(_line("Applicant national id", get_path(record, "applicant.national_id")))
    lines.append(_line("Applicant insurance number", get_path(record, "applicant.insurance_number")))
    lines.append(_line("Applicant address", get_path(record, "applicant.address")))
    lines.append(_line("Housing", get_path(record, "housing.type")))
    lines.append(_line("Family social status", get_path(record, "social_assessment.family_social_status")))
    lines.append(_line("Health notes", get_path(record, "social_assessment.health_notes")))
    lines.append(_line("Economic status", get_path(record, "social_assessment.economic_status")))
    lines.append(_line("Family needs", get_path(record, "social_assessment.family_needs")))

    rows = record.get("family_members") or []
    if rows:
        lines.append(f"- Family table: {len(rows)} readable row(s), copied in row order.")
    else:
        lines.append("- Family table: no readable rows on this page or not present.")
    boxes = record.get("checkbox_answers") or []
    if boxes:
        lines.append(f"- Checkboxes: {len(boxes)} checked option(s) recorded; unchecked options omitted.")
    else:
        lines.append("- Checkboxes: no checked options visible or this page is handwriting-only.")

    uncertain = record.get("uncertain_fields") or []
    if uncertain:
        lines.append("- Left null / flagged because unclear: " + ", ".join(uncertain) + ".")
    else:
        lines.append("- Every non-null field below was visible enough to transcribe.")
    lines.append(f"- review_required = {str(bool(record.get('review_required'))).lower()}.")
    return "\n".join(lines)


def build_target_with_reasoning(record_json: str, record: dict[str, Any]) -> str:
    return f"{build_reasoning(record)}\n\n{FINAL_MARKER}\n{record_json}"

