"""Generate reasoning traces for labeled printed social-insurance documents.

Teaches the model to:
  1. Identify the document sub-type from the title / template codes.
  2. Walk every printed label in order, naming the value.
  3. Decode and cross-check each national ID digit-by-digit.
  4. Name anything left null and why.

Traces are brace-free so the JSON that follows is still the first '{' in the string.
"""

from __future__ import annotations

from typing import Any

from ..schema import get_path
from ..validate import _GOVERNORATES, _valid_national_id, normalize_digits

FINAL_MARKER = "FINAL JSON:"


def _digits(value: Any) -> str | None:
    if not value:
        return None
    d = "".join(ch for ch in normalize_digits(str(value)) if ch.isdigit())
    return d or None


def _id_decode_line(label: str, value: Any, expected_gender: str | None = None) -> str:
    """One reasoning line that decodes and cross-checks a national ID."""
    d = _digits(value)
    if d is None:
        return (
            f"- {label}: not clearly readable (faded / stamp-covered), "
            f"leaving null instead of inventing digits."
        )
    if not _valid_national_id(d):
        return (
            f"- {label}: reads {d} — not a clean 14-digit id "
            f"(wrong length or impossible date), keeping as-is but flagging."
        )
    century = "2000s" if d[0] == "3" else "1900s"
    year  = (2000 if d[0] == "3" else 1900) + int(d[1:3])
    month = int(d[3:5])
    day   = int(d[5:7])
    dob   = f"{day:02d}/{month:02d}/{year}"
    gov   = _GOVERNORATES.get(d[7:9], "an Egyptian governorate")
    parity = "odd" if int(d[12]) % 2 == 1 else "even"
    gender_from_id = "ذكر" if int(d[12]) % 2 == 1 else "أنثى"

    parts = [
        f"- {label} {d}: first digit {d[0]} → born in the {century}",
        f"digits 2-7 ({d[1:3]}{d[3:5]}{d[5:7]}) decode to {dob}",
        f"governorate digits {d[7:9]} = {gov}",
    ]
    if expected_gender:
        match = "matches" if gender_from_id == expected_gender else "does NOT match"
        parts.append(f"13th digit {d[12]} is {parity} → {gender_from_id}, which {match} النوع")
    else:
        parts.append(f"13th digit {d[12]} is {parity} → {gender_from_id}")
    return ". ".join(parts) + "."


def _build_inquiry_reasoning(record: dict[str, Any]) -> list[str]:
    """Reasoning lines for a social_insurance_inquiry document."""
    lines: list[str] = []

    doc_type = get_path(record, "document_header.document_type") or "social_insurance_inquiry"
    template = get_path(record, "document_header.template_code")
    lines.append(
        f"- Document type: this is a social insurance inquiry "
        f"(استعلام بيانات مؤمن عليه). "
        f"Confirmed by template code {template or 'INDNO050/INDNS08'} top-left "
        f"and the PF10 == انتهاء التعامل marker."
    )

    # Office / header fields.
    region = get_path(record, "document_header.region_name")
    office = get_path(record, "document_header.office_name")
    ref    = get_path(record, "document_header.reference_number")
    if region or office:
        lines.append(
            f"- Header: منطقة = {region or 'null'}, مكتب = {office or 'null'}."
            + (f" Stamped reference: {ref}." if ref else "")
        )

    # Personal fields — walk in the order they appear on the form.
    nat_id   = get_path(record, "ids.national_id")
    ins_num  = get_path(record, "ids.insurance_number")
    name     = get_path(record, "personal.full_name_triple")
    family   = get_path(record, "personal.family_name")
    mother   = get_path(record, "personal.mother_name")
    gender   = get_path(record, "personal.gender")
    dob      = get_path(record, "personal.date_of_birth")
    gov_name = get_path(record, "personal.governorate_name")
    dist     = get_path(record, "personal.district_name")
    sector   = get_path(record, "personal.sector_law")

    lines.append(f"- الرقم القومي = {nat_id or 'null (not readable)'}.")
    lines.append(f"- الرقم التأميني = {ins_num or 'null (not readable)'}.")
    lines.append(f"- الاسم ثلاثي = {name or 'null'}.")
    lines.append(f"- اسم العائلة = {family or 'null'}.")
    lines.append(f"- اسم الوالدة = {mother or 'null'}.")
    lines.append(
        f"- النوع = {gender or 'null'}؛ تاريخ الميلاد = {dob or 'null'}؛ "
        f"محافظة = {gov_name or 'null'}؛ قسم/مركز = {dist or 'null'}؛ "
        f"قانون/قطاع = {sector or 'null'}."
    )

    # National ID cross-check.
    lines.append(_id_decode_line("National id", nat_id, gender))

    # Insurance status.
    status      = get_path(record, "social_insurance.insurance_status")
    status_note = get_path(record, "social_insurance.insurance_status_note")
    if status_note:
        lines.append(
            f"- ** section reads: \"{status_note}\". "
            f"Normalised insurance_status = \"{status or 'null'}\"."
        )
    else:
        lines.append("- ** section: not readable or blank → insurance_status null.")

    # Contact info.
    website  = get_path(record, "social_insurance.website")
    cs_num   = get_path(record, "social_insurance.customer_service_number")
    lines.append(
        f"- Bottom: website = {website or 'null'}, "
        f"customer service = {cs_num or 'null'}."
    )
    return lines


def _build_payroll_reasoning(record: dict[str, Any]) -> list[str]:
    """Reasoning lines for a payroll_slip document."""
    lines: list[str] = []

    template = get_path(record, "document_header.template_code")
    lines.append(
        f"- Document type: this is a periodic payroll / pension slip "
        f"(بيانات الصرف الدوري). "
        f"Confirmed by template code {template or 'BENINPBO/BEINSBO'} top-left "
        f"and the presence of PF06 / PF03 form codes."
    )

    region = get_path(record, "document_header.region_name")
    office = get_path(record, "document_header.office_name")
    unit   = get_path(record, "document_header.unit")
    ref    = get_path(record, "document_header.reference_number")
    lines.append(
        f"- Header: منطقة = {region or 'null'}, مكتب = {office or 'null'}, "
        f"وحدة = {unit or 'null'}."
        + (f" Stamped reference: {ref}." if ref else "")
    )

    # Pension holder (primary person).
    name    = get_path(record, "personal.full_name_triple")
    nat_id  = get_path(record, "ids.national_id")
    ins_num = get_path(record, "ids.insurance_number")
    gender  = get_path(record, "personal.gender")
    lines.append(
        f"- رقم تأميني صاحب المعاش = {ins_num or 'null'} (pension holder name printed after 'على': {name or 'null'})."
    )
    lines.append(f"- رقم قومي صاحب المعاش = {nat_id or 'null'}.")

    # Paying agent.
    agent_ins = get_path(record, "ids.paying_agent_insurance_number")
    agent_id  = get_path(record, "ids.paying_agent_national_id")
    agent_name = get_path(record, "payroll.paying_agent.name") if get_path(record, "payroll") else None
    lines.append(
        f"- رقم تأميني القائم بالصرف = {agent_ins or 'null'} (name: {agent_name or 'null'})."
    )
    lines.append(f"- رقم قومي القائم بالصرف = {agent_id or 'null'}.")

    # National ID cross-checks.
    lines.append(_id_decode_line("Pension holder national id", nat_id, gender))
    lines.append(_id_decode_line("Paying agent national id", agent_id))

    # Payroll fields.
    payroll = get_path(record, "payroll") or {}
    lines.append(
        f"- قطاع المعاش = {payroll.get('pension_sector') or 'null'}؛ "
        f"بداية الصرف = {payroll.get('payment_start_date') or 'null'}؛ "
        f"جهة الصرف = {payroll.get('disbursement_bank') or 'null'}."
    )
    lines.append(
        f"- إجمالي الاستحقاق = {payroll.get('total_entitlement') or 'null'}؛ "
        f"إجمالي الاستقطاع = {payroll.get('total_deductions') or 'null'}؛ "
        f"صافي المعاش = {payroll.get('net_pension') or 'null'}."
    )
    codes = payroll.get("form_codes") or []
    lines.append(f"- Form codes at bottom: {', '.join(codes) if codes else 'none visible'}.")
    return lines


def build_reasoning(record: dict[str, Any]) -> str:
    """Return a brace-free reasoning trace consistent with `record`."""
    doc_type = (get_path(record, "document_header.document_type") or "").lower()

    lines: list[str] = ["Reasoning:"]
    if "payroll" in doc_type:
        lines.extend(_build_payroll_reasoning(record))
    else:
        lines.extend(_build_inquiry_reasoning(record))

    # Honesty summary.
    uncertain = record.get("uncertain_fields") or []
    if uncertain:
        lines.append(
            "- Left null / flagged because not clearly readable: "
            + ", ".join(uncertain) + "."
        )
    else:
        lines.append("- Everything recorded below was actually readable in the image.")
    lines.append(f"- review_required = {str(bool(record.get('review_required'))).lower()}.")
    return "\n".join(lines)


def build_target_with_reasoning(record_json: str, record: dict[str, Any]) -> str:
    return f"{build_reasoning(record)}\n\n{FINAL_MARKER}\n{record_json}"
