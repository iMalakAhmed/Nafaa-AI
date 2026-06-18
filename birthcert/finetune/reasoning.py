"""Generate a short, deterministic reasoning trace for each labeled certificate.

The goal is to teach the model to *think the way a careful human does* before it
emits JSON — but only with reasoning that is verifiable from a SINGLE image (the
family/sibling cross-checking used while labeling is not available at inference):

  1. Walk the fixed layout and say which box is child / father / mother.
  2. For every 14-digit national id, decode it and check it against the printed
     date of birth, the century digit, the governorate, and the gender parity
     of the 13th digit. A number that fails these checks (or is unreadable) is
     left null instead of guessed.
  3. Explicitly name anything left null.

The trace is built from the gold label, so it always agrees with the target JSON.
It is intentionally brace-free ('{' / '}' never appear) so the JSON object that
follows is still the first brace in the string and stays trivially parseable.
"""

from __future__ import annotations

from typing import Any

from ..schema import get_path
from ..validate import _GOVERNORATES, _valid_national_id, normalize_digits

# Marker that separates the reasoning from the JSON answer. Kept brace-free.
FINAL_MARKER = "FINAL JSON:"


def _digits(value: Any) -> str | None:
    if not value:
        return None
    d = "".join(ch for ch in normalize_digits(str(value)) if ch.isdigit())
    return d or None


def _id_check_line(who_en: str, expected_gender: str | None, value: Any, child_dob: str | None) -> str:
    """One reasoning line that decodes and sanity-checks a national id (or says why it's null)."""
    d = _digits(value)
    if d is None:
        return (
            f"- {who_en} national id: I cannot read all 14 digits clearly "
            f"(faded / covered by a stamp), so I leave it null instead of inventing digits."
        )
    if not _valid_national_id(d):
        return (
            f"- {who_en} national id reads {d}, but that is not a clean 14-digit id "
            f"(wrong length or impossible date), so I keep what I see but flag it for review."
        )
    century = "2000s" if d[0] == "3" else "1900s"
    yy, mm, dd = d[1:3], d[3:5], d[5:7]
    year = (2000 if d[0] == "3" else 1900) + int(yy)
    dob = f"{int(dd):02d}/{int(mm):02d}/{year}"
    gov = _GOVERNORATES.get(d[7:9], "an Egyptian governorate")
    parity = "odd" if int(d[12]) % 2 == 1 else "even"
    gender_from_id = "ذكر" if int(d[12]) % 2 == 1 else "أنثى"

    bits = [
        f"- {who_en} national id {d}: first digit {d[0]} → born in the {century}",
        f"digits 2-7 ({yy}{mm}{dd}) decode to {dob}",
    ]
    if who_en == "Child" and child_dob:
        if child_dob.replace("-", "/") == dob:
            bits.append("which matches the printed تاريخ الميلاد")
        else:
            bits.append(f"(the handwritten date reads {child_dob})")
    bits.append(f"governorate digits {d[7:9]} = {gov}")
    if expected_gender:
        ok = "matches" if gender_from_id == expected_gender else "does NOT match"
        bits.append(
            f"the 13th digit {d[12]} is {parity} → {gender_from_id}, which {ok} the النوع/role"
        )
    else:
        bits.append(f"the 13th digit {d[12]} is {parity} → {gender_from_id}")
    return ". ".join(bits) + "."


def build_reasoning(record: dict[str, Any]) -> str:
    """Return a brace-free reasoning trace consistent with `record`."""
    child_name = get_path(record, "personal_and_other.child.name")
    father_name = get_path(record, "personal_and_other.father.name")
    mother_name = get_path(record, "personal_and_other.mother.name")
    gender = get_path(record, "personal_and_other.child.gender")
    religion = get_path(record, "personal_and_other.child.religion")
    pob = get_path(record, "personal_and_other.child.place_of_birth")
    dob = get_path(record, "personal_and_other.child.date_of_birth")

    lines: list[str] = ["Reasoning:"]
    lines.append(
        "- Layout: this is the fixed Egyptian صورة قيد الميلاد template. The number on the "
        "الرقم القومي line above the boxes is the CHILD's own national id. The first box "
        "بيانات المولود is the child, the box headed بيانات الأب is the father, the box headed "
        "بيانات الأم is the mother, and the bottom box holds the registration/office data."
    )

    # Child demographics.
    if child_name:
        lines.append(f'- Child (first box): name is "{child_name}".')
    else:
        lines.append("- Child (first box): the name is not clearly readable, so I leave it null.")
    demo = []
    if gender:
        demo.append(f"النوع = {gender}")
    if religion:
        demo.append(f"الديانة = {religion}")
    if pob:
        demo.append(f"محل الميلاد = {pob}")
    if dob:
        demo.append(f"تاريخ الميلاد = {dob}")
    if demo:
        lines.append("- Child details: " + "؛ ".join(demo) + ".")

    # Parents — emphasise they are two different people and use the printed headings.
    fa = f'"{father_name}"' if father_name else "not readable (null)"
    mo = f'"{mother_name}"' if mother_name else "not readable (null)"
    lines.append(
        f"- Parents: the بيانات الأب box gives the father = {fa}; the بيانات الأم box gives the "
        f"mother = {mo}. These are two different people, so I never copy one name into both, "
        f"and never put a parent's name in the child field."
    )

    # National-id verification — the core transferable reasoning.
    lines.append(_id_check_line("Child", gender, get_path(record, "ids.child_national_id"), dob))
    lines.append(_id_check_line("Father", "ذكر", get_path(record, "ids.father_national_id"), None))
    lines.append(_id_check_line("Mother", "أنثى", get_path(record, "ids.mother_national_id"), None))

    # Honesty summary.
    uncertain = record.get("uncertain_fields") or []
    if uncertain:
        lines.append(
            "- Left null / flagged because I could not read them confidently: "
            + ", ".join(uncertain)
            + "."
        )
    else:
        lines.append("- Everything I recorded below I could actually read on the image.")
    lines.append(
        f"- review_required = {str(bool(record.get('review_required'))).lower()}."
    )
    return "\n".join(lines)


def build_target_with_reasoning(record_json: str, record: dict[str, Any]) -> str:
    """Reasoning trace, then the marker, then the gold JSON object."""
    return f"{build_reasoning(record)}\n\n{FINAL_MARKER}\n{record_json}"
