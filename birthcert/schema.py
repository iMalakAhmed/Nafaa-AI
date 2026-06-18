"""The fixed birth-certificate record shape and field metadata.

Everything downstream (prompt, validation, evaluation) is driven from here so the
JSON shape stays consistent in one place. Matches schemas/birth_certificate.schema.json.
"""

from __future__ import annotations

import copy
from typing import Any

# Field "kinds" drive validation. Anything not listed is treated as free Arabic text.
KIND_NATIONAL_ID = "national_id"
KIND_DATE = "date"
KIND_GENDER = "gender"
KIND_RELIGION = "religion"
KIND_TEXT = "text"

# Dotted path -> validation kind. Paths address into the nested record below.
FIELD_KINDS: dict[str, str] = {
    "birth_certificate.document_title_ar": KIND_TEXT,
    "birth_certificate.serial_number": KIND_TEXT,
    "birth_certificate.registration_number": KIND_TEXT,
    "birth_certificate.registration_date": KIND_DATE,
    "birth_certificate.issue_date": KIND_DATE,
    "birth_certificate.health_office": KIND_TEXT,
    "birth_certificate.civil_registry": KIND_TEXT,
    "birth_certificate.issuing_authority": KIND_TEXT,
    "birth_certificate.governorate_or_administration": KIND_TEXT,
    "birth_certificate.form_reference": KIND_TEXT,
    "birth_certificate.issuance_system_version": KIND_TEXT,
    "birth_certificate.barcode_or_machine_notes": KIND_TEXT,
    "ids.child_national_id": KIND_NATIONAL_ID,
    "ids.father_national_id": KIND_NATIONAL_ID,
    "ids.mother_national_id": KIND_NATIONAL_ID,
    "personal_and_other.child.name": KIND_TEXT,
    "personal_and_other.child.gender": KIND_GENDER,
    "personal_and_other.child.religion": KIND_RELIGION,
    "personal_and_other.child.nationality": KIND_TEXT,
    "personal_and_other.child.place_of_birth": KIND_TEXT,
    "personal_and_other.child.date_of_birth": KIND_DATE,
    "personal_and_other.father.name": KIND_TEXT,
    "personal_and_other.father.religion": KIND_RELIGION,
    "personal_and_other.father.nationality": KIND_TEXT,
    "personal_and_other.mother.name": KIND_TEXT,
    "personal_and_other.mother.religion": KIND_RELIGION,
    "personal_and_other.mother.nationality": KIND_TEXT,
    "personal_and_other.misc_notes": KIND_TEXT,
}

# Scalar field paths, in a stable order, used for labeling templates and scoring.
SCALAR_FIELD_PATHS: list[str] = list(FIELD_KINDS.keys())

_EMPTY_RECORD: dict[str, Any] = {
    "document_id": None,
    "source_files": [],
    "page_count": 1,
    "birth_certificate": {
        "document_title_ar": None,
        "serial_number": None,
        "registration_number": None,
        "registration_date": None,
        "issue_date": None,
        "health_office": None,
        "civil_registry": None,
        "issuing_authority": None,
        "governorate_or_administration": None,
        "form_reference": None,
        "issuance_system_version": None,
        "barcode_or_machine_notes": None,
    },
    "ids": {
        "child_national_id": None,
        "father_national_id": None,
        "mother_national_id": None,
        "other_ids": [],
    },
    "personal_and_other": {
        "child": {
            "name": None,
            "gender": None,
            "religion": None,
            "nationality": None,
            "place_of_birth": None,
            "date_of_birth": None,
        },
        "father": {"name": None, "religion": None, "nationality": None},
        "mother": {"name": None, "religion": None, "nationality": None},
        "misc_notes": None,
    },
    "uncertain_fields": [],
    "review_required": False,
    "review_notes": [],
}


def empty_record(document_id: str | None = None, source_files: list[str] | None = None) -> dict[str, Any]:
    """A fresh, fully-null record matching the schema."""
    record = copy.deepcopy(_EMPTY_RECORD)
    if document_id is not None:
        record["document_id"] = document_id
    if source_files is not None:
        record["source_files"] = list(source_files)
        record["page_count"] = max(1, len(source_files))
    return record


def get_path(record: dict[str, Any], path: str) -> Any:
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_path(record: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = record
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value
