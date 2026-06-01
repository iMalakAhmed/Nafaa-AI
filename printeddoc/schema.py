"""Printed social-insurance document record shape and field metadata.

Covers two sub-types issued by NOSI / صندوق العاملين:
  - social_insurance_inquiry  (استعلام بيانات مؤمن عليه)
  - payroll_slip               (بيانات الصرف الدوري)

Matches schemas/printed_document.schema.json.
Everything downstream (prompt, validation, evaluation) is driven from here.
"""

from __future__ import annotations

import copy
from typing import Any

KIND_NATIONAL_ID = "national_id"
KIND_DATE        = "date"
KIND_GENDER      = "gender"
KIND_TEXT        = "text"

# Dotted path -> validation kind for every scored scalar field.
FIELD_KINDS: dict[str, str] = {
    # document_header
    "document_header.document_type":        KIND_TEXT,
    "document_header.document_title_ar":    KIND_TEXT,
    "document_header.issuing_authority_ar": KIND_TEXT,
    "document_header.template_code":        KIND_TEXT,
    "document_header.print_datetime":       KIND_TEXT,
    "document_header.region_code":          KIND_TEXT,
    "document_header.region_name":          KIND_TEXT,
    "document_header.office_code":          KIND_TEXT,
    "document_header.office_name":          KIND_TEXT,
    "document_header.unit":                 KIND_TEXT,
    "document_header.reference_number":     KIND_TEXT,
    # ids
    "ids.national_id":                      KIND_NATIONAL_ID,
    "ids.insurance_number":                 KIND_TEXT,
    "ids.paying_agent_insurance_number":    KIND_TEXT,
    "ids.paying_agent_national_id":         KIND_NATIONAL_ID,
    # personal
    "personal.full_name_triple":            KIND_TEXT,
    "personal.family_name":                 KIND_TEXT,
    "personal.mother_name":                 KIND_TEXT,
    "personal.gender":                      KIND_GENDER,
    "personal.date_of_birth":              KIND_DATE,
    "personal.governorate_code":            KIND_TEXT,
    "personal.governorate_name":            KIND_TEXT,
    "personal.district_code":              KIND_TEXT,
    "personal.district_name":              KIND_TEXT,
    "personal.sector_law":                  KIND_TEXT,
    # social_insurance (null for payroll_slip)
    "social_insurance.insurance_status":        KIND_TEXT,
    "social_insurance.insurance_status_note":   KIND_TEXT,
    "social_insurance.main_center_address":     KIND_TEXT,
    "social_insurance.website":                 KIND_TEXT,
    "social_insurance.customer_service_number": KIND_TEXT,
    "social_insurance.fax_number":              KIND_TEXT,
    # payroll (null for social_insurance_inquiry)
    "payroll.pension_sector":           KIND_TEXT,
    "payroll.follow_up_office":         KIND_TEXT,
    "payroll.payment_start_date":       KIND_TEXT,
    "payroll.disbursement_bank":        KIND_TEXT,
    "payroll.payment_type":             KIND_TEXT,
    "payroll.current_account_number":   KIND_TEXT,
    "payroll.total_entitlement":        KIND_TEXT,
    "payroll.total_deductions":         KIND_TEXT,
    "payroll.net_pension":              KIND_TEXT,
    "payroll.paying_agent.name":        KIND_TEXT,
}

SCALAR_FIELD_PATHS: list[str] = list(FIELD_KINDS.keys())

_EMPTY_RECORD: dict[str, Any] = {
    "document_id":   None,
    "source_files":  [],
    "page_count":    1,
    "document_header": {
        "document_type":        None,
        "document_title_ar":    None,
        "issuing_authority_ar": None,
        "template_code":        None,
        "print_datetime":       None,
        "region_code":          None,
        "region_name":          None,
        "office_code":          None,
        "office_name":          None,
        "unit":                 None,
        "reference_number":     None,
    },
    "ids": {
        "national_id":                   None,
        "insurance_number":              None,
        "paying_agent_insurance_number": None,
        "paying_agent_national_id":      None,
    },
    "personal": {
        "full_name_triple":  None,
        "family_name":       None,
        "mother_name":       None,
        "gender":            None,
        "date_of_birth":     None,
        "governorate_code":  None,
        "governorate_name":  None,
        "district_code":     None,
        "district_name":     None,
        "sector_law":        None,
    },
    "social_insurance": {
        "insurance_status":        None,
        "insurance_status_note":   None,
        "main_center_address":     None,
        "website":                 None,
        "customer_service_number": None,
        "fax_number":              None,
    },
    "payroll": {
        "pension_sector":         None,
        "follow_up_office":       None,
        "payment_start_date":     None,
        "disbursement_bank":      None,
        "payment_type":           None,
        "current_account_number": None,
        "total_entitlement":      None,
        "total_deductions":       None,
        "net_pension":            None,
        "paying_agent":           {"name": None},
        "form_codes":             [],
    },
    "uncertain_fields": [],
    "review_required":  False,
    "review_notes":     [],
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
