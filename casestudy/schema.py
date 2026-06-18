"""Case-study form record shape and field metadata."""

from __future__ import annotations

import copy
from typing import Any

KIND_TEXT = "text"
KIND_NATIONAL_ID = "national_id"
KIND_DATE = "date"
KIND_INT = "int"
KIND_BOOL = "bool"

FIELD_KINDS: dict[str, str] = {
    "document.document_type": KIND_TEXT,
    "document.form_version": KIND_TEXT,
    "document.page_side": KIND_TEXT,
    "document.case_number": KIND_TEXT,
    "document.research_date": KIND_DATE,
    "document.source_entity": KIND_TEXT,
    "office.directorate": KIND_TEXT,
    "office.administration": KIND_TEXT,
    "office.social_unit": KIND_TEXT,
    "office.researcher_name": KIND_TEXT,
    "office.requesting_entity": KIND_TEXT,
    "office.research_purpose": KIND_TEXT,
    "applicant.full_name": KIND_TEXT,
    "applicant.national_id": KIND_NATIONAL_ID,
    "applicant.insurance_number": KIND_TEXT,
    "applicant.age": KIND_INT,
    "applicant.gender": KIND_TEXT,
    "applicant.phone": KIND_TEXT,
    "applicant.address": KIND_TEXT,
    "applicant.governorate": KIND_TEXT,
    "applicant.district": KIND_TEXT,
    "applicant.marital_status": KIND_TEXT,
    "applicant.education_status": KIND_TEXT,
    "applicant.employment_status": KIND_TEXT,
    "applicant.health_status": KIND_TEXT,
    "housing.type": KIND_TEXT,
    "housing.ownership": KIND_TEXT,
    "housing.area": KIND_TEXT,
    "housing.rooms_count": KIND_INT,
    "housing.rent_value": KIND_TEXT,
    "housing.independence": KIND_TEXT,
    "housing.water_source": KIND_TEXT,
    "housing.electricity": KIND_TEXT,
    "housing.appliances_condition": KIND_TEXT,
    "assets.properties": KIND_TEXT,
    "social_assessment.family_social_status": KIND_TEXT,
    "social_assessment.health_notes": KIND_TEXT,
    "social_assessment.economic_status": KIND_TEXT,
    "social_assessment.family_needs": KIND_TEXT,
    "social_assessment.researcher_summary": KIND_TEXT,
    "signatures.researcher_name": KIND_TEXT,
    "signatures.unit_manager_name": KIND_TEXT,
    "signatures.stamp_present": KIND_BOOL,
}

SCALAR_FIELD_PATHS = list(FIELD_KINDS.keys())

FIELD_LABELS_AR: dict[str, list[str]] = {
    "document.form_version": ["نوع / إصدار النموذج"],
    "document.page_side": ["وجه / ظهر الصفحة"],
    "document.case_number": ["رقم البحث", "رقم الحالة", "رقم الملف"],
    "document.research_date": ["تاريخ البحث"],
    "document.source_entity": ["الجهة طالبة البحث", "مصدر البحث"],
    "office.directorate": ["مديرية التضامن الاجتماعي"],
    "office.administration": ["الإدارة الاجتماعية"],
    "office.social_unit": ["الوحدة الاجتماعية"],
    "office.researcher_name": ["اسم القائم بالبحث", "اسم الباحث"],
    "office.requesting_entity": ["الجهة طالبة البحث"],
    "office.research_purpose": ["الغرض من البحث"],
    "applicant.full_name": ["الاسم", "اسم الحالة", "اسم رب الأسرة", "اسم صاحب البحث"],
    "applicant.national_id": ["الرقم القومي"],
    "applicant.insurance_number": ["الرقم التأميني"],
    "applicant.age": ["السن"],
    "applicant.gender": ["النوع"],
    "applicant.phone": ["تليفون", "رقم الهاتف"],
    "applicant.address": ["العنوان", "محل الإقامة"],
    "applicant.governorate": ["محافظة"],
    "applicant.district": ["مركز", "قسم"],
    "applicant.marital_status": ["الحالة الاجتماعية"],
    "applicant.education_status": ["الحالة التعليمية"],
    "applicant.employment_status": ["حالة العمل", "الحالة العملية أو الوظيفية"],
    "applicant.health_status": ["الحالة الصحية"],
    "housing.type": ["المسكن", "نوع السكن"],
    "housing.ownership": ["نوع السكن", "تمليك", "إيجار"],
    "housing.area": ["المساحة"],
    "housing.rooms_count": ["عدد الغرف"],
    "housing.rent_value": ["القيمة الإيجارية الشهرية للسكن"],
    "housing.independence": ["استقلالية السكن", "مستقل بالأسرة", "مشترك"],
    "housing.water_source": ["مصدر المياه"],
    "housing.electricity": ["إنارة", "كهرباء"],
    "housing.appliances_condition": ["حالة الأثاث", "حالة السكن"],
    "assets.properties": ["حيازة العقارات والأملاك والأرض الزراعية"],
    "social_assessment.family_social_status": ["الحالة الاجتماعية للأسرة"],
    "social_assessment.health_notes": ["الحالة الصحية", "تذكر حالات الإعاقة والأمراض الأخرى"],
    "social_assessment.economic_status": ["الحالة الاقتصادية", "الدخل وأوجه الإنفاق"],
    "social_assessment.family_needs": ["احتياجات الأسرة"],
    "social_assessment.researcher_summary": ["رأي الباحث", "مدى الاستفادة من منظومة الضمان الاجتماعي"],
    "signatures.researcher_name": ["الباحث", "اسم الباحث"],
    "signatures.unit_manager_name": ["اسم رئيس الوحدة", "رئيس الوحدة"],
    "signatures.stamp_present": ["ختم الوحدة", "ختم الشعار"],
}

FAMILY_MEMBER_LABELS_AR: dict[str, list[str]] = {
    "row_index": ["م"],
    "name": ["الاسم"],
    "relationship": ["الصلة"],
    "age": ["السن"],
    "national_id": ["الرقم القومي"],
    "marital_status": ["الحالة الاجتماعية"],
    "education_status": ["الحالة التعليمية"],
    "employment_status": ["حالة العمل / الوظيفة"],
    "health_status": ["الحالة الصحية"],
    "notes": ["ملاحظات"],
}

_EMPTY_RECORD: dict[str, Any] = {
    "document_id": None,
    "source_files": [],
    "page_count": 1,
    "document": {
        "document_type": "case_study",
        "form_version": None,
        "page_side": None,
        "case_number": None,
        "research_date": None,
        "source_entity": None,
    },
    "office": {
        "directorate": None,
        "administration": None,
        "social_unit": None,
        "researcher_name": None,
        "requesting_entity": None,
        "research_purpose": None,
    },
    "applicant": {
        "full_name": None,
        "national_id": None,
        "insurance_number": None,
        "age": None,
        "gender": None,
        "phone": None,
        "address": None,
        "governorate": None,
        "district": None,
        "marital_status": None,
        "education_status": None,
        "employment_status": None,
        "health_status": None,
    },
    "family_members": [],
    "housing": {
        "type": None,
        "ownership": None,
        "area": None,
        "rooms_count": None,
        "rent_value": None,
        "independence": None,
        "water_source": None,
        "electricity": None,
        "appliances_condition": None,
    },
    "assets": {"properties": None},
    "social_assessment": {
        "family_social_status": None,
        "health_notes": None,
        "economic_status": None,
        "family_needs": None,
        "researcher_summary": None,
    },
    "checkbox_answers": [],
    "signatures": {
        "researcher_name": None,
        "unit_manager_name": None,
        "stamp_present": None,
    },
    "uncertain_fields": [],
    "review_required": False,
    "review_notes": [],
}

FAMILY_MEMBER_TEMPLATE: dict[str, Any] = {
    "row_index": None,
    "name": None,
    "relationship": None,
    "age": None,
    "national_id": None,
    "marital_status": None,
    "education_status": None,
    "employment_status": None,
    "health_status": None,
    "notes": None,
}

CHECKBOX_TEMPLATE: dict[str, Any] = {
    "section": None,
    "question": None,
    "answer": None,
    "checked": None,
    "confidence": None,
}


def empty_record(document_id: str | None = None, source_files: list[str] | None = None) -> dict[str, Any]:
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


def field_catalog_text() -> str:
    """Arabic label -> JSON path catalog for prompts and label review."""
    lines = ["Label-to-field catalog:"]
    for path in SCALAR_FIELD_PATHS:
        labels = " / ".join(FIELD_LABELS_AR.get(path, [path]))
        lines.append(f"- {labels} -> {path}")
    lines.append("Family table columns:")
    for key, labels in FAMILY_MEMBER_LABELS_AR.items():
        lines.append(f"- {' / '.join(labels)} -> family_members[].{key}")
    lines.append("Checked boxes:")
    lines.append("- Any checked option -> checkbox_answers[] with section, question, answer, checked=true, confidence")
    return "\n".join(lines)
