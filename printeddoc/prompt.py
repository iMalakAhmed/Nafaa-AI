"""Prompt for the printed social-insurance document extractor.

Two sub-types share the same fixed label/value layout:
  1. استعلام بيانات مؤمن عليه  (social insurance inquiry)
  2. بيانات الصرف الدوري        (periodic payroll / pension slip)

The prompt instructs the model to identify the type first, then fill only
the section that applies and set the other to null.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful Arabic document transcriber. Before you write JSON, you "
    "reason step by step: identify the document type from the title and template "
    "codes, walk every labelled field in order, read national IDs digit by digit "
    "and cross-check them against the printed birth date and gender, and say "
    "explicitly what you leave null because it is unreadable or not present on this "
    "sub-type. You only copy text that is actually visible in the image. You never "
    "guess, never translate, and never invent plausible-looking values. "
    "Returning null is always better than guessing."
)

USER_INSTRUCTION = """\
This image is an Egyptian printed social-insurance document issued by NOSI
(الهيكلة القومية للتأمين الاجتماعي) or the General Social Insurance Authority.
There are exactly TWO sub-types — identify which one this is FIRST, then extract
all fields systematically.

HOW TO RESPOND (two parts, in this order):
1. REASONING — write a short "Reasoning:" section (plain text, NO curly braces) that:
   - States the document sub-type you identified and which template codes confirm it.
   - Walks every printed label in order, stating the value next to it.
   - For each national ID you can read: state the digits, decode digits 2-7 as the
     birth date, digit 1 (2=1900s / 3=2000s), digits 8-9 as governorate code,
     digit 13 (odd=ذكر, even=أنثى). Flag contradictions or stamp-covered digits.
   - Lists any field you leave null and why (unreadable, covered, or wrong sub-type).
2. FINAL JSON — on its own line write exactly: FINAL JSON:
   then ONE valid JSON object (no markdown fences, no comments).

════════════════════════════════════════
SUB-TYPE 1 — SOCIAL INSURANCE INQUIRY
(استعلام بيانات مؤمن عليه بالملف التأميني)
════════════════════════════════════════
IDENTIFIERS: template codes INDNO050 / INDNS08 top-left; "PF10 == انتهاء التعامل"
             in the top-right corner; title includes "استعلام بيانات مؤمن عليه".
             → document_type = "social_insurance_inquiry"
             → payroll section = null

LAYOUT (right side, top to bottom):
  منطقة  → document_header.region_code + region_name   (e.g. 07 المنوفية)
  مكتب   → document_header.office_code + office_name   (e.g. 13 شبين الكوم خان)
  الرقم القومي    → ids.national_id            (14-digit national ID)
  الرقم التأميني  → ids.insurance_number
  الاسم ثلاثي     → personal.full_name_triple  (three-part given name)
  اسم العائلة     → personal.family_name
  اسم الوالدة     → personal.mother_name
  النوع           → personal.gender            (ذكر or أنثى)
  تاريخ الميلاد   → personal.date_of_birth
  محافظة          → personal.governorate_code + governorate_name
  قسم / مركز      → personal.district_code + district_name
  قانون / قطاع    → personal.sector_law        (often blank → null)

** section (insurance status note printed below the personal data):
  Read the full Arabic text of this note exactly as printed.
  → social_insurance.insurance_status_note   (full text)
  → social_insurance.insurance_status        (normalised to ONE of these three exact strings):
      "مؤمن عليه حاليا"           if the person is currently insured
      "غير مؤمن عليه حاليا"       if the person is not currently insured
      "لا توجد بيانات تأمينية"     if no insurance data exists for this person

BOTTOM section:
  المركز الرئيسي  → social_insurance.main_center_address
  الموقع الالكتروني → social_insurance.website
  خدمة العملاء    → social_insurance.customer_service_number
  رقم الفاكس      → social_insurance.fax_number

Stamped reference number on the left margin → document_header.reference_number

════════════════════════════════════════
SUB-TYPE 2 — PERIODIC PAYROLL / PENSION SLIP
(بيانات الصرف الدوري للقائم بالصرف)
════════════════════════════════════════
IDENTIFIERS: template codes BENINPBO / BEINSBO top-left; title includes
             "بيانات الصرف الدوري"; both PF06 and PF03 codes appear at the bottom.
             → document_type = "payroll_slip"
             → social_insurance section = null

LAYOUT (top section):
  منطقة  → document_header.region_code + region_name
  مكتب   → document_header.office_code + office_name
  وحدة   → document_header.unit

  رقم تأميني صاحب المعاش  → ids.insurance_number
                             (the name printed after "على" belongs to the pension holder
                              → personal.full_name_triple)
  رقم قومي صاحب المعاش    → ids.national_id           (14-digit ID of pension holder)
  رقم تأميني القائم بالصرف → ids.paying_agent_insurance_number
                             (the name after the number → payroll.paying_agent.name)
  رقم قومي القائم بالصرف   → ids.paying_agent_national_id

MIDDLE section:
  قطاع المعاش            → payroll.pension_sector
  منطقة / مكتب المتابعة  → payroll.follow_up_office
  بداية الصرف الدوري     → payroll.payment_start_date   (e.g. 1-2-21)
  جهة الصرف              → payroll.disbursement_bank
  نوع المدفوع             → payroll.payment_type
  رقم الحساب الجاري       → payroll.current_account_number

FINANCIAL section (right side):
  إجمالي الاستحقاق  → payroll.total_entitlement   (copy the full printed value)
  إجمالي الاستقطاع  → payroll.total_deductions
  صافي المعاش       → payroll.net_pension

FOOTER form codes (PF10, PF06, PF03, etc.) → payroll.form_codes  (list all visible codes)

For payroll slips: personal.family_name and personal.mother_name are not printed
→ leave both null. personal.gender, date_of_birth, governorate, district come
from the pension holder's national ID cross-check, not from printed labels.

════════════════════════════════════════
READING IDs CAREFULLY:
- Egyptian national ID = exactly 14 digits, always starts with 2 (born 1900-1999)
  or 3 (born 2000-2099). Never starts with any other digit.
- If a stamp covers some digits, copy what you can see; if fewer than 10 digits are
  clear, set the field to null rather than guessing the rest.
- Insurance numbers follow no fixed length — copy them exactly as printed.

HARD RULES (same as always):
- Copy Arabic text exactly; do not translate or paraphrase.
- gender must be exactly "ذكر" or "أنثى" (or null). Nothing else.
- If a label is printed but its value is blank, faded, or covered → null, and add
  the JSON path to "uncertain_fields".
- If the image is mostly unreadable → set review_required = true and explain in
  review_notes.

After your Reasoning section and the line "FINAL JSON:", return exactly this structure
(null for anything not clearly readable; set the wrong sub-type's section to null):
{
  "document_id": "<the id given to you, or null>",
  "source_files": ["<the path given to you>"],
  "page_count": 1,
  "document_header": {
    "document_type": null,
    "document_title_ar": null,
    "issuing_authority_ar": null,
    "template_code": null,
    "print_datetime": null,
    "region_code": null,
    "region_name": null,
    "office_code": null,
    "office_name": null,
    "unit": null,
    "reference_number": null
  },
  "ids": {
    "national_id": null,
    "insurance_number": null,
    "paying_agent_insurance_number": null,
    "paying_agent_national_id": null
  },
  "personal": {
    "full_name_triple": null,
    "family_name": null,
    "mother_name": null,
    "gender": null,
    "date_of_birth": null,
    "governorate_code": null,
    "governorate_name": null,
    "district_code": null,
    "district_name": null,
    "sector_law": null
  },
  "social_insurance": {
    "insurance_status": null,
    "insurance_status_note": null,
    "main_center_address": null,
    "website": null,
    "customer_service_number": null,
    "fax_number": null
  },
  "payroll": {
    "pension_sector": null,
    "follow_up_office": null,
    "payment_start_date": null,
    "disbursement_bank": null,
    "payment_type": null,
    "current_account_number": null,
    "total_entitlement": null,
    "total_deductions": null,
    "net_pension": null,
    "paying_agent": {"name": null},
    "form_codes": []
  },
  "uncertain_fields": [],
  "review_required": false,
  "review_notes": []
}
"""


def build_user_instruction(document_id: str, source_file: str) -> str:
    return (
        f'document_id = "{document_id}"\n'
        f'source path = "{source_file}"\n\n'
        + USER_INSTRUCTION
    )
