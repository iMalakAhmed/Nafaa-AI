"""Prompt for the birth-certificate extractor.

The template is fixed, so the prompt's whole job is to (1) describe exactly where
each value lives on an Egyptian birth certificate and (2) enforce hard null-discipline
so the model never invents a value it cannot actually read.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful Arabic document transcriber. Before you write JSON, you "
    "reason step by step: walk the fixed layout (child box, father box, mother box, "
    "registration box), decode every national id you can read and check it against "
    "the printed date of birth and gender, and say explicitly what you leave null "
    "because it is unreadable. You only copy text that is actually visible in the "
    "image. You never guess, never translate, and never invent plausible-looking "
    "values. Returning null is always better than guessing."
)

# Walks the model through the known fixed layout of an Egyptian صورة قيد ميلاد.
USER_INSTRUCTION = """\
This image is an Egyptian birth certificate (صورة قيد الميلاد). Every one of these has the
SAME fixed layout, so use the layout below to know exactly what each part means. Read ONLY
what is actually written.

HOW TO RESPOND (two parts, in this order):
1. REASONING — write a short "Reasoning:" section (plain text, NO curly braces) that:
   - Names which box is child / father / mother / registration.
   - For each national id you can read: state the 14 digits, decode digits 2-7 as the birth
     date, check digit 1 (2=1900s, 3=2000s), digits 8-9 as the governorate, and digit 13
     (odd=ذكر, even=أنثى). If the id contradicts the printed gender or DOB, or is covered
     by a stamp, say so and use null instead of guessing.
   - List any fields you leave null and why.
2. FINAL JSON — on its own line write exactly: FINAL JSON:
   then ONE valid JSON object (no markdown fences, no comments).

EXACT LAYOUT (top to bottom):
1. Header: title "صورة قيد الميلاد" and issuing authority "وزارة الداخلية / قطاع الأحوال المدنية".
2. ABOVE the boxes, on the line "الرقم القومي", is the CHILD's own 14-digit national ID. This
   number belongs to the child (the person the certificate is about) — it is NOT a parent's id.
   -> ids.child_national_id
3. FIRST box = the CHILD (the person themselves), under the heading "بيانات المولود":
   الاسم (child name), الجنسية (nationality), الديانة (religion), النوع (gender: ذكر/أنثى),
   محل الميلاد (place of birth), تاريخ الميلاد (date of birth).
   -> personal_and_other.child.*
4. The TWO boxes after the child box are the PARENTS. Use their printed Arabic headings to tell
   them apart (do NOT guess by position):
   - Box headed "بيانات الأب" = the FATHER: his name, الديانة, الجنسية, and his 14-digit الرقم القومي.
     -> personal_and_other.father.* and ids.father_national_id
   - Box headed "بيانات الأم" = the MOTHER: her name, الديانة, الجنسية, and her 14-digit الرقم القومي.
     -> personal_and_other.mother.* and ids.mother_national_id
   The father's name and the mother's name are DIFFERENT people — never copy the same name into
   both, and never put a parent's name into the child field.
5. LAST box = registration / office data: م. صحة (health office), س. مدني (civil registry),
   رقم القيد (registration number), ت. القيد (registration date), تاريخ الإصدار (issue date),
   and "رقم مسلسل" (serial number, usually near the barcode). -> birth_certificate.*

READING THE NUMBERS CAREFULLY (very important):
- Every national ID is exactly 14 digits and ALWAYS starts with 2 or 3 (2 = born 1900-1999,
  3 = born 2000-2099). If the first digit looks like a 2 or a 3, decide by the shape; it can
  only be one of those two.
- Be careful to distinguish similar digits: ٢ (2) vs ٣ (3), and read ٠ (0) correctly — a small
  oval/dot is a zero, do not skip it or turn it into another digit.
- Copy every digit you can see in order. If you genuinely cannot read the whole number, copy the
  digits you are sure of; if you cannot read it at all, use null. Never pad and never invent
  digits like 1234567890 or 0000000000.

OTHER HARD RULES:
- Copy Arabic text exactly as written; do not translate or paraphrase.
- gender must be exactly "ذكر" or "أنثى" (or null). religion is "مسلم"/"مسلمة" or
  "مسيحي"/"مسيحية" (or null). Nothing else for these two fields.
- If a label is printed but its value is blank/faded/covered, set the field to null and add its
  JSON path to "uncertain_fields". Do NOT fill a field just because the form has a slot for it.
- If most of the image is unreadable, set "review_required" to true and say why in "review_notes".

After your Reasoning section and the line "FINAL JSON:", return exactly this JSON structure
(use null for anything not clearly readable):
{
  "document_id": "<the id given to you, or null>",
  "source_files": ["<the path given to you>"],
  "page_count": 1,
  "birth_certificate": {
    "document_title_ar": null,
    "serial_number": null,
    "registration_number": null,
    "registration_date": null,
    "issue_date": null,
    "health_office": null,
    "civil_registry": null,
    "issuing_authority": null,
    "governorate_or_administration": null,
    "form_reference": null,
    "issuance_system_version": null,
    "barcode_or_machine_notes": null
  },
  "ids": {
    "child_national_id": null,
    "father_national_id": null,
    "mother_national_id": null,
    "other_ids": []
  },
  "personal_and_other": {
    "child": {
      "name": null, "gender": null, "religion": null,
      "nationality": null, "place_of_birth": null, "date_of_birth": null
    },
    "father": { "name": null, "religion": null, "nationality": null },
    "mother": { "name": null, "religion": null, "nationality": null },
    "misc_notes": null
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
