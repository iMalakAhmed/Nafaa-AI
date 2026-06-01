"""Prompt for Arabic case-study form extraction."""

from __future__ import annotations

import json

from .schema import empty_record

SYSTEM_PROMPT = (
    "You are a careful Arabic social-case-study form transcriber. You read printed labels, "
    "handwritten values, family tables, stamps, and checkboxes. You reason step by step before "
    "JSON: classify the form version and page side, walk sections top-to-bottom, transcribe "
    "handwriting exactly, map checked boxes to their printed options, and say what is unreadable. "
    "Never guess missing handwriting. Return null instead of inventing values."
)


def build_user_instruction(document_id: str, source: str) -> str:
    template = json.dumps(empty_record(document_id, [source]), ensure_ascii=False, indent=2)
    return f"""document_id = "{document_id}"
source path = "{source}"

This is an Egyptian Arabic social case-study form. It may be one of two layouts:
- old handwritten-heavy layout: mostly dotted lines, free handwriting, family table, front/back pages.
- new checkbox-heavy layout: dense printed questions, many square checkboxes, family table, front/back pages.

HOW TO RESPOND, in this exact order:
1. Reasoning:
   - Classify form_version as "old", "new", or null.
   - Classify page_side as "front", "back", or null.
   - Walk the page from top to bottom. For every visible printed label, state the value next to it.
   - For handwritten values, copy the Arabic exactly. If a word is not clear, leave the field null and add the JSON path to uncertain_fields.
   - For the family table, output one family_members item per readable row. Preserve row order.
   - For checkboxes, inspect the box itself: checked means there is a clear tick/mark inside or crossing the box. Output checked options in checkbox_answers with section/question/answer.
   - For national IDs, copy only clear digits. If 14 digits are readable, decode age/date/gender only as validation evidence; do not invent missing digits.
   - List fields left null because they are blank, cut off, shadowed, stamped, or belong to the other page side.
2. FINAL JSON:
   then exactly one valid JSON object matching the schema below. No markdown fences.

FIELD GUIDANCE:
- document.form_version: "old" or "new".
- document.page_side: "front" or "back".
- Front pages usually contain office/applicant/family/housing fields.
- Back pages usually contain housing continuation, assets, social/health/economic assessment, needs, checkbox blocks, signatures/stamp.
- checkbox_answers is a list of only checked answers, not every unchecked box.
- family_members rows use keys: row_index, name, relationship, age, national_id, marital_status, education_status, employment_status, health_status, notes.
- signatures.stamp_present is true only if a visible official stamp appears.

HARD RULES:
- Arabic output stays Arabic; do not translate.
- Do not summarize names or handwritten notes.
- Use null for unreadable values.
- If the page is too blurry or cropped to trust, set review_required = true.
- Add a short review_note for major issues: blur, crop, shadow, stamp covering text, uncertain checkbox.

Schema:
{template}
"""

