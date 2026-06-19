import os
from typing import Dict, Any, List, Union
from app.state import CaseState
from app.services.vqa import answer_three_questions_batch


VQA_QUESTIONS = [
    "ما هو مضمون هذه الصورة؟",
    "هل تحتوي الصورة على نص واضح؟ إذا كان نعم، فما هو؟",
    "هل يوجد أي مؤشر على أن الصورة عبارة عن وثيقة طبية أو إيصال دعم مالي؟"
]


def vqa_node(state: CaseState) -> Dict[str, Any]:
    # --------------------------------------------------
    # 1. SAFE STATE EXTRACTION
    # --------------------------------------------------
    evidence = state.get("evidence") or {}
    inquiry_history = state.get("inquiry_history") or []

    image_paths = state.get("images") or []
    intake_text = state.get("text") or ""

    question = state.get("reasoning", {}).get("question_or_query")

    # Normalize question into a list safely
    if isinstance(question, list):
        questions = question
    elif isinstance(question, str) and question.strip():
        questions = [question]
    else:
        questions = VQA_QUESTIONS

    # --------------------------------------------------
    # 2. NO IMAGES CASE
    # --------------------------------------------------
    if not image_paths:
        evidence["vqa_analysis"] = {
            "questions": questions,
            "results": [],
            "metadata": {
                "image_count": 0,
                "execution_status": "skipped"
            }
        }

        return {
            "evidence": evidence,
            "inquiry_history": inquiry_history
        }

    # --------------------------------------------------
    # 3. PREP OCR CONTEXT
    # --------------------------------------------------
    ocr_texts = [intake_text] * len(image_paths)

    # --------------------------------------------------
    # 4. RUN VQA (SAFE)
    # --------------------------------------------------
    try:
        vqa_results = answer_three_questions_batch(
            image_paths=image_paths,
            ocr_texts=ocr_texts,
            description="تحليل صور الطلب والتناسق مع نص الشكوى",
            questions=questions
        )
        status = "completed"

    except Exception as e:
        vqa_results = [{"error": str(e)}]
        status = "failed"

    # --------------------------------------------------
    # 5. STORE RAW EVIDENCE
    # --------------------------------------------------
    evidence["vqa_analysis"] = {
        "questions": questions,
        "results": vqa_results,
        "metadata": {
            "image_count": len(image_paths),
            "execution_status": status
        }
    }

    # --------------------------------------------------
    # 6. BUILD SAFE SUMMARY FOR LLM (IMPORTANT)
    # --------------------------------------------------
    summary_results = []

    for res in vqa_results:
        if not isinstance(res, dict):
            continue

        for r in res.get("results", []):
            if not isinstance(r, dict):
                continue

            q = r.get("question", "")
            a = r.get("answer", "")

            summary_results.append(f"Q: {q} -> A: {a}")

    summary_text = "\n".join(summary_results)

    # --------------------------------------------------
    # 7. UPDATE INQUIRY HISTORY (SAFE + LIGHTWEIGHT)
    # --------------------------------------------------
    inquiry_history = inquiry_history + [{
        "type": "vqa",
        "target": (
            "multi_question_vqa"
            if len(questions) > 1
            else questions[0] if questions else "general_image_analysis"
        ),
        "content": summary_text
    }]

    # --------------------------------------------------
    # 8. RETURN STATE UPDATE
    # --------------------------------------------------
    return {
        "evidence": evidence,
        "inquiry_history": inquiry_history
    }