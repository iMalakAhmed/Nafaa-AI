import os
from typing import Dict, Any, List
from app.state import CaseState

# Import your native processing scripts
from app.services.fraud_detection import ai_generated_probability
from app.services.reverse_image import find_duplicates
from app.services.quality_gate_finalized import check_quality


def _normalize_risk_profile(
    ai_probability: float,
    duplicate: Dict[str, Any],
    quality: Dict[str, Any]
) -> Dict[str, Any]:
    duplicate_risk = 0.0
    if duplicate.get("duplicate_different_user"):
        duplicate_risk = 0.3
    elif duplicate.get("duplicate_same_user"):
        duplicate_risk = 0.2

    quality_score = quality.get("quality_score", 1.0)
    quality_risk = (1.0 - quality_score) * 0.2

    score = max(0.0, min(1.0, ai_probability * 0.5 + duplicate_risk + quality_risk))
    if score >= 0.75:
        level = "high"
    elif score >= 0.4:
        level = "medium"
    else:
        level = "low"

    signals = []
    if ai_probability > 0.7:
        signals.append("احتمال توليد صناعي مرتفع")
    if duplicate.get("duplicate_different_user"):
        signals.append("الصورة موجودة عند مستخدم آخر")
    elif duplicate.get("duplicate_same_user"):
        signals.append("الصورة مكررة لدى نفس المستخدم")
    if quality_score < 0.5:
        signals.append("جودة الصورة منخفضة أو احتمال تعديل")

    return {
        "risk_score": round(score, 3),
        "risk_level": level,
        "risk_signals": signals,
        "components": {
            "ai_probability": round(ai_probability, 3),
            "duplicate_risk": round(duplicate_risk, 3),
            "quality_risk": round(quality_risk, 3),
            "quality_score": round(quality_score, 3)
        }
    }


def build_evidence(image_path: str, user_id: str, intake_text: str = "") -> Dict[str, Any]:
    """Runs deterministic cold acquisition checks for a single image."""
    ai_prob = ai_generated_probability(image_path)
    duplicate = find_duplicates(image_path, user_id)
    quality = check_quality(image_path)

    risk_profile = _normalize_risk_profile(ai_prob, duplicate, quality)

    return {
        "image_path": image_path,
        "user_id": user_id,
        "ai_probability": ai_prob,
        "duplicate": duplicate,
        "quality": quality,
        "risk_profile": risk_profile,
        "cold_acquisition": {
            "intake_text_snapshot": intake_text,
            "image_path": image_path
        }
    }


def evidence_node(state: CaseState) -> dict:
    """
    Standard LangGraph node wrapper.
    Processes all images stored in the state, builds the evidence metrics, 
    and writes them cleanly to the shared Graph State.
    """
    image_paths = state.get("images") or []
    user_id = state.get("user_id") or "anonymous_user"
    intake_text = state.get("text") or ""

    image_evidence_reports = []

    # Run structural pipeline only if images are present
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                report = build_evidence(img_path, user_id, intake_text)
                image_evidence_reports.append(report)
            except Exception as e:
                image_evidence_reports.append({
                    "image_path": img_path,
                    "error": f"Failed to build evidence: {str(e)}",
                    "risk_profile": {
                        "risk_score": 1.0,
                        "risk_level": "high",
                        "risk_signals": ["cold acquisition failed"]
                    }
                })

    overall_risk_score = 0.0
    for report in image_evidence_reports:
        score = report.get("risk_profile", {}).get("risk_score", 0.0)
        overall_risk_score = max(overall_risk_score, score)

    overall_risk_level = "low"
    if overall_risk_score >= 0.75:
        overall_risk_level = "high"
    elif overall_risk_score >= 0.4:
        overall_risk_level = "medium"

    # Get existing evidence dictionary to avoid overwriting previous search node data
    current_evidence = state.get("evidence") or {}
    current_evidence["cold_acquisition"] = {
        "text_snapshot": intake_text,
        "image_count": len(image_paths),
        "images": image_paths,
        "image_analysis": image_evidence_reports,
        "overall_risk_score": round(overall_risk_score, 3),
        "overall_risk_level": overall_risk_level,
        "immutable_metadata": {
            "user_id": user_id,
            "source_text_present": bool(intake_text.strip()),
            "ocr_executed": bool(state.get("normalized_case", {}).get("ocr_executed", False))
        }
    }

    return {
        "evidence": current_evidence
    }