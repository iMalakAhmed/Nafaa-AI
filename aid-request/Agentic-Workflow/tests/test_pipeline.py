import importlib
import json
import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _ensure_mock_service(module_name: str, functions: Dict[str, Any]) -> types.ModuleType:
    if module_name in sys.modules:
        return sys.modules[module_name]

    module = types.ModuleType(module_name)
    for name, func in functions.items():
        setattr(module, name, func)
    sys.modules[module_name] = module
    return module


def _mock_transcribe(audio_path: str) -> str:
    return f"[mock transcript from {os.path.basename(audio_path)}]"


def _mock_extract_text_from_image(image_path: str) -> str:
    return f"[mock OCR text from {os.path.basename(image_path)}]"


def _mock_ai_generated_probability(image_path: str) -> float:
    return 0.25


def _mock_find_duplicates(image_path: str, user_id: str) -> Dict[str, Any]:
    return {"duplicate_different_user": False, "duplicate_same_user": False}


def _mock_check_quality(image_path: str) -> Dict[str, Any]:
    return {"quality_score": 0.9}


def _mock_answer_three_questions_batch(*args, **kwargs):
    questions = kwargs.get("questions") or []
    image_paths = kwargs.get("image_paths") or (args[0] if len(args) > 0 else [])
    return [
        {
            "image_path": image_paths[0] if image_paths else "",
            "results": [
                {"question": q, "answer": f"mock answer for {q}", "confidence": 0.95}
                for q in questions
            ]
        }
    ]


def _mock_answer_single_question(image_path: str, question: str, ocr_text: str = "", context: str = "") -> Dict[str, Any]:
    return {
        "question": question,
        "answer": "mocked answer",
        "confidence": 0.85,
        "reasoning": "Mocked reasoning for VQA followup."
    }


def _mock_llm_invoke(prompt: str):
    prompt_lower = prompt.strip().lower()
    
    # For reasoning node
    if "خبير دعم اتخاذ قرار" in prompt:
        if "عدد الصور المرفقة: 0" in prompt or "images: 0" in prompt.lower():
            return SimpleNamespace(content='{"next_step": "search", "action_details": {"target": "معلومات عن الحالة", "reasoning": "لا توجد صور، استخدم البحث للمعلومات الإضافية."}}' )
        else:
            return SimpleNamespace(content='{"next_step": "vqa", "action_details": {"target": "ما محتوى الصور المرفقة؟", "reasoning": "هناك صور بحاجة لتحليل."}}' )
    
    # For report node
    return SimpleNamespace(content='{"case_summary": "حالة طبية طارئة بحاجة لرعاية عاجلة.", "urgent_need": "دعم تكاليف علاج عاجل", "severity": "عالي", "recommended_action": "قبول الدعم المالي العاجل", "reasoning": "الحالة واضحة وتستحق الدعم.", "confidence": 0.92}')


def _install_pipeline_stubs():
    _ensure_mock_service(
        "app.services.stt",
        {"transcribe": _mock_transcribe}
    )
    _ensure_mock_service(
        "app.services.ocr",
        {"extract_text_from_image": _mock_extract_text_from_image}
    )
    _ensure_mock_service(
        "app.services.fraud_detection",
        {"ai_generated_probability": _mock_ai_generated_probability}
    )
    _ensure_mock_service(
        "app.services.reverse_image",
        {"find_duplicates": _mock_find_duplicates}
    )
    _ensure_mock_service(
        "app.services.quality_gate_finalized",
        {"check_quality": _mock_check_quality}
    )
    _ensure_mock_service(
        "app.services.vqa",
        {
            "answer_three_questions_batch": _mock_answer_three_questions_batch,
            "answer_single_question": _mock_answer_single_question
        }
    )
    _ensure_mock_service(
        "app.services.llm",
        {
            "llm": SimpleNamespace(invoke=_mock_llm_invoke),
            "llm_model": SimpleNamespace(invoke=_mock_llm_invoke, bind_tools=lambda tools: SimpleNamespace(invoke=_mock_llm_invoke))
        }
    )


def _create_dummy_file(path: str, content: str = "dummy content") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def run_pipeline_case(case_name: str, initial_state: Dict[str, Any]) -> None:
    print(f"\n=== {case_name} ===")
    _install_pipeline_stubs()

    intake_module = importlib.import_module("app.nodes.intake")
    evidence_module = importlib.import_module("app.nodes.evidence")
    vqa_module = importlib.import_module("app.nodes.vqa")
    reasoning_module = importlib.import_module("app.nodes.reasoning")
    report_module = importlib.import_module("app.nodes.report")

    state = dict(initial_state)
    state = {**state, **intake_module.intake_node(state)}
    state = {**state, **evidence_module.evidence_node(state)}
    
    # Simulate reasoning loop (only do one iteration for testing)
    reasoning_output = reasoning_module.reasoning_node(state)
    state = {**state, **reasoning_output}
    print(f"Reasoning decision: {state.get('reasoning', {}).get('next_step', 'unknown')}")
    
    # If reasoning suggests VQA, run it
    if state.get('reasoning', {}).get('next_step') == 'vqa':
        vqa_output = vqa_module.vqa_node(state)
        state = {**state, **vqa_output}
    
    # Simulate search results
    state["search_results"] = {
        "definitions": [],
        "medical_analysis": [{"case_or_drug": "دواء", "usage_causes": "علاج ألم", "source": "mock"}],
        "pricing": []
    }
    evidence = state.get("evidence", {})
    evidence["search"] = state["search_results"]
    state["evidence"] = evidence
    
    # Final report
    report_output = report_module.report_node(state)
    state = {**state, **report_output}

    # Print summary
    print("\nState after pipeline:")
    print(json.dumps({
        "reasoning_step": state.get("reasoning", {}).get("next_step"),
        "final_output": state.get("final_output")[:100] if state.get("final_output") else None
    }, ensure_ascii=False, indent=2))


def main():
    sample_audio = "data/v333.mp3"
    sample_image_1 = "data/img55.jpg"
    sample_image_2 = "data/prescription.jpg"
    _create_dummy_file(sample_audio)
    _create_dummy_file(sample_image_1)
    _create_dummy_file(sample_image_2)

    cases: List[Dict[str, Any]] = [
        {
            "name": "Full pipeline with text + image",
            "state": {
                "text": "الشكوى: مريض يحتاج دواء لعلاج ألم الحوض بعد حادث.",
                "voice_path": sample_audio,
                "images": [sample_image_1, sample_image_2],
                "user_id": "pipeline_user_1",
                "request_category": "medical",
                "evidence": {}
            }
        },
        {
            "name": "Pipeline with text only",
            "state": {
                "text": "أنا بحاجة إلى مساعدة مالية لتغطية فواتير علاج.",
                "voice_path": None,
                "images": [],
                "user_id": "pipeline_user_2",
                "request_category": "medical",
                "evidence": {}
            }
        }
    ]

    for case in cases:
        run_pipeline_case(case["name"], case["state"])


if __name__ == "__main__":
    main()
