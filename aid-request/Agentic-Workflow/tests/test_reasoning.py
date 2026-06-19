import importlib
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


def _mock_llm_invoke(prompt: str):
    # Return the new format with action_details
    if "يستخدم VQA" in prompt or "VQA" in prompt:
        return SimpleNamespace(content='{{"next_step": "vqa", "action_details": {{"target": "ما محتوى الصورة بالتفصيل؟", "reasoning": "هناك صور مرفقة بحاجة لتحليل."}}}}')
    elif "يستخدم Search" in prompt or "Search" in prompt:
        return SimpleNamespace(content='{{"next_step": "search", "action_details": {{"target": "سعر دواء معين", "reasoning": "نحتاج معرفة السعر الحالي للدواء."}}}}')
    else:
        return SimpleNamespace(content='{{"next_step": "report", "action_details": {{"target": "", "reasoning": "المعلومات كافية لإصدار التقرير النهائي."}}}}')


def _mock_answer_single_question(image_path: str, question: str, ocr_text: str = "", context: str = "") -> Dict[str, Any]:
    return {
        "question": question,
        "answer": "هذا نص تجريبي للمسألة.",
        "confidence": 0.9,
        "reasoning": "إجابة مُولَّدة بشكل وهمي لأغراض الاختبار."
    }


def import_reasoning_module():
    _ensure_mock_service(
        "app.services.llm",
        {
            "llm": SimpleNamespace(invoke=_mock_llm_invoke),
            "llm_model": SimpleNamespace(invoke=_mock_llm_invoke, bind_tools=lambda tools: SimpleNamespace(invoke=_mock_llm_invoke))
        }
    )
    module = importlib.import_module("app.nodes.reasoning")
    return module


def run_case(case_name: str, state: Dict[str, Any]) -> None:
    reasoning_module = import_reasoning_module()
    result = reasoning_module.reasoning_node(state)
    print(f"\n=== {case_name} ===")
    print(result)


def main():
    cases = [
        {
            "name": "Reasoning: Images present -> use VQA",
            "state": {
                "normalized_case": {
                    "extracted_text": "الحالة تشير إلى مريض يحتاج نقل طبي عاجل.",
                    "image_count": 2
                },
                "evidence": {
                    "cold_acquisition": {
                        "overall_risk_score": 0.82,
                        "overall_risk_level": "high"
                    }
                },
                "images": ["data/img55.jpg", "data/prescription.jpg"],
                "text": "الشكوى مكتوبة بأن المريض يعاني من ألم حاد ويحتاج دعم عاجل."
            }
        },
        {
            "name": "Reasoning: Drug name known -> use Search",
            "state": {
                "normalized_case": {
                    "extracted_text": "اسم الدواء: إسبرين",
                    "image_count": 0
                },
                "evidence": {
                    "cold_acquisition": {
                        "overall_risk_score": 0.35,
                        "overall_risk_level": "medium"
                    }
                },
                "images": [],
                "text": "المريض يعاني من صداع ويحتاج إسبرين."
            }
        },
        {
            "name": "Reasoning: Clear case -> Report",
            "state": {
                "normalized_case": {
                    "extracted_text": "دواء معروف + سعر واضح",
                    "image_count": 0
                },
                "evidence": {
                    "cold_acquisition": {
                        "overall_risk_score": 0.1,
                        "overall_risk_level": "low"
                    }
                },
                "images": [],
                "text": "طلب واضح مع جميع المعلومات المطلوبة."
            }
        }
    ]

    for case in cases:
        run_case(case["name"], case["state"])


if __name__ == "__main__":
    main()
