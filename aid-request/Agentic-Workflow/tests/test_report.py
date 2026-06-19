import importlib
import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict

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
    # Return report format
    return SimpleNamespace(content='{"case_summary": "حالة طبية طارئة بحاجة لرعاية عاجلة.", "urgent_need": "دعم تكاليف علاج عاجل", "severity": "عالي", "key_evidence": "وثيقة طبية", "concerns": [], "recommended_action": "قبول الدعم المالي العاجل", "support_options": ["مساعدة مالية كاملة"], "suggested_next_steps": ["التواصل مع وحدة الدعم"], "confidence": 0.9, "admin_summary": "اطلب دعم عاجل"}')


def import_report_module():
    _ensure_mock_service(
        "app.services.llm",
        {
            "llm": SimpleNamespace(invoke=_mock_llm_invoke),
            "llm_model": SimpleNamespace(invoke=_mock_llm_invoke, bind_tools=lambda tools: SimpleNamespace(invoke=_mock_llm_invoke))
        }
    )
    module = importlib.import_module("app.nodes.report")
    module.llm.invoke = _mock_llm_invoke
    return module


def run_case(case_name: str, state: Dict[str, Any]) -> None:
    report_module = import_report_module()
    result = report_module.report_node(state)
    print(f"\n=== {case_name} ===")
    print(result)


def main():
    cases = [
        {
            "name": "Report generation with clear reasoning",
            "state": {
                "normalized_case": {"extracted_text": "الشكوى تشير لاحتياج طبي عاجل"},
                "evidence": {
                    "cold_acquisition": {"overall_risk_level": "high"},
                    "vqa_analysis": {"results": [{"question": "ما محتوى الصورة؟", "answer": "روشتة"}]}
                },
                "reasoning": {
                    "next_step": "report",
                    "question_or_query": "",
                    "reasoning": "المعلومات كافية لاصدار التقرير"
                }
            }
        },
        {
            "name": "Report generation with search results",
            "state": {
                "normalized_case": {"extracted_text": "طلب دعم لشراء دواء."},
                "evidence": {
                    "cold_acquisition": {"overall_risk_level": "medium"},
                    "search": {"pricing": [{"item": "دواء", "price": "50-100"}]}
                },
                "reasoning": {
                    "next_step": "report",
                    "question_or_query": "",
                    "reasoning": "جميع المعلومات متاحة للتقرير"
                }
            }
        }
    ]

    for case in cases:
        run_case(case["name"], case["state"])


if __name__ == "__main__":
    main()
