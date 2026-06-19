import importlib
import os
import sys
import json
from typing import Any, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _load_search_module():
    try:
        module = importlib.import_module("app.agents.search")
        return module
    except Exception as exc:
        print("Failed to import app.agents.search:", exc)
        print("Please ensure required dependencies are installed: langgraph, langchain_core, tavily, rapidfuzz, langchain_ollama.")
        return None


def _mock_search_invoke(state_input: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "search_results": {
            "definitions": [
                {"term": "روشتة", "definition": "وثيقة طبية تحتوي على أدوية يوصى بها الطبيب"}
            ],
            "medical_analysis": [
                {"case_or_drug": "دواء", "usage_causes": "يُستخدم لعلاج الألم", "source": "mock"}
            ],
            "pricing": [
                {"item": "دواء مسكن", "price": "50-100 جنيه", "type": "purchase", "sources": ["mock source"]}
            ]
        }
    }


def run_case(case_name: str, state: Dict[str, Any]) -> None:
    print(f"\n=== {case_name} ===")
    search_module = _load_search_module()
    if search_module is None:
        print("Skipping search node test due to import failure.")
        return

    if hasattr(search_module, "compiled_search_agent"):
        search_module.compiled_search_agent.invoke = _mock_search_invoke

    if hasattr(search_module, "search_agent"):
        result = search_module.search_agent(state)
        print(result)
    else:
        print("Search module imported but no search_agent symbol found.")


def main():
    cases = [
        {
            "name": "Search for drug information",
            "state": {
                "text": "بحث عن دواء الأسبرين وسعره.",
                "images": [],
                "evidence": {},
                "reasoning": {
                    "next_step": "search",
                    "question_or_query": "سعر وبدائل الأسبرين"
                }
            }
        },
        {
            "name": "Search for medical case",
            "state": {
                "text": "مريض يعاني من صداع بعد ربط نسيب؏",
                "images": [],
                "evidence": {},
                "reasoning": {
                    "next_step": "search",
                    "question_or_query": "علاج الصداع بعد ربط نسيبي"
                }
            }
        }
    ]


if __name__ == "__main__":
    main()
