import importlib
import os
import sys
import types
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


def import_evidence_module():
    try:
        return importlib.import_module("app.nodes.evidence")
    except ImportError:
        _ensure_mock_service(
            "app.services.fraud_detection",
            {"ai_generated_probability": lambda image_path: 0.72}
        )
        _ensure_mock_service(
            "app.services.reverse_image",
            {"find_duplicates": lambda image_path, user_id: {"duplicate_different_user": False, "duplicate_same_user": False}}
        )
        _ensure_mock_service(
            "app.services.quality_gate_finalized",
            {"check_quality": lambda image_path: {"quality_score": 0.88}}
        )
        return importlib.import_module("app.nodes.evidence")


def create_dummy_image(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("dummy image content")


def run_case(case_name: str, state: Dict[str, Any]) -> None:
    evidence_module = import_evidence_module()
    result = evidence_module.evidence_node(state)
    print(f"\n=== {case_name} ===")
    print(result)


def main():
    sample_image = "data/sample_evidence_image.jpg"
    create_dummy_image(sample_image)

    cases = [
        {
            "name": "Single image with baseline text",
            "state": {
                "text": "هذا فحص لمستند طبي",
                "images": [sample_image],
                "user_id": "user_evidence_1",
                "evidence": {}
            }
        },
        {
            "name": "Image and existing evidence payload",
            "state": {
                "text": "طلب دعم عاجل بعد حادث.",
                "images": [sample_image],
                "user_id": "user_evidence_2",
                "evidence": {"previous_notes": "existing evidence"}
            }
        },
        {
            "name": "No image input",
            "state": {
                "text": "لا يوجد صور.",
                "images": [],
                "user_id": "user_evidence_3",
                "evidence": {}
            }
        }
    ]

    for case in cases:
        run_case(case["name"], case["state"])


if __name__ == "__main__":
    main()
