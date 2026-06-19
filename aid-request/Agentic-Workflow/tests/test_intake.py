import importlib
import importlib.util
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


def import_intake_module():
    try:
        return importlib.import_module("app.nodes.intake")
    except ImportError:
        _ensure_mock_service(
            "app.services.stt",
            {"transcribe": lambda audio_path: f"[mock transcript for {os.path.basename(audio_path)}]"}
        )
        _ensure_mock_service(
            "app.services.ocr",
            {"extract_text_from_image": lambda image_path: f"[mock OCR text from {os.path.basename(image_path)}]"}
        )
        return importlib.import_module("app.nodes.intake")


def create_dummy_file(path: str, content: str = "mock data") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def run_case(case_name: str, state: Dict[str, Any]) -> None:
    intake_module = import_intake_module()
    result = intake_module.intake_node(state)
    print(f"\n=== {case_name} ===")
    print(result)


def main():
    sample_audio = "data/v333.mp3"
    sample_image_1 = "data/img55.jpg"
    sample_image_2 = "data/prescription.jpg"
 
    cases = [
        {
            "name": "Text only input",
            "state": {
                "text": "الشكوى: مريض يحتاج دواء عاجل.",
                "voice_path": None,
                "images": [],
                "user_id": "test_user_1",
                "request_category": "medical",
                "evidence": {}
            }
        },
        {
            "name": "Audio only input",
            "state": {
                "text": "",
                "voice_path": sample_audio,
                "images": [],
                "user_id": "test_user_2",
                "request_category": "medical",
                "evidence": {}
            }
        },
        {
            "name": "Image only input",
            "state": {
                "text": "",
                "voice_path": None,
                "images": [sample_image_1, sample_image_2],
                "user_id": "test_user_3",
                "request_category": "medical",
                "evidence": {}
            }
        },
        {
            "name": "Combined text + image input",
            "state": {
                "text": "شاب محتاج كرسي متحرك بعد حادث.",
                "voice_path": sample_audio,
                "images": [sample_image_1, sample_image_2],
                "user_id": "test_user_4",
                "request_category": "medical",
                "evidence": {}
            }
        }
    ]

    for case in cases:
        run_case(case["name"], case["state"])


if __name__ == "__main__":
    main()
