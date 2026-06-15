"""RunPod Serverless handler for document extraction.

Input JSON:
  {
    "input": {
      "image": "<base64 image bytes or data:image/...;base64,...>",
      "image_url": "https://...",                  # optional alternative
      "document_type": "birthcert|casestudy|national_id|auto",
      "filename": "upload.jpg",
      "use_regions": true,                         # case-study only
      "include_raw": false                         # include Gemini raw text
    }
  }

Output JSON:
  {
    "output": {
      "document_type": "birthcert|casestudy|national_id",
      "record": {...},
      "raw_text": "..."                            # only when include_raw=true
    }
  }
"""

from __future__ import annotations

import base64
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen


PROJECT_DIR = "/app"
NATIONAL_ID_MODEL_DIR = os.environ.get("NATIONAL_ID_MODEL_DIR", "/app/national-id-models")
BC_ADAPTER_PATH = os.environ.get("ADAPTER_BC_PATH") or None

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

_national_id_reader = None
_birthcert_extractor = None
_classifier_reader = None


def _decode_image(inp: dict) -> tuple[bytes, str]:
    filename = inp.get("filename") or "upload.jpg"

    image_b64 = inp.get("image")
    image_url = inp.get("image_url")
    if image_b64:
        if "," in image_b64 and image_b64.lstrip().startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        try:
            return base64.b64decode(image_b64), filename
        except Exception as exc:
            raise ValueError("'image' is not valid base64") from exc

    if image_url:
        with urlopen(image_url, timeout=30) as response:
            return response.read(), filename or Path(image_url).name or "upload.jpg"

    raise ValueError("Missing required field: 'image' base64 or 'image_url'")


def _write_temp_image(image_bytes: bytes, filename: str) -> tuple[str, str]:
    if not image_bytes:
        raise ValueError("Uploaded image is empty")
    suffix = Path(filename).suffix or ".jpg"
    doc_id = Path(filename).stem or "upload"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        return tmp.name, doc_id


def _normalize_document_type(value: str | None) -> str:
    doc_type = (value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "birth_certificate": "birthcert",
        "birth_cert": "birthcert",
        "bc": "birthcert",
        "case_study": "casestudy",
        "case": "casestudy",
        "cs": "casestudy",
        "nationalid": "national_id",
        "national_id_card": "national_id",
        "id": "national_id",
        "nid": "national_id",
    }
    return aliases.get(doc_type, doc_type)


def _gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY is not configured on the RunPod endpoint")
    return key


def _classify_document(image_path: str) -> str:
    """Classify without Gemini.

    Fast path:
    - Egyptian national IDs are landscape cards, so route them by aspect ratio.
    - For full-page documents, OCR only the title/top band and look for stable
      Arabic keywords.
    """
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    aspect = width / max(1, height)

    if aspect >= float(os.environ.get("AUTO_NID_MIN_ASPECT", "1.25")):
        return "national_id"

    top_ratio = float(os.environ.get("AUTO_TITLE_CROP_RATIO", "0.38"))
    top = image.crop((0, 0, width, int(height * top_ratio)))
    text = _ocr_top_text(top)
    compact = re.sub(r"\s+", "", text)

    birth_keywords = (
        "\u0634\u0647\u0627\u062f\u0629\u0645\u064a\u0644\u0627\u062f",  # شهادةميلاد
        "\u0645\u064a\u0644\u0627\u062f",  # ميلاد
        "\u0648\u0627\u0642\u0639\u0629\u0645\u064a\u0644\u0627\u062f",  # واقعةميلاد
    )
    case_keywords = (
        "\u0628\u062d\u062b\u0627\u062c\u062a\u0645\u0627\u0639\u064a",  # بحثاجتماعي
        "\u0627\u0644\u062a\u0636\u0627\u0645\u0646\u0627\u0644\u0627\u062c\u062a\u0645\u0627\u0639\u064a",  # التضامنالاجتماعي
        "\u0627\u0644\u0648\u062d\u062f\u0629\u0627\u0644\u0627\u062c\u062a\u0645\u0627\u0639\u064a\u0629",  # الوحدةالاجتماعية
        "\u0627\u0644\u0623\u0633\u0631\u0629",  # الأسرة
        "\u0627\u0644\u062d\u0627\u0644\u0629\u0627\u0644\u0627\u062c\u062a\u0645\u0627\u0639",  # الحالةالاجتماع...
        "\u0627\u0644\u0623\u0645\u0644\u0627\u0643",  # الأملاك
        "\u0627\u0644\u0627\u0639\u0627\u0642\u0629",  # الاعاقة
        "\u0627\u0644\u0635\u062d\u064a\u0629",  # الصحية
    )
    national_id_keywords = (
        "\u0628\u0637\u0627\u0642\u0629\u062a\u062d\u0642\u064a\u0642\u0627\u0644\u0634\u062e\u0635\u064a\u0629",  # بطاقةتحقيقالشخصية
        "\u0628\u0637\u0627\u0642\u0629",  # بطاقة
    )

    if any(keyword in compact for keyword in national_id_keywords):
        return "national_id"
    if any(keyword in compact for keyword in birth_keywords):
        return "birthcert"
    if any(keyword in compact for keyword in case_keywords):
        return "casestudy"

    fallback = os.environ.get("AUTO_DEFAULT_DOCUMENT_TYPE", "casestudy").strip()
    if fallback:
        doc_type = _normalize_document_type(fallback)
        if doc_type in {"birthcert", "casestudy", "national_id"}:
            return doc_type

    raise ValueError(
        "Could not auto-classify document type from layout/title. "
        "Send document_type explicitly as one of: birthcert, casestudy, national_id."
    )


def _ocr_top_text(image) -> str:
    global _classifier_reader
    try:
        import numpy as np
        import easyocr

        if _classifier_reader is None:
            gpu = os.environ.get("AUTO_CLASSIFIER_OCR_GPU", "false").lower() in {"1", "true", "yes", "on"}
            _classifier_reader = easyocr.Reader(["ar", "en"], gpu=gpu)
        result = _classifier_reader.readtext(np.array(image), detail=0, paragraph=True)
        return " ".join(str(item) for item in result)
    except Exception as exc:
        print(f"[auto-classifier] title OCR failed: {exc}", flush=True)
        return ""


def _get_birthcert_extractor():
    global _birthcert_extractor
    if _birthcert_extractor is None:
        from birthcert.extract import BirthCertExtractor

        if not BC_ADAPTER_PATH or not Path(BC_ADAPTER_PATH).exists():
            raise FileNotFoundError(
                "Birth certificate adapter not found. Set ADAPTER_BC_PATH to bc_lora_v4 "
                "or bake it into /app/deploy/adapters/bc_lora_v4."
            )
        _birthcert_extractor = BirthCertExtractor(
            model_name=os.environ.get("BIRTHCERT_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
            torch_dtype=os.environ.get("BIRTHCERT_TORCH_DTYPE", "bfloat16"),
            attn_implementation=os.environ.get("BIRTHCERT_ATTN", "sdpa"),
            max_pixels=int(os.environ.get("BIRTHCERT_MAX_PIXELS", "1280000")),
            adapter_path=BC_ADAPTER_PATH,
        )
    return _birthcert_extractor


def _extract_birthcert(image_path: str, doc_id: str) -> tuple[dict, str]:
    extractor = _get_birthcert_extractor()
    return extractor.extract(
        image_path,
        document_id=doc_id,
        max_new_tokens=int(os.environ.get("BIRTHCERT_MAX_NEW_TOKENS", "2048")),
        enhance_image=True,
    )


def _extract_casestudy(image_path: str, doc_id: str, *, use_regions: bool) -> tuple[dict, str]:
    from casestudy.extract_gemini import extract_one

    return extract_one(
        image_path,
        api_key=_gemini_key(),
        document_id=doc_id,
        model=os.environ.get("GEMINI_CASESTUDY_MODEL", "gemini-2.5-flash"),
        max_tokens=int(os.environ.get("GEMINI_CASESTUDY_MAX_TOKENS", "4096")),
        use_regions=use_regions,
        rpm_delay=0,
    )


def _get_national_id_reader():
    global _national_id_reader
    if _national_id_reader is None:
        from nationalid.extract import NationalIdReader

        model_dir = Path(NATIONAL_ID_MODEL_DIR)
        _national_id_reader = NationalIdReader(
            finder_model=model_dir / "card_finder_seg.pt",
            divider_model=model_dir / "card_divider_model.pt",
            gpu=True,
        )
    return _national_id_reader


def _extract_national_id(image_path: str, doc_id: str) -> dict:
    backend = os.environ.get("NATIONAL_ID_BACKEND", "gemini").strip().lower()
    if backend != "easyocr" and os.environ.get("GEMINI_API_KEY"):
        from nationalid.extract_gemini import extract_one

        try:
            return extract_one(
                image_path,
                api_key=_gemini_key(),
                document_id=doc_id,
                model=os.environ.get("GEMINI_NATIONAL_ID_MODEL", "gemini-2.5-flash-lite"),
                max_tokens=int(os.environ.get("GEMINI_NATIONAL_ID_MAX_TOKENS", "2048")),
            )
        except Exception as exc:
            message = str(exc)
            if "429" not in message and "RESOURCE_EXHAUSTED" not in message and "quota" not in message.lower():
                raise
            return {
                "document_id": doc_id,
                "document_type": "national_id",
                "source_files": [image_path],
                "full_name": None,
                "first_name": None,
                "remaining_name": None,
                "address": None,
                "street": None,
                "city": None,
                "governorate": None,
                "national_id": None,
                "raw_national_id": None,
                "review_required": True,
                "review_notes": [
                    "Gemini quota exhausted for national ID extraction; retry later or use a paid/higher-quota key."
                ],
            }
    return _get_national_id_reader().extract_image(image_path, document_id=doc_id)


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    tmp_path: str | None = None

    try:
        image_bytes, filename = _decode_image(inp)
        tmp_path, doc_id = _write_temp_image(image_bytes, filename)

        doc_type = _normalize_document_type(inp.get("document_type"))
        if doc_type == "auto":
            doc_type = _classify_document(tmp_path)
        if doc_type not in {"birthcert", "casestudy", "national_id"}:
            raise ValueError(
                "document_type must be one of: birthcert, casestudy, national_id, auto"
            )

        include_raw = bool(inp.get("include_raw", False))
        if doc_type == "birthcert":
            record, raw_text = _extract_birthcert(tmp_path, doc_id)
        elif doc_type == "casestudy":
            use_regions = bool(inp.get("use_regions", True))
            record, raw_text = _extract_casestudy(tmp_path, doc_id, use_regions=use_regions)
        else:
            record = _extract_national_id(tmp_path, doc_id)
            raw_text = ""

        output = {"document_type": doc_type, "record": record}
        if include_raw and raw_text:
            output["raw_text"] = raw_text
        return output

    except Exception as exc:
        return {"error": str(exc)}

    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    print("[runpod] handler module loaded", flush=True)
    try:
        import runpod

        print("[runpod] starting serverless worker", flush=True)
        runpod.serverless.start({"handler": handler})
    except Exception:
        import traceback

        traceback.print_exc()
        raise
