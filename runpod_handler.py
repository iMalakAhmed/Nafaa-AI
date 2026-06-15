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
    import google.genai as genai
    from google.genai import types

    key = _gemini_key()
    client = genai.Client(api_key=key)
    image_bytes = Path(image_path).read_bytes()
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_CLASSIFIER_MODEL", "gemini-2.5-flash"),
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(
                text=(
                    "Classify this document image. Return exactly one label only: "
                    "birthcert, casestudy, national_id. No explanation."
                )
            ),
        ],
        config=types.GenerateContentConfig(max_output_tokens=16, temperature=0.0),
    )
    label = re.sub(r"[^a-zA-Z_]", "", response.text or "").lower()
    doc_type = _normalize_document_type(label)
    if doc_type not in {"birthcert", "casestudy", "national_id"}:
        raise ValueError(f"Could not auto-classify document type from Gemini label: {response.text!r}")
    return doc_type


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

        return extract_one(
            image_path,
            api_key=_gemini_key(),
            document_id=doc_id,
            model=os.environ.get("GEMINI_NATIONAL_ID_MODEL", "gemini-2.5-flash-lite"),
            max_tokens=int(os.environ.get("GEMINI_NATIONAL_ID_MAX_TOKENS", "2048")),
        )
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
