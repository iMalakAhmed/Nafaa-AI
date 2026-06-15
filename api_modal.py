"""Modal web API for JSON document extraction.

Deploy:
  modal deploy api_modal.py

Endpoints:
  POST /birthcert/extract    multipart image -> birth certificate JSON via Gemini
  POST /casestudy/extract    multipart image -> case-study JSON via Gemini
  POST /national-id/extract  multipart image -> Egyptian national ID JSON via YOLO+EasyOCR
  GET  /health
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import modal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse


PROJECT_DIR = "/root/project"
HF_CACHE_VOL = "case-study-hf-cache"
NATIONAL_ID_MODEL_DIR = "/root/national-id-models"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .pip_install(
        "fastapi>=0.115.0",
        "python-multipart>=0.0.9",
        "google-genai>=1.0.0",
        "opencv-python-headless>=4.9.0",
        "easyocr>=1.7.1",
        "ultralytics>=8.3.0",
    )
    .run_commands(
        f"mkdir -p {NATIONAL_ID_MODEL_DIR}",
        "curl -L --retry 5 --retry-delay 2 "
        "https://raw.githubusercontent.com/ebrahimabdelghfar/National-ID-Reader/main/card_finder_seg.pt "
        f"-o {NATIONAL_ID_MODEL_DIR}/card_finder_seg.pt",
        "curl -L --retry 5 --retry-delay 2 "
        "https://raw.githubusercontent.com/ebrahimabdelghfar/National-ID-Reader/main/card_divider_model.pt "
        f"-o {NATIONAL_ID_MODEL_DIR}/card_divider_model.pt",
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[
            ".venv",
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".modal",
            "outputs",
            "notebooks",
            "agent-tools",
        ],
    )
)

hf_volume = modal.Volume.from_name(HF_CACHE_VOL, create_if_missing=True)

try:
    gemini_secret = modal.Secret.from_name("gemini-api-key")
except Exception:
    gemini_secret = None

app = modal.App("document-extraction-api")
web_app = FastAPI(title="Document Extraction API", version="1.0.0")

_national_id_reader = None


def _setup() -> None:
    import sys

    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)


def _require_image(file: UploadFile) -> None:
    if file.content_type and file.content_type.startswith("image/"):
        return
    suffix = Path(file.filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return
    raise HTTPException(status_code=400, detail="File must be an image upload")


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    _require_image(file)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    doc_id = Path(file.filename or "upload").stem
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        return tmp.name, doc_id


def _cleanup(path: str) -> None:
    Path(path).unlink(missing_ok=True)


def _gemini_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the deployed Modal app",
        )
    return key


def _get_national_id_reader():
    global _national_id_reader
    if _national_id_reader is None:
        _setup()
        from nationalid.extract import NationalIdReader

        _national_id_reader = NationalIdReader(gpu=True)
    return _national_id_reader


@web_app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "document-extraction-api",
        "endpoints": [
            "/birthcert/extract",
            "/casestudy/extract",
            "/national-id/extract",
        ],
    }


@web_app.post("/birthcert/extract")
async def extract_birthcert(file: UploadFile = File(...)):
    tmp_path, doc_id = await _save_upload(file)
    try:
        _setup()
        from birthcert.extract_gemini import extract_one

        record, raw_text = extract_one(
            tmp_path,
            api_key=_gemini_key(),
            document_id=doc_id,
            model=os.environ.get("GEMINI_BIRTHCERT_MODEL", "gemini-2.5-flash"),
            max_tokens=int(os.environ.get("GEMINI_BIRTHCERT_MAX_TOKENS", "4096")),
            rpm_delay=0,
        )
        return JSONResponse(content={"record": record, "raw_text": raw_text})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(tmp_path)


@web_app.post("/casestudy/extract")
async def extract_casestudy(file: UploadFile = File(...), use_regions: bool = True):
    tmp_path, doc_id = await _save_upload(file)
    try:
        _setup()
        from casestudy.extract_gemini import extract_one

        record, raw_text = extract_one(
            tmp_path,
            api_key=_gemini_key(),
            document_id=doc_id,
            model=os.environ.get("GEMINI_CASESTUDY_MODEL", "gemini-2.5-flash"),
            max_tokens=int(os.environ.get("GEMINI_CASESTUDY_MAX_TOKENS", "4096")),
            use_regions=use_regions,
            rpm_delay=0,
        )
        return JSONResponse(content={"record": record, "raw_text": raw_text})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(tmp_path)


@web_app.post("/national-id/extract")
async def extract_national_id(file: UploadFile = File(...)):
    tmp_path, doc_id = await _save_upload(file)
    try:
        reader = _get_national_id_reader()
        record = reader.extract_image(tmp_path, document_id=doc_id)
        return JSONResponse(content={"record": record})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(tmp_path)


function_kwargs = {
    "image": image,
    "gpu": "L4",
    "timeout": 60 * 10,
    "scaledown_window": 300,
    "env": {
        "NATIONAL_ID_MODEL_DIR": NATIONAL_ID_MODEL_DIR,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    "volumes": {
        "/root/.cache/huggingface": hf_volume,
    },
}
if gemini_secret is not None:
    function_kwargs["secrets"] = [gemini_secret]


@app.function(**function_kwargs)
@modal.asgi_app()
def fastapi_app():
    return web_app
