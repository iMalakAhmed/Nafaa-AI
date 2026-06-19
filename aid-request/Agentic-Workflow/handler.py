import base64
import os
import tempfile
import runpod

from app.workflow import graph
from app.services.fraud_detection import detect_fraud
from app.services.vqa import get_vqa_model
from app.services.stt import transcribe_pipeline


# -----------------------------
# Helpers
# -----------------------------
def _strip_base64_header(data: str) -> str:
    """Removes data:*base64, prefix if present."""
    if isinstance(data, str):
        if "base64," in data:
            return data.split("base64,", 1)[1]
        if data.startswith("data:") and "," in data:
            return data.split(",", 1)[1]
    return data


def _decode_to_temp_file(b64_data: str, suffix: str, temp_files: list):
    """Decode base64 and store in temp file."""
    raw = _strip_base64_header(b64_data)
    file_bytes = base64.b64decode(raw)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()

    temp_files.append(tmp.name)
    return tmp.name


def _is_base64_string(data: str) -> bool:
    """Return True if the value is a valid base64-encoded string or data URI."""
    if not isinstance(data, str):
        return False

    raw = _strip_base64_header(data)
    if not raw.strip():
        return False

    try:
        base64.b64decode(raw, validate=True)
        return True
    except Exception:
        return False


# -----------------------------
# Handler
# -----------------------------
def handler(job):
    job_input = job.get("input", {})
    action = job_input.get("action", "full_pipeline")

    temp_files = []

    # =========================
    # 🔧 BASE64 NORMALIZATION
    # =========================

    voice_path = None
    voice_base64 = job_input.get("voice_base64") or job_input.get("audio_base64")
    if voice_base64:
        try:
            voice_path = _decode_to_temp_file(
                voice_base64,
                ".mp3",
                temp_files
            )
        except Exception as e:
            return {"error": f"Failed to decode voice audio: {str(e)}"}
    else:
        candidate_voice_path = job_input.get("voice_path") or job_input.get("audio_path")
        if isinstance(candidate_voice_path, str) and os.path.exists(candidate_voice_path):
            voice_path = candidate_voice_path

    images = []
    image_inputs = job_input.get("images_base64") or job_input.get("images")
    if image_inputs:
        if not isinstance(image_inputs, list):
            image_inputs = [image_inputs]

        for idx, img_item in enumerate(image_inputs):
            if isinstance(img_item, str) and os.path.exists(img_item):
                images.append({
                    "image_id": f"IMG-{idx+1:03d}",
                    "image_path": img_item,
                    "ocr_extracted_text": ""
                })
                continue

            if not isinstance(img_item, str):
                print(f"[WARN] Image {idx} skipped: unsupported type {type(img_item).__name__}")
                continue

            if not _is_base64_string(img_item):
                print(f"[WARN] Image {idx} skipped: not a valid base64 string or existing path")
                continue

            suffix = ".jpg"
            if img_item.startswith("data:image/png") or "image/png" in img_item:
                suffix = ".png"

            try:
                img_path = _decode_to_temp_file(
                    img_item,
                    suffix,
                    temp_files
                )

                images.append(img_path)

            except Exception as e:
                print(f"[WARN] Image {idx} failed: {e}")

    # =========================
    # ROUTING
    # =========================

    if action == "full_pipeline":
        initial_state = {
            "text": job_input.get("text", ""),
            "user_id":  job_input.get("user_id", ""),
            "images": images,
            "voice_path": voice_path,
            "evidence": {},
            "inquiry_history": [],
            "loop_count": 0
        }
        return graph.invoke(initial_state)

    elif action == "vqa":
        vqa_model, vqa_processor = get_vqa_model()
        image_path = images[0] if images else None
        return {"status": "vqa_done", "image_used": image_path}

    elif action == "stt":
        return {
            "text": transcribe_pipeline(voice_path)
        }

    else:
        return {"error": f"Action '{action}' not supported."}


runpod.serverless.start({"handler": handler})