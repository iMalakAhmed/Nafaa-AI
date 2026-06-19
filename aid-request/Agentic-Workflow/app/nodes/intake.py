import os
import json

from app.state import CaseState
from app.services.stt import transcribe
# Assuming you have a utility function that wraps Tesseract, EasyOCR, or a cloud API
from app.services.ocr import extract_text_from_image 

def intake_node(state: CaseState) -> dict:
    """
    Deterministic Multi-Modal Intake Node.
    Extracts text raw data from Audio (STT) and Images (OCR) dynamically using Python.
    """
    raw_text = state.get("text") or ""
    voice_path = state.get("voice_path")
    image_paths = state.get("images") or []

    transcript = None
    ocr_results = []

    # 1. Run STT Tool if audio exists
    if voice_path and os.path.exists(voice_path):
        try:
            transcript = transcribe(voice_path)
        except Exception as e:
            transcript = f"[خطأ في تفريغ الصوت: {str(e)}]"

    # 2. Run OCR Tool across all uploaded images
    for img_path in image_paths:
        if os.path.exists(img_path):
            try:
                extracted_text = extract_text_from_image(img_path)
                if extracted_text.strip():
                    ocr_results.append(f"--- نص مستخرج من صورة ({os.path.basename(img_path)}) ---\n{extracted_text.strip()}")
            except Exception as e:
                ocr_results.append(f"[خطأ في قراءة الصورة {os.path.basename(img_path)}: {str(e)}]")

    # 3. Consolidate all extracted textual streams cleanly
    combined_elements = []
    
    if raw_text and str(raw_text).strip():
        combined_elements.append(f"--- الشكوى النصية المكتوبة ---\n{str(raw_text).strip()}")
        
    if transcript:
        # ROBUST CHECK: Extract text if transcript is returned as a dict or string
        clean_transcript = ""
        if isinstance(transcript, dict):
            clean_transcript = transcript.get("text", json.dumps(transcript, ensure_ascii=False))
        else:
            clean_transcript = str(transcript)
            
        if clean_transcript.strip():
            combined_elements.append(f"--- تفريغ الشكوى الصوتية ---\n{clean_transcript.strip()}")
            
    if ocr_results:
        combined_elements.extend(ocr_results)

    final_text = "\n\n".join(combined_elements) if combined_elements else ""

    # 4. Build metadata structure for down-stream validation
    normalized_case = {
        "extracted_text": final_text,
        "has_voice": voice_path is not None,
        "has_images": len(image_paths) > 0,
        "image_count": len(image_paths),
        "ocr_executed": len(ocr_results) > 0
    }

    return {
        "text": final_text,
        "transcript": transcript if isinstance(transcript, str) else json.dumps(transcript, ensure_ascii=False),
        "normalized_case": normalized_case,
        "images": image_paths,
        "user_id": state.get("user_id"),
        "request_category": state.get("request_category")
    }