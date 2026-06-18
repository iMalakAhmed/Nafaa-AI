"""Modal batch function — YOLO + QARI-OCR on birth-certificate images.

Deploy & run:
    modal deploy document_parsing/birthcert/qari_modal.py
    modal run    document_parsing/birthcert/qari_modal.py        # runs full val set
    modal run    document_parsing/birthcert/qari_modal.py::run_batch --limit 3

Or call the API endpoint directly:
    POST /birthcert/qari   multipart file upload, returns JSON record
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import modal

PROJECT_DIR   = "/root/project"
HF_CACHE_VOL  = "case-study-hf-cache"
YOLO_WEIGHTS  = f"{PROJECT_DIR}/document_parsing/outputs/birthcert_yolo/field_detector/weights/best.pt"
YOLO_DATASET  = f"{PROJECT_DIR}/document_parsing/data/birthcert_yolo"
QARI_MODEL    = "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install_from_requirements("requirements.txt")
    .pip_install(
        "torchvision",
        "peft>=0.12.0",
        "ultralytics",
        "Pillow",
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[
            ".venv", ".git", "__pycache__", ".pytest_cache",
            "outputs", "notebooks", ".modal", "agent-tools",
        ],
    )
    .add_local_file(
        "document_parsing/outputs/birthcert_yolo/field_detector/weights/best.pt",
        remote_path=f"{PROJECT_DIR}/document_parsing/outputs/birthcert_yolo/field_detector/weights/best.pt",
    )
)

hf_volume = modal.Volume.from_name(HF_CACHE_VOL, create_if_missing=True)
app = modal.App("birthcert-qari-eval")


@app.function(
    image=image,
    gpu="L4",
    timeout=600,
    volumes={"/cache/huggingface": hf_volume},
    scaledown_window=120,
)
def extract_one_b64(image_b64: str, filename: str = "image.jpeg") -> dict:
    """Extract a single birth certificate image (base64-encoded JPEG/PNG) with YOLO+QARI."""
    import os, sys
    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from pathlib import Path as _Path
    from PIL import Image as _Image
    from document_parsing.birthcert.yolo_ocr import (
        HfVlmOcrBackend, load_class_names, detect_fields,
        pick_best_by_class, padded_crop, enhance_crop,
        normalize_field_text, CLASS_TO_PATH, _looks_degenerate,
    )
    from document_parsing.birthcert.schema import empty_record, set_path
    from document_parsing.birthcert.validate import validate_record
    from ultralytics import YOLO
    import PIL.ImageOps as _ImageOps

    # Load models (cached across warm containers)
    global _yolo_model, _ocr
    if "_yolo_model" not in globals() or _yolo_model is None:
        _yolo_model = YOLO(YOLO_WEIGHTS)
    if "_ocr" not in globals() or _ocr is None:
        _ocr = HfVlmOcrBackend(QARI_MODEL)

    # Decode image
    img_bytes = base64.b64decode(image_b64)
    image = _Image.open(io.BytesIO(img_bytes))
    image = _ImageOps.exif_transpose(image).convert("RGB")

    # Save temp for YOLO (needs a file path)
    tmp_path = _Path(f"/tmp/{filename}")
    image.save(str(tmp_path))

    class_names = load_class_names(_Path(YOLO_DATASET))
    detections  = pick_best_by_class(
        detect_fields(tmp_path, yolo_model=_yolo_model, class_names=class_names, conf=0.25)
    )

    stem = _Path(filename).stem
    raw_record = empty_record(document_id=stem, source_files=[filename])
    raw_items  = []

    for field_name, det in sorted(detections.items()):
        crop = enhance_crop(padded_crop(image, det.box_xyxy))
        raw_text  = _ocr.read(crop, field_name=field_name)
        degenerate = _looks_degenerate(raw_text)
        value = normalize_field_text(field_name, raw_text) if not degenerate else None
        raw_items.append({
            "field": field_name,
            "confidence": det.confidence,
            "ocr_raw": raw_text,
            "degenerate": degenerate,
            "value": value,
        })
        if value is not None:
            set_path(raw_record, CLASS_TO_PATH[field_name], value)

    record = validate_record(raw_record, document_id=stem, source_files=[filename])
    if not detections:
        record["review_required"] = True
        record["review_notes"].append("YOLO detected no fields.")
    return {"record": record, "raw_items": raw_items}


@app.local_entrypoint()
def run_batch(limit: int = 0):
    """Send val images to the Modal function and save results + CER/WER eval."""
    import base64, json, subprocess, sys
    from pathlib import Path

    images_dir = Path("document_parsing/data/birthcert_yolo/images/val")
    out_dir    = Path("document_parsing/outputs/birthcert/yolo_qari/records")
    raw_dir    = Path("document_parsing/outputs/birthcert/yolo_qari/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if limit:
        image_paths = image_paths[:limit]

    print(f"[qari-batch] Processing {len(image_paths)} images …")
    for image_path in image_paths:
        print(f"  {image_path.name} …", end=" ", flush=True)
        img_b64 = base64.b64encode(image_path.read_bytes()).decode()
        result  = extract_one_b64.remote(img_b64, image_path.name)
        out_dir.joinpath(f"{image_path.stem}.json").write_text(
            json.dumps(result["record"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raw_dir.joinpath(f"{image_path.stem}.json").write_text(
            json.dumps(result["raw_items"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("OK")

    # Evaluate
    labels_dir = Path("document_parsing/data/birth_cert_labels")
    if labels_dir.exists():
        print("\n[qari-batch] Field-level evaluation (structured):")
        subprocess.run([sys.executable, "-m", "document_parsing.birthcert.evaluate",
                        "--pred", str(out_dir), "--labels", str(labels_dir)])
        print("\n[qari-batch] CER/WER evaluation (OCR text quality):")
        subprocess.run([sys.executable, "-m", "document_parsing.birthcert.evaluate_yolo_ocr_text",
                        "--reads", str(raw_dir), "--labels", str(labels_dir)])
