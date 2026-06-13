"""Birth-certificate extraction via YOLO field detection + Arabic OCR.

The detector finds value regions. OCR reads each crop. Rules map OCR text into
the existing birth-certificate JSON schema and validation layer.

Supported OCR backends:
  - easyocr: quick local baseline for Arabic/English printed text.
  - hf-vlm: HuggingFace vision-language OCR model, configured by --ocr-model.
            Use this for QARI-OCR, Baseer, Arabic-GLM, or similar models when
            their exact HuggingFace model IDs are available.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageOps

from .schema import empty_record, set_path
from .validate import normalize_digits, validate_record


CLASS_TO_PATH = {
    "child_national_id": "ids.child_national_id",
    "child_name": "personal_and_other.child.name",
    "date_of_birth": "personal_and_other.child.date_of_birth",
    "place_of_birth": "personal_and_other.child.place_of_birth",
    "father_name": "personal_and_other.father.name",
    "mother_name": "personal_and_other.mother.name",
    "registration_number": "birth_certificate.registration_number",
    "registration_date": "birth_certificate.registration_date",
    "issue_date": "birth_certificate.issue_date",
    "serial_number": "birth_certificate.serial_number",
}


FIELD_PROMPTS = {
    "child_national_id": "Read only the national ID digits in this crop. Return digits only.",
    "registration_number": "Read only the registration number in this crop. Return the value only.",
    "registration_date": "Read only the registration date in this crop. Return the date only.",
    "issue_date": "Read only the issue date in this crop. Return the date only.",
    "serial_number": "Read only the serial number in this crop. Return digits only.",
}


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


class OcrBackend(Protocol):
    def read(self, image: Image.Image, *, field_name: str) -> str:
        ...


def _clean_ocr_text(text: str) -> str | None:
    text = normalize_digits(text or "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;،")
    return text or None


def _digits_only(text: str) -> str | None:
    digits = re.sub(r"\D", "", normalize_digits(text or ""))
    return digits or None


def _date_like(text: str) -> str | None:
    norm = normalize_digits(text or "")
    match = re.search(r"\d{1,4}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{1,4}", norm)
    if match:
        return re.sub(r"\s+", "", match.group(0))
    return _clean_ocr_text(norm)


def normalize_field_text(field_name: str, text: str) -> str | None:
    if field_name in {"child_national_id", "serial_number"}:
        return _digits_only(text)
    if field_name in {"registration_number"}:
        return _digits_only(text) or _clean_ocr_text(text)
    if field_name in {"registration_date", "issue_date", "date_of_birth"}:
        return _date_like(text)
    return _clean_ocr_text(text)


def padded_crop(image: Image.Image, box: tuple[float, float, float, float], pad_ratio: float = 0.04) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    bw = right - left
    bh = bottom - top
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    left = max(0, int(left - pad_x))
    top = max(0, int(top - pad_y))
    right = min(width, int(right + pad_x))
    bottom = min(height, int(bottom + pad_y))
    return image.crop((left, top, right, bottom))


def enhance_crop(crop: Image.Image, min_height: int = 80) -> Image.Image:
    crop = ImageOps.exif_transpose(crop).convert("RGB")
    if crop.height < min_height:
        scale = min_height / max(1, crop.height)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    return ImageOps.autocontrast(crop, cutoff=1)


class EasyOcrBackend:
    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        import easyocr

        self.reader = easyocr.Reader(languages or ["ar", "en"], gpu=gpu, verbose=False)

    def read(self, image: Image.Image, *, field_name: str) -> str:
        import numpy as np

        results = self.reader.readtext(np.array(image.convert("RGB")), detail=0, paragraph=False)
        return " ".join(str(item) for item in results)


class HfVlmOcrBackend:
    """Generic HuggingFace VLM OCR backend.

    This is intentionally configurable because QARI/Baseer/Arabic-GLM model IDs
    vary by release. Pass --ocr-model <hf-model-id>.
    """

    def __init__(self, model_name: str, *, torch_dtype: str = "float16") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = getattr(torch, torch_dtype)
        try:
            from transformers import Qwen2VLForConditionalGeneration

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        except Exception:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def read(self, image: Image.Image, *, field_name: str) -> str:
        import torch

        prompt = FIELD_PROMPTS.get(
            field_name,
            "Read the Arabic printed text in this crop. Return only the field value. Do not explain.",
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(next(self.model.parameters()).device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return out[0] if out else ""


def make_ocr_backend(name: str, *, model_name: str | None = None, gpu: bool = True) -> OcrBackend:
    if name == "easyocr":
        return EasyOcrBackend(gpu=gpu)
    if name == "hf-vlm":
        if not model_name:
            raise ValueError("--ocr-model is required for --ocr-backend hf-vlm")
        return HfVlmOcrBackend(model_name)
    raise ValueError(f"unknown OCR backend: {name}")


def load_class_names(dataset_dir: Path) -> list[str]:
    classes_file = dataset_dir / "classes.txt"
    if classes_file.exists():
        return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise FileNotFoundError(f"missing class list: {classes_file}")


def detect_fields(
    image_path: Path,
    *,
    yolo_model: Any,
    class_names: list[str],
    conf: float = 0.25,
) -> list[Detection]:
    result = yolo_model.predict(str(image_path), conf=conf, verbose=False)[0]
    detections: list[Detection] = []
    if result.boxes is None:
        return detections

    boxes = result.boxes
    for cls_value, conf_value, xyxy_value in zip(boxes.cls, boxes.conf, boxes.xyxy):
        cls_id = int(cls_value.item())
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        class_name = class_names[cls_id]
        if class_name not in CLASS_TO_PATH:
            continue
        xyxy = tuple(float(v) for v in xyxy_value.tolist())
        detections.append(Detection(class_name, float(conf_value.item()), xyxy))
    return detections


def pick_best_by_class(detections: list[Detection]) -> dict[str, Detection]:
    best: dict[str, Detection] = {}
    for det in detections:
        current = best.get(det.class_name)
        if current is None or det.confidence > current.confidence:
            best[det.class_name] = det
    return best


def extract_one(
    image_path: Path,
    *,
    yolo_model: Any,
    class_names: list[str],
    ocr: OcrBackend,
    conf: float = 0.25,
    crops_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    detections = pick_best_by_class(detect_fields(image_path, yolo_model=yolo_model, class_names=class_names, conf=conf))

    raw_record = empty_record(document_id=image_path.stem, source_files=[str(image_path).replace("\\", "/")])
    raw_items: list[dict[str, Any]] = []

    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for field_name, det in sorted(detections.items()):
        crop = enhance_crop(padded_crop(image, det.box_xyxy))
        if crops_dir:
            crop.save(crops_dir / f"{image_path.stem}_{field_name}.png")

        raw_text = ocr.read(crop, field_name=field_name)
        value = normalize_field_text(field_name, raw_text)
        raw_items.append({
            "field": field_name,
            "confidence": det.confidence,
            "box_xyxy": det.box_xyxy,
            "ocr_raw": raw_text,
            "value": value,
        })
        if value is not None:
            set_path(raw_record, CLASS_TO_PATH[field_name], value)

    record = validate_record(
        raw_record,
        document_id=image_path.stem,
        source_files=[str(image_path).replace("\\", "/")],
    )
    if not detections:
        record["review_required"] = True
        record["review_notes"].append("YOLO did not detect any configured birth-certificate fields.")
    return record, raw_items


def run_batch(
    image_paths: list[Path],
    *,
    weights: Path,
    dataset_dir: Path,
    output_dir: Path,
    raw_dir: Path | None,
    crops_dir: Path | None,
    ocr_backend: str,
    ocr_model: str | None,
    conf: float,
    gpu: bool,
) -> list[Path]:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(dataset_dir)
    yolo_model = YOLO(str(weights))
    ocr = make_ocr_backend(ocr_backend, model_name=ocr_model, gpu=gpu)

    written: list[Path] = []
    for image_path in image_paths:
        print(f"[birthcert-yolo-ocr] extracting {image_path.name}", flush=True)
        record, raw_items = extract_one(
            image_path,
            yolo_model=yolo_model,
            class_names=class_names,
            ocr=ocr,
            conf=conf,
            crops_dir=(crops_dir / image_path.stem) if crops_dir else None,
        )
        out_file = output_dir / f"{image_path.stem}.json"
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_file)
        if raw_dir:
            (raw_dir / f"{image_path.stem}.json").write_text(
                json.dumps(raw_items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract birth certificates with YOLO + Arabic OCR.")
    parser.add_argument("--weights", required=True, help="YOLO weights, e.g. outputs/birthcert_yolo/field_detector/weights/best.pt")
    parser.add_argument("--dataset", default="data/birthcert_yolo")
    parser.add_argument("--images", default="data/birthcert_yolo/images/val")
    parser.add_argument("--out", default="outputs/birthcert_yolo_ocr/records")
    parser.add_argument("--raw", default="outputs/birthcert_yolo_ocr/raw")
    parser.add_argument("--crops", default="outputs/birthcert_yolo_ocr/crops")
    parser.add_argument("--ocr-backend", choices=["easyocr", "hf-vlm"], default="easyocr")
    parser.add_argument("--ocr-model", default=None, help="HuggingFace OCR/VLM model ID for QARI/Baseer/Arabic-GLM/etc.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU for OCR backends that support CPU mode.")
    args = parser.parse_args()

    image_root = Path(args.images)
    image_paths = sorted(list(image_root.glob("*.jpeg")) + list(image_root.glob("*.jpg")) + list(image_root.glob("*.png")))
    if not image_paths:
        raise SystemExit(f"no images found in {image_root}")

    written = run_batch(
        image_paths,
        weights=Path(args.weights),
        dataset_dir=Path(args.dataset),
        output_dir=Path(args.out),
        raw_dir=Path(args.raw) if args.raw else None,
        crops_dir=Path(args.crops) if args.crops else None,
        ocr_backend=args.ocr_backend,
        ocr_model=args.ocr_model,
        conf=args.conf,
        gpu=not args.cpu,
    )
    print(f"[birthcert-yolo-ocr] wrote {len(written)} record(s) to {args.out}")


if __name__ == "__main__":
    main()
