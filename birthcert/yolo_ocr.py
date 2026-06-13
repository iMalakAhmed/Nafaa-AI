"""Birth-certificate extraction via YOLO field detection + Arabic OCR.

The detector finds value regions. OCR reads each crop. Rules map OCR text into
the existing birth-certificate JSON schema and validation layer.

Supported OCR backends:
  - easyocr: quick local baseline for Arabic/English printed text.
  - paddleocr: stronger traditional OCR baseline for Arabic printed text.
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

from PIL import Image, ImageFilter, ImageOps

from .schema import empty_record, set_path
from .validate import normalize_digits, validate_record


def _looks_degenerate(text: str, threshold: int = 5) -> bool:
    """Return True if the model output looks like a hallucination or collapse."""
    if not text or not text.strip():
        return True
    # Repeated single character — e.g. "!!!!!" or "......"
    if re.search(r"(.)\1{" + str(threshold) + r",}", text):
        return True
    # Almost no Arabic / digit content → likely garbage
    good = sum(1 for c in text if "؀" <= c <= "ۿ" or c.isdigit() or c in "/-.، ")
    if len(text.strip()) > 4 and good / max(len(text), 1) < 0.15:
        return True
    return False


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
    "child_national_id": "اقرأ رقم الهوية الوطنية في هذه الصورة. أعد الأرقام فقط بدون أي نص آخر.",
    "child_name": "اقرأ اسم الطفل في هذه الصورة. أعد الاسم فقط بدون أي نص آخر.",
    "date_of_birth": "اقرأ تاريخ الميلاد في هذه الصورة. أعد التاريخ فقط بدون أي نص آخر.",
    "place_of_birth": "اقرأ محل الميلاد في هذه الصورة. أعد المكان فقط بدون أي نص آخر.",
    "father_name": "اقرأ اسم الأب في هذه الصورة. أعد الاسم فقط بدون أي نص آخر.",
    "mother_name": "اقرأ اسم الأم في هذه الصورة. أعد الاسم فقط بدون أي نص آخر.",
    "registration_number": "اقرأ رقم القيد في هذه الصورة. أعد الرقم فقط بدون أي نص آخر.",
    "registration_date": "اقرأ تاريخ القيد في هذه الصورة. أعد التاريخ فقط بدون أي نص آخر.",
    "issue_date": "اقرأ تاريخ الإصدار في هذه الصورة. أعد التاريخ فقط بدون أي نص آخر.",
    "serial_number": "اقرأ الرقم التسلسلي في هذه الصورة. أعد الأرقام فقط بدون أي نص آخر.",
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


def _repair_mojibake(text: str) -> str:
    if not text or not re.search(r"[\u0080-\u00ff]", text):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    if re.search(r"[\u0600-\u06ff]", repaired):
        return repaired
    return text


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


def padded_crop(image: Image.Image, box: tuple[float, float, float, float], pad_ratio: float = 0.30) -> Image.Image:
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


def enhance_crop(crop: Image.Image, min_height: int = 180) -> Image.Image:
    crop = ImageOps.exif_transpose(crop).convert("RGB")
    if crop.height < min_height:
        scale = min_height / max(1, crop.height)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    crop = ImageOps.grayscale(crop)
    crop = ImageOps.autocontrast(crop, cutoff=1)
    crop = crop.filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=2))
    return crop.convert("RGB")


class EasyOcrBackend:
    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        import easyocr

        self.reader = easyocr.Reader(languages or ["ar", "en"], gpu=gpu, verbose=False)

    def read(self, image: Image.Image, *, field_name: str) -> str:
        import numpy as np

        results = self.reader.readtext(np.array(image.convert("RGB")), detail=0, paragraph=False)
        return " ".join(str(item) for item in results)


class PaddleOcrBackend:
    def __init__(self, lang: str = "ar") -> None:
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def read(self, image: Image.Image, *, field_name: str) -> str:
        import numpy as np

        outputs = self.ocr.predict(np.array(image.convert("RGB")))
        texts: list[str] = []
        for item in outputs:
            if isinstance(item, dict):
                texts.extend(str(text) for text in item.get("rec_texts", []) if text)
            elif isinstance(item, list):
                for row in item:
                    try:
                        texts.append(str(row[1][0]))
                    except Exception:
                        continue
        return " ".join(texts)


class HfVlmOcrBackend:
    """HuggingFace VLM OCR backend — works with QARI, Qwen2-VL, Qwen2.5-VL.

    Pass --ocr-model <hf-model-id>, e.g. NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct.
    Always loads in bfloat16 (avoids the fp16 "!!!!" collapse on Qwen-family models).
    """

    def __init__(self, model_name: str, *, torch_dtype: str = "bfloat16") -> None:
        import json, os, shutil
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from huggingface_hub import snapshot_download
        from peft import PeftModel

        dtype = getattr(torch, torch_dtype)
        hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

        # QARI is a pure LoRA-adapter repo (no base weights, no config.json).
        # Strategy: load the base Qwen2-VL model in bfloat16, then apply the
        # QARI adapters via PeftModel, then merge — no bitsandbytes needed.
        raw_dir = snapshot_download(model_name, cache_dir=hf_cache)

        # Copy adapter files to a writable temp dir and strip any quantization.
        patched_dir = "/tmp/_hfvlm_noquant"
        if os.path.isdir(patched_dir):
            shutil.rmtree(patched_dir)
        shutil.copytree(raw_dir, patched_dir, symlinks=False)
        for root, _, files in os.walk(patched_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                os.chmod(fp, os.stat(fp).st_mode | 0o644)

        adapter_cfg_path = os.path.join(patched_dir, "adapter_config.json")
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        adapter_cfg.pop("quantization_config", None)
        # QARI was trained on unsloth's 4-bit base; redirect to standard base
        # so we can load in bfloat16 without bitsandbytes.
        base_model_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
        if "unsloth" in base_model_id or "bnb" in base_model_id:
            base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
        adapter_cfg["base_model_name_or_path"] = base_model_id
        with open(adapter_cfg_path, "w") as f:
            json.dump(adapter_cfg, f)

        # Load clean bfloat16 base model (Qwen2-VL-2B-Instruct has no quant config).
        last_exc: Exception | None = None
        for attn_impl in ("sdpa", "eager"):
            try:
                base = AutoModelForImageTextToText.from_pretrained(
                    base_model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    attn_implementation=attn_impl,
                    cache_dir=hf_cache,
                )
                break
            except (ValueError, ImportError) as exc:
                last_exc = exc
        else:
            raise RuntimeError(f"Could not load base model {base_model_id}") from last_exc

        # Apply QARI LoRA adapters and merge for fast inference.
        peft_model = PeftModel.from_pretrained(base, patched_dir)
        self.model = peft_model.merge_and_unload()
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(patched_dir, trust_remote_code=True)

    def read(self, image: Image.Image, *, field_name: str) -> str:
        import torch

        prompt = FIELD_PROMPTS.get(
            field_name,
            "اقرأ النص العربي المطبوع داخل الصورة فقط. أعد قيمة الحقل فقط بدون شرح.",
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
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=5,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        result = (out[0] if out else "").strip()
        return "" if _looks_degenerate(result) else result


def make_ocr_backend(name: str, *, model_name: str | None = None, gpu: bool = True) -> OcrBackend:
    if name == "easyocr":
        return EasyOcrBackend(gpu=gpu)
    if name == "paddleocr":
        return PaddleOcrBackend()
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

        raw_text = _repair_mojibake(ocr.read(crop, field_name=field_name))
        degenerate = _looks_degenerate(raw_text)
        value = normalize_field_text(field_name, raw_text) if not degenerate else None
        raw_items.append({
            "field": field_name,
            "confidence": det.confidence,
            "box_xyxy": det.box_xyxy,
            "ocr_raw": raw_text,
            "degenerate": degenerate,
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
    parser.add_argument("--ocr-backend", choices=["easyocr", "paddleocr", "hf-vlm"], default="easyocr")
    parser.add_argument("--ocr-model", default=None, help="HuggingFace OCR/VLM model ID for QARI/Baseer/Arabic-GLM/etc.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true", help="Disable GPU for OCR backends that support CPU mode.")
    args = parser.parse_args()

    image_root = Path(args.images)
    if image_root.is_file():
        image_paths = [image_root]
    else:
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
