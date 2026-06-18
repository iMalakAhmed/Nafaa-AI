"""Two-stage case-study extractor: Arabic OCR model + schema extraction model.

Stage 1 — Arabic handwriting OCR:
    sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2
    Reads each region crop and returns raw Arabic text. This model is fine-tuned
    specifically for Arabic cursive handwriting, so it handles the hard part.

Stage 2 — Schema extraction:
    Qwen/Qwen2.5-VL-3B-Instruct (with or without LoRA adapter)
    Given the full image + the OCR text already extracted, it maps text to JSON
    fields. With the hard OCR work done, the extraction model just needs to
    structure the text — a much easier task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image as _PILImage
from .extract import _merge_crop_raw, _looks_degenerate
from .jsonparse import extract_first_json_object
from .preprocess import prepare
from .prompt import SYSTEM_PROMPT, build_user_instruction
from .regions import iter_crops, regions_for
from .schema import empty_record
from .validate import validate_record

OCR_MODEL = "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2"
EXTRACTION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
OCR_PROMPT = (
    "ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية "
    "بدون اي اختصار ودون زيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:"
)


def _load_ocr_model(model_name: str = OCR_MODEL):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"[casestudy-ocr] loading OCR model: {model_name}", flush=True)
    # Load with bfloat16 + eager attention — the model is already quantized internally.
    # Do NOT pass a BitsAndBytesConfig; that would double-quantize and corrupt weights.
    # Eager attention upcasts softmax to fp32, preventing the "!!!!" NaN collapse.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",   # eager upcasts softmax to fp32, preventing NaN
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


def _resize_for_ocr(image, max_long_side: int = 1024):
    """Shrink to max_long_side so the OCR model's attention matrix fits in VRAM."""
    w, h = image.size
    long = max(w, h)
    if long <= max_long_side:
        return image
    scale = max_long_side / long
    return image.resize((round(w * scale), round(h * scale)), _PILImage.LANCZOS)


def _ocr_image(image, ocr_model, ocr_processor, max_tokens: int = 512) -> str:
    """Run the OCR model on one image and return raw Arabic text."""
    import torch

    image = _resize_for_ocr(image)   # keep attention matrix within VRAM budget
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": OCR_PROMPT},
        ],
    }]
    text = ocr_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ocr_processor(
        text=[text], images=[image], padding=True, return_tensors="pt",
    ).to(next(ocr_model.parameters()).device)

    with torch.inference_mode():
        generated = ocr_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=ocr_processor.tokenizer.eos_token_id,
        )
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    out = ocr_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return (out[0] if out else "").strip()


def _build_instruction_with_ocr(document_id: str, source: str, ocr_text: str) -> str:
    """Full-page instruction enhanced with pre-extracted OCR text as context."""
    base = build_user_instruction(document_id, source)
    ocr_section = (
        "\n\nOCR PRE-EXTRACTED TEXT (from a dedicated Arabic handwriting OCR model):\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{ocr_text}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Use the OCR text above to fill handwritten fields. It was read by an Arabic "
        "cursive specialist model — trust it for field values even if the image looks "
        "unclear. Match each Arabic label on the form to its OCR-extracted value."
    )
    return base + ocr_section


class TwoStageExtractor:
    """Loads the OCR model and the extraction model, runs the two-stage pipeline."""

    def __init__(
        self,
        ocr_model_name: str = OCR_MODEL,
        extraction_model_name: str = EXTRACTION_MODEL,
        *,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_pixels: int = 3_600_000,
        adapter_path: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Stage 1: OCR model (4-bit quantized, small footprint).
        self.ocr_model, self.ocr_processor = _load_ocr_model(ocr_model_name)

        # Stage 2: Extraction model.
        # Use float16 + eager attention to avoid dtype conflicts with the 4-bit OCR model.
        print(f"[casestudy-ocr] loading extraction model: {extraction_model_name}", flush=True)
        self.ext_model = AutoModelForImageTextToText.from_pretrained(
            extraction_model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel
            print(f"[casestudy-ocr] attaching adapter: {adapter_path}", flush=True)
            self.ext_model = PeftModel.from_pretrained(self.ext_model, adapter_path)
        self.ext_model.eval()
        from transformers import AutoProcessor as AP
        self.ext_processor = AP.from_pretrained(
            extraction_model_name, max_pixels=max_pixels, trust_remote_code=True,
        )
        self.max_pixels = max_pixels

    def _generate_ext(self, image, instruction: str, max_new_tokens: int) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]
        text = self.ext_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.ext_processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
            max_pixels=self.max_pixels,
        ).to(next(self.ext_model.parameters()).device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            generated = self.ext_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=self.ext_processor.tokenizer.eos_token_id,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = self.ext_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return out[0] if out else ""

    def extract(
        self,
        image_path: str | Path,
        *,
        document_id: str | None = None,
        max_new_tokens: int = 4096,
        enhance_image: bool = True,
    ) -> tuple[dict[str, Any], str]:
        image_path = Path(image_path)
        doc_id = document_id or image_path.stem
        source = str(image_path).replace("\\", "/")
        full_image = prepare(image_path, enhance_image=enhance_image)

        # ── Stage 1: OCR all regions ─────────────────────────────────────
        print(f"[casestudy-ocr] {doc_id}: running OCR on full page + regions", flush=True)

        # Full-page OCR first to get page_side for crop selection.
        full_ocr = _ocr_image(full_image, self.ocr_model, self.ocr_processor)
        print(f"[casestudy-ocr] {doc_id}: full-page OCR ({len(full_ocr)} chars)", flush=True)

        # Determine page_side from OCR text keywords.
        page_side = None
        if any(h in full_ocr for h in ("بيان بجميع أفراد", "البيانات الأولية", "اسم رب الأسرة")):
            page_side = "front"
        elif any(h in full_ocr for h in ("الحالة الاجتماعية", "الحالة الصحية", "توقيع الباحث", "احتياجات الأسرة")):
            page_side = "back"

        # OCR each crop and concatenate with region labels.
        region_ocr_parts = [f"=== FULL PAGE ===\n{full_ocr}"]
        for region, crop_image in iter_crops(full_image, page_side):
            crop_ocr = _ocr_image(crop_image, self.ocr_model, self.ocr_processor, max_tokens=256)
            region_ocr_parts.append(f"=== {region.name} ({region.description}) ===\n{crop_ocr}")
            print(f"[casestudy-ocr] {doc_id}/{region.name}: {len(crop_ocr)} chars", flush=True)

        combined_ocr = "\n\n".join(region_ocr_parts)

        # ── Stage 2: Extraction with OCR context ─────────────────────────
        instruction = _build_instruction_with_ocr(doc_id, source, combined_ocr)
        try:
            raw_text = self._generate_ext(full_image, instruction, max_new_tokens)
            if _looks_degenerate(raw_text):
                print(f"[casestudy-ocr] {doc_id}: degenerate, retrying ...", flush=True)
                raw_text = self._generate_ext(full_image, instruction, max_new_tokens)
        except Exception as exc:
            record = empty_record(document_id=doc_id, source_files=[source])
            record["review_required"] = True
            record["review_notes"] = [f"extraction failed: {exc!r}"]
            return record, combined_ocr

        try:
            raw_obj = extract_first_json_object(raw_text)
        except ValueError:
            record = empty_record(document_id=doc_id, source_files=[source])
            record["review_required"] = True
            record["review_notes"] = ["extraction output was not valid JSON"]
            return record, combined_ocr + "\n\n--- EXTRACTION ---\n" + raw_text

        full_raw = combined_ocr + "\n\n--- EXTRACTION ---\n" + raw_text
        return validate_record(raw_obj, document_id=doc_id, source_files=[source]), full_raw


def run_batch_ocr(
    image_paths: list[str | Path],
    output_dir: str | Path,
    *,
    ocr_model_name: str = OCR_MODEL,
    extraction_model_name: str = EXTRACTION_MODEL,
    max_pixels: int = 1_800_000,
    max_new_tokens: int = 4096,
    enhance_image: bool = True,
    skip_existing: bool = True,
    raw_dir: str | Path | None = None,
    adapter_path: str | None = None,
    commit_each=None,
) -> list[Path]:
    """Two-phase batch: OCR all images first, then unload OCR model, then extract.

    Never keeps both models in VRAM simultaneously — fits within 22 GB L4.
    """
    import gc
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_dir) if raw_dir else None
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)

    pending = [Path(p) for p in image_paths
               if not (skip_existing and (output_dir / f"{Path(p).stem}.json").exists())]
    if not pending:
        print("[casestudy-ocr] nothing to do.", flush=True)
        return []

    # ── Phase 1: OCR all images (OCR model only in VRAM) ─────────────────
    print(f"[casestudy-ocr] Phase 1 — OCR {len(pending)} image(s)", flush=True)
    ocr_model, ocr_processor = _load_ocr_model(ocr_model_name)

    ocr_results: dict[str, str] = {}
    for image_path in pending:
        full_image = prepare(image_path, enhance_image=enhance_image)
        full_ocr = _ocr_image(full_image, ocr_model, ocr_processor, max_tokens=512)

        page_side = None
        if any(h in full_ocr for h in ("بيان بجميع أفراد", "البيانات الأولية", "اسم رب الأسرة")):
            page_side = "front"
        elif any(h in full_ocr for h in ("الحالة الاجتماعية", "الحالة الصحية", "توقيع الباحث", "احتياجات الأسرة")):
            page_side = "back"

        parts = [f"=== FULL PAGE ===\n{full_ocr}"]
        for region, crop_image in iter_crops(full_image, page_side):
            crop_ocr = _ocr_image(crop_image, ocr_model, ocr_processor, max_tokens=256)
            parts.append(f"=== {region.name} ===\n{crop_ocr}")
            print(f"[casestudy-ocr] {image_path.stem}/{region.name}: {len(crop_ocr)} chars", flush=True)

        ocr_results[image_path.stem] = "\n\n".join(parts)
        print(f"[casestudy-ocr] {image_path.stem}: OCR done ({len(ocr_results[image_path.stem])} chars total)", flush=True)

    # Unload OCR model before loading extraction model.
    del ocr_model, ocr_processor
    gc.collect()
    torch.cuda.empty_cache()
    print("[casestudy-ocr] OCR model unloaded. Loading extraction model …", flush=True)

    # ── Phase 2: Extract JSON using OCR text as context ───────────────────
    from transformers import AutoModelForImageTextToText, AutoProcessor as AP
    ext_model = AutoModelForImageTextToText.from_pretrained(
        extraction_model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",   # consistent with OCR model dtype
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        ext_model = PeftModel.from_pretrained(ext_model, adapter_path)
    ext_model = ext_model.to(torch.float16)  # ensure uniform dtype
    ext_model.eval()
    ext_processor = AP.from_pretrained(
        extraction_model_name, max_pixels=max_pixels, trust_remote_code=True,
    )

    def _generate(image, instruction):
        import torch as _t
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]
        text = ext_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = ext_processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
            max_pixels=max_pixels,
        ).to(next(ext_model.parameters()).device)
        with _t.inference_mode(), _t.autocast("cuda", dtype=_t.float16):
            generated = ext_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=ext_processor.tokenizer.eos_token_id,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = ext_processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return out[0] if out else ""

    written: list[Path] = []
    for idx, image_path in enumerate(pending, start=1):
        print(f"[casestudy-ocr] ({idx}/{len(pending)}) extracting {image_path.name}", flush=True)
        doc_id = image_path.stem
        source = str(image_path).replace("\\", "/")
        full_image = prepare(image_path, enhance_image=enhance_image)
        ocr_text = ocr_results.get(doc_id, "")

        instruction = _build_instruction_with_ocr(doc_id, source, ocr_text)
        try:
            raw_text = _generate(full_image, instruction)
            raw_obj = extract_first_json_object(raw_text)
        except Exception as exc:
            record = empty_record(document_id=doc_id, source_files=[source])
            record["review_required"] = True
            record["review_notes"] = [str(exc)]
            raw_text = ""
            raw_obj = {}

        record = validate_record(raw_obj, document_id=doc_id, source_files=[source]) if raw_obj else \
                 empty_record(document_id=doc_id, source_files=[source])

        out_file = output_dir / f"{doc_id}.json"
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_file)
        if raw_path:
            combined = f"=== OCR ===\n{ocr_text}\n\n=== EXTRACTION ===\n{raw_text}"
            (raw_path / f"{doc_id}.txt").write_text(combined, encoding="utf-8")
        if commit_each:
            commit_each()

    print(f"[casestudy-ocr] wrote {len(written)} record(s) to {output_dir}", flush=True)
    return written
