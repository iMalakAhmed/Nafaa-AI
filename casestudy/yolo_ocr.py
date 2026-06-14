"""Case-study extraction via YOLO (optional) + QARI / HF VLM OCR.

Two modes depending on whether a YOLO model is available:
  - YOLO mode : YOLO crops field regions → VLM reads each crop.
  - Full-page  : VLM reads the whole page once, asking for every field in one pass.

Since a YOLO model for case studies is not yet trained, full-page mode is the default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .schema import FIELD_LABELS_AR, SCALAR_FIELD_PATHS, empty_record, get_path, set_path
from .validate import normalize_digits, validate_record


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _looks_degenerate(text: str, threshold: int = 5) -> bool:
    if not text or not text.strip():
        return True
    if re.search(r"(.)\1{" + str(threshold) + r",}", text):
        return True
    good = sum(1 for c in text if "؀" <= c <= "ۿ" or c.isdigit() or c in "/-.، ")
    if len(text.strip()) > 4 and good / max(len(text), 1) < 0.15:
        return True
    return False


def _norm_digits(text: str) -> str:
    return normalize_digits(text or "")


def _clean(text: str | None) -> str | None:
    if not text:
        return None
    text = _norm_digits(text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;،")
    return text or None


# --------------------------------------------------------------------------- #
# Full-page VLM backend
# --------------------------------------------------------------------------- #

_FULL_PAGE_SYSTEM = (
    "أنت نظام استخراج بيانات. يُعطى لك صورة لنموذج بحث اجتماعي عربي مطبوع. "
    "استخرج القيم المطلوبة من النموذج. أجب بقيم مختصرة بدون شرح."
)


def _build_field_prompts() -> dict[str, str]:
    prompts: dict[str, str] = {}
    for path in SCALAR_FIELD_PATHS:
        labels = FIELD_LABELS_AR.get(path, [path])
        label_str = " / ".join(labels)
        prompts[path] = (
            "Read the Arabic form image and extract this field only.\n"
            f"Field path: {path}\n"
            f"Field label: {label_str}\n"
            "Return only the field value. Do not include labels, explanations, bullets, or JSON. "
            "If the value is not visible, return exactly: NOT_FOUND"
        )
    return prompts


FIELD_PROMPTS = _build_field_prompts()

_NULL_ANSWERS = {"", "-", "--", "N/A", "NA", "NOT_FOUND", "not found", "Not found", "لا يوجد", "لا يوجد.", "غير محدد", "غير متوفر", "لا توجد"}


class HfVlmBackend:
    """Loads any Qwen2-VL / Qwen2.5-VL / QARI model in bfloat16."""

    def __init__(self, model_name: str, *, torch_dtype: str = "bfloat16") -> None:
        import json
        import os
        import shutil
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from huggingface_hub import snapshot_download

        dtype = getattr(torch, torch_dtype)
        hf_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        raw_dir = snapshot_download(model_name, cache_dir=hf_cache)

        def load_model(target: str):
            last_exc: Exception | None = None
            for kwargs in (
                {"attn_implementation": "sdpa"},
                {"attn_implementation": "eager"},
                {},
            ):
                try:
                    return AutoModelForImageTextToText.from_pretrained(
                        target,
                        torch_dtype=dtype,
                        device_map="auto",
                        trust_remote_code=True,
                        cache_dir=hf_cache,
                        **kwargs,
                    )
                except Exception as exc:
                    last_exc = exc
            raise RuntimeError(f"Could not load HF VLM model {target}") from last_exc

        if os.path.exists(os.path.join(raw_dir, "adapter_config.json")):
            from peft import PeftModel

            patched_dir = "/tmp/_casestudy_hfvlm_noquant"
            if os.path.isdir(patched_dir):
                shutil.rmtree(patched_dir)
            shutil.copytree(raw_dir, patched_dir, symlinks=False)
            for root, _, files in os.walk(patched_dir):
                for fn in files:
                    fp = os.path.join(root, fn)
                    os.chmod(fp, os.stat(fp).st_mode | 0o644)

            adapter_cfg_path = os.path.join(patched_dir, "adapter_config.json")
            with open(adapter_cfg_path, encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            adapter_cfg.pop("quantization_config", None)
            base_model_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
            if "unsloth" in base_model_id or "bnb" in base_model_id:
                base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
            adapter_cfg["base_model_name_or_path"] = base_model_id
            with open(adapter_cfg_path, "w", encoding="utf-8") as f:
                json.dump(adapter_cfg, f)

            base = load_model(base_model_id)
            peft_model = PeftModel.from_pretrained(base, patched_dir)
            self.model = peft_model.merge_and_unload()
            processor_source = patched_dir
        else:
            self.model = load_model(model_name)
            processor_source = model_name

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, cache_dir=hf_cache)
        self._device = next(self.model.parameters()).device

    def ask(self, image: Image.Image, prompt: str, *, max_new_tokens: int = 80) -> str:
        import torch

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _FULL_PAGE_SYSTEM}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self._device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=5,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        result = (out[0] if out else "").strip()
        return "" if _looks_degenerate(result) else result


# --------------------------------------------------------------------------- #
# Full-page extraction
# --------------------------------------------------------------------------- #

def extract_fullpage(
    image_path: Path,
    *,
    vlm: HfVlmBackend,
    fields: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    target_fields = fields or SCALAR_FIELD_PATHS
    record = empty_record(document_id=image_path.stem, source_files=[str(image_path).replace("\\", "/")])
    raw_items: list[dict[str, Any]] = []

    for field_path in target_fields:
        prompt = FIELD_PROMPTS.get(field_path)
        if not prompt:
            continue
        raw_answer = vlm.ask(image, prompt)
        degenerate = _looks_degenerate(raw_answer)
        value: str | None = None
        if not degenerate and raw_answer.strip() not in _NULL_ANSWERS:
            value = _clean(raw_answer)
        raw_items.append({
            "field": field_path,
            "ocr_raw": raw_answer,
            "degenerate": degenerate,
            "value": value,
        })
        if value is not None:
            set_path(record, field_path, value)

    record = validate_record(
        record,
        document_id=image_path.stem,
        source_files=[str(image_path).replace("\\", "/")],
    )
    return record, raw_items


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #

def run_batch(
    image_paths: list[Path],
    *,
    model_name: str,
    output_dir: Path,
    raw_dir: Path | None,
    fields: list[str] | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[casestudy-qari] Loading {model_name} …", flush=True)
    vlm = HfVlmBackend(model_name)

    written: list[Path] = []
    for image_path in image_paths:
        print(f"[casestudy-qari] extracting {image_path.name}", flush=True)
        record, raw_items = extract_fullpage(image_path, vlm=vlm, fields=fields)
        out_file = output_dir / f"{image_path.stem}.json"
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_file)
        if raw_dir:
            (raw_dir / f"{image_path.stem}.json").write_text(
                json.dumps(raw_items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract case-study forms with QARI / HF VLM (full-page).")
    parser.add_argument("--images", default="data/raw_images/DataSet/cast study")
    parser.add_argument("--model", default="NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct")
    parser.add_argument("--out", default="outputs/casestudy_qari/records")
    parser.add_argument("--raw", default="outputs/casestudy_qari/raw")
    parser.add_argument("--fields", default=None, help="Comma-separated field paths to extract (default: all).")
    args = parser.parse_args()

    image_root = Path(args.images)
    if image_root.is_file():
        image_paths = [image_root]
    else:
        image_paths = sorted(
            list(image_root.glob("*.jpeg")) +
            list(image_root.glob("*.jpg")) +
            list(image_root.glob("*.png"))
        )
    if not image_paths:
        raise SystemExit(f"No images found in {image_root}")

    fields = [f.strip() for f in args.fields.split(",")] if args.fields else None

    written = run_batch(
        image_paths,
        model_name=args.model,
        output_dir=Path(args.out),
        raw_dir=Path(args.raw) if args.raw else None,
        fields=fields,
    )
    print(f"[casestudy-qari] wrote {len(written)} record(s) to {args.out}")


if __name__ == "__main__":
    main()
