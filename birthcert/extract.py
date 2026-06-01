"""Core birth-certificate extractor: Qwen2.5-VL-3B -> validated JSON record.

Heavy deps (torch / transformers) are imported lazily so this package can be
imported on a plain machine for labeling and evaluation without a GPU stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .jsonparse import extract_first_json_object
from .preprocess import prepare
from .prompt import SYSTEM_PROMPT, build_user_instruction
from .schema import empty_record
from .validate import validate_record

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _looks_degenerate(text: str) -> bool:
    """Detect the Qwen-VL float16 collapse: long runs of one char (usually '!').

    When fp16 attention overflows, logits become NaN and argmax picks token id 0
    ('!'), so the output is a wall of identical punctuation with no JSON at all.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if "{" not in stripped:
        # No JSON object at all; check for a degenerate single-char run.
        compact = stripped.replace(" ", "").replace("\n", "")
        if compact and len(set(compact)) <= 2 and len(compact) >= 20:
            return True
        return True  # no JSON object => unusable anyway
    # Even with a brace, a huge run of one punctuation char means it collapsed.
    import re as _re

    return bool(_re.search(r"(.)\1{40,}", stripped))


class BirthCertExtractor:
    """Loads the vision model once, then extracts many certificates."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        torch_dtype: str = "float16",  # T4 has no bf16 tensor cores
        # eager attention upcasts the softmax to fp32, which avoids the fp16
        # overflow that makes Qwen2.5-VL collapse into "!!!!" output on a T4.
        attn_implementation: str = "eager",
        max_pixels: int = 1_280_000,
        min_pixels: int = 256 * 28 * 28,
        device_map: str = "auto",
        adapter_path: str | None = None,
    ) -> None:
        import torch
        # AutoModelForImageTextToText picks the right class for both Qwen2.5-VL
        # (our default) and Qwen2-VL based models like MBZUAI/AIN, so the same
        # extractor can run an Arabic-specialized model with no code change.
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            torch_dtype
        ]
        self.model_name = model_name
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        print(f"[birthcert] loading {model_name} ({torch_dtype}, attn={attn_implementation}) …", flush=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True,
        )
        # Load a fine-tuned LoRA adapter on top of the base model, if provided.
        if adapter_path:
            from peft import PeftModel

            print(f"[birthcert] attaching LoRA adapter: {adapter_path}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            trust_remote_code=True,
        )

    def _generate(
        self,
        image,
        instruction: str,
        max_new_tokens: int,
        *,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 0,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,  # deterministic: no creative guessing
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out[0] if out else ""

    def extract(
        self,
        image_path: str | Path,
        *,
        document_id: str | None = None,
        max_new_tokens: int = 2048,
        enhance_image: bool = True,
    ) -> tuple[dict[str, Any], str]:
        """Return (validated_record, raw_model_text) for one certificate image."""
        image_path = Path(image_path)
        doc_id = document_id or image_path.stem
        source = str(image_path).replace("\\", "/")
        image = prepare(image_path, enhance_image=enhance_image)
        instruction = build_user_instruction(doc_id, source)
        try:
            raw_text = self._generate(image, instruction, max_new_tokens)
            if _looks_degenerate(raw_text):
                # Break the repeat loop and try once more with a stronger penalty.
                print(f"[birthcert] {doc_id}: degenerate output, retrying …", flush=True)
                raw_text = self._generate(
                    image,
                    instruction,
                    max_new_tokens,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )
        except Exception as exc:  # OOM / runtime — record, do not crash the batch
            record = empty_record(document_id=doc_id, source_files=[source])
            record["review_required"] = True
            record["review_notes"] = [f"generation failed: {exc!r}"]
            return record, ""

        try:
            raw_obj = extract_first_json_object(raw_text)
        except ValueError:
            record = empty_record(document_id=doc_id, source_files=[source])
            record["review_required"] = True
            record["review_notes"] = ["model output was not valid JSON"]
            return record, raw_text

        record = validate_record(raw_obj, document_id=doc_id, source_files=[source])
        return record, raw_text


def run_batch(
    image_paths: list[str | Path],
    output_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    max_pixels: int = 1_280_000,
    max_new_tokens: int = 2048,
    enhance_image: bool = True,
    skip_existing: bool = True,
    raw_dir: str | Path | None = None,
    adapter_path: str | None = None,
    commit_each: Any = None,
) -> list[str]:
    """Extract every image, writing one validated JSON per certificate.

    `commit_each`, if given, is called after each file is written (used on Modal to
    persist the volume incrementally so a dropped client never loses progress).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_dir) if raw_dir else None
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)

    pending = []
    for p in image_paths:
        p = Path(p)
        out_file = output_dir / f"{p.stem}.json"
        if skip_existing and out_file.exists():
            print(f"[birthcert] skip existing {out_file.name}", flush=True)
            continue
        pending.append(p)

    if not pending:
        print("[birthcert] nothing to do (all outputs exist).", flush=True)
        return []

    extractor = BirthCertExtractor(
        model_name=model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        max_pixels=max_pixels,
        adapter_path=adapter_path,
    )

    written: list[str] = []
    for idx, image_path in enumerate(pending, start=1):
        print(f"[birthcert] ({idx}/{len(pending)}) {image_path.name}", flush=True)
        record, raw_text = extractor.extract(
            image_path, max_new_tokens=max_new_tokens, enhance_image=enhance_image
        )
        out_file = output_dir / f"{image_path.stem}.json"
        out_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(str(out_file))
        if raw_path:
            (raw_path / f"{image_path.stem}.txt").write_text(raw_text, encoding="utf-8")
        if callable(commit_each):
            commit_each()
    print(f"[birthcert] wrote {len(written)} record(s) to {output_dir}", flush=True)
    return written
