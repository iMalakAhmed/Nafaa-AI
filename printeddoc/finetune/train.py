"""QLoRA fine-tuning of Qwen2.5-VL-3B for printed social-insurance document extraction.

Identical training loop to birthcert/finetune/train.py — only imports differ.
Run via printeddoc_finetune_modal.py on Modal A10G.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 12,
    lr: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_pixels: int = 1_280_000,
    min_pixels: int = 256 * 28 * 28,
    grad_accum: int = 4,
    max_seq_len: int = 4096,
) -> str:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from PIL import Image, ImageOps
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _read_jsonl(data_dir / "train.jsonl")
    if not train_rows:
        raise SystemExit(f"No training rows in {data_dir/'train.jsonl'}. Label some documents first.")
    print(f"[ft] {len(train_rows)} training example(s).", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    ))
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(
        base_model, max_pixels=max_pixels, min_pixels=min_pixels, trust_remote_code=True,
    )
    device = next(model.parameters()).device
    eos    = processor.tokenizer.eos_token or "<|im_end|>"

    def load_image(path: str) -> Image.Image:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img) or img
        return img.convert("RGB")

    def build_example(row: dict[str, Any]):
        image       = load_image(row["image"])
        messages    = [
            {"role": "system", "content": row.get("system", "")},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": row["instruction"]}]},
        ]
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text   = prompt_text + row["target"] + eos
        full        = processor(text=[full_text], images=[image], return_tensors="pt", max_pixels=max_pixels, min_pixels=min_pixels)
        prompt_only = processor(text=[prompt_text], images=[image], return_tensors="pt", max_pixels=max_pixels, min_pixels=min_pixels)
        prompt_len  = prompt_only["input_ids"].shape[1]
        labels      = full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[full["input_ids"] == pad_id] = -100
        full["labels"] = labels
        if full["input_ids"].shape[1] > max_seq_len:
            return None
        return {k: v.to(device) for k, v in full.items()}

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    total_steps = max(1, math.ceil(len(train_rows) * epochs / grad_accum))
    sched = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

    model.train()
    step = total_skipped = 0
    for epoch in range(epochs):
        running = seen = skipped = 0
        optim.zero_grad()
        for i, row in enumerate(train_rows):
            batch = build_example(row)
            if batch is None:
                skipped += 1; total_skipped += 1; continue
            out  = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running += out.loss.item(); seen += 1
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step(); sched.step(); optim.zero_grad(); step += 1
        avg      = running / max(1, seen)
        skip_pct = 100.0 * skipped / max(1, len(train_rows))
        print(f"[ft] epoch {epoch+1}/{epochs}  avg_loss={avg:.4f}  skipped={skipped}/{len(train_rows)} ({skip_pct:.0f}%)", flush=True)
        if skip_pct > 5.0 and epoch == 0:
            print(f"[ft] WARNING: >{skip_pct:.0f}% exceed max_seq_len={max_seq_len}. Consider increasing to 5120.", flush=True)

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"[ft] saved LoRA adapter -> {output_dir}", flush=True)
    return str(output_dir)
