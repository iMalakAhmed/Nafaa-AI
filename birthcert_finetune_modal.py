"""
Fine-tune the birth-certificate extractor (QLoRA on Qwen2.5-VL-3B) on Modal.

Steps:
  1. Label certificates:   python -m birthcert.make_labels --count 40
     (fill in data/birth_cert_labels/*.json with the correct values)
  2. Build the dataset:     python -m birthcert.finetune.prepare_data
  3. Train on Modal:        modal run birthcert_finetune_modal.py --epochs 12
  4. Download the adapter:  modal volume get --force birthcert-outputs adapters/bc_lora ./outputs/adapters/bc_lora
  5. Run inference with it: modal run birthcert_modal.py --limit 0 --adapter-path /root/project/outputs/adapters/bc_lora

The dataset (data/birth_cert_ft/*.jsonl) is uploaded with the code mount.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "birthcert-finetune"
PROJECT_DIR = "/root/project"
OUTPUT_VOL = "birthcert-outputs"
HF_CACHE_VOL = "case-study-hf-cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-finetune.txt")
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[
            ".venv", ".git", "__pycache__", ".pytest_cache",
            "outputs", "notebooks", ".modal", "agent-tools",
        ],
    )
)

output_volume = modal.Volume.from_name(OUTPUT_VOL, create_if_missing=True)
hf_volume = modal.Volume.from_name(HF_CACHE_VOL, create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 4,
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    volumes={
        f"{PROJECT_DIR}/outputs": output_volume,
        "/root/.cache/huggingface": hf_volume,
    },
)
def run_finetune(
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    epochs: int = 12,
    lr: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_pixels: int = 1_280_000,
    max_seq_len: int = 4096,
    adapter_name: str = "bc_lora",
) -> str:
    import os
    import sys

    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from birthcert.finetune.train import train

    data_dir = Path(PROJECT_DIR) / "data" / "birth_cert_ft"
    out_dir = Path(PROJECT_DIR) / "outputs" / "adapters" / adapter_name
    result = train(
        data_dir,
        out_dir,
        base_model=base_model,
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_pixels=max_pixels,
        max_seq_len=max_seq_len,
    )
    output_volume.commit()
    return result


@app.local_entrypoint()
def main(
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    epochs: int = 12,
    lr: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_pixels: int = 1_280_000,
    max_seq_len: int = 4096,
    adapter_name: str = "bc_lora",
):
    data_dir = Path("data/birth_cert_ft")
    if not (data_dir / "train.jsonl").exists():
        print("No data/birth_cert_ft/train.jsonl found.")
        print("Run first:  python -m birthcert.finetune.prepare_data")
        return
    print(f"[ft] submitting fine-tune to Modal A10G (base={base_model}, epochs={epochs}) …")
    result = run_finetune.remote(
        base_model=base_model,
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_pixels=max_pixels,
        max_seq_len=max_seq_len,
        adapter_name=adapter_name,
    )
    print(f"[ft] adapter saved on volume at: {result}")
    print(f"Download: modal volume get --force {OUTPUT_VOL} adapters/{adapter_name} ./outputs/adapters/{adapter_name}")
