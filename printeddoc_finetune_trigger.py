"""Fire the printeddoc fine-tune job server-side.

Steps:
  modal deploy printeddoc_finetune_modal.py
  python printeddoc_finetune_trigger.py

Download adapter when done:
  modal volume get --force birthcert-outputs adapters/ps_lora ./outputs/adapters/ps_lora
"""

from __future__ import annotations
import modal

APP_NAME      = "printeddoc-finetune"
FUNCTION_NAME = "run_finetune"

CALL_KWARGS = dict(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    epochs=12, lr=1e-4, lora_r=16, lora_alpha=32,
    max_pixels=1_280_000, max_seq_len=4096,
    adapter_name="ps_lora",
)


def main() -> None:
    fn     = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    handle = fn.spawn(**CALL_KWARGS)
    print(f"Spawned printeddoc fine-tune: {handle.object_id}")
    print("Download when done: modal volume get --force birthcert-outputs adapters/ps_lora ./outputs/adapters/ps_lora")


if __name__ == "__main__":
    main()
