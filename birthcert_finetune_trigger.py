"""Fire the fine-tune job server-side, fully decoupled from this machine.

Steps:
  modal deploy birthcert_finetune_modal.py   # once; re-run after code changes
  python birthcert_finetune_trigger.py        # fire-and-forget

Monitor:
  modal app logs <app-id>
  modal volume ls birthcert-outputs adapters/bc_lora

Download adapter when done:
  modal volume get --force birthcert-outputs adapters/bc_lora ./outputs/adapters/bc_lora
"""

from __future__ import annotations

import modal

APP_NAME = "birthcert-finetune"
FUNCTION_NAME = "run_finetune"

CALL_KWARGS = dict(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    epochs=12,
    lr=1e-4,
    lora_r=16,
    lora_alpha=32,
    max_pixels=1_280_000,
    max_seq_len=4096,
    adapter_name="bc_lora",
)


def main() -> None:
    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    handle = fn.spawn(**CALL_KWARGS)
    print(f"Spawned fine-tune job: {handle.object_id}")
    print("Running entirely on Modal A10G — your connection cannot cancel it.")
    print("Monitor: modal app logs --follow $(modal app list | grep birthcert-finetune | head -1 | awk '{print $2}')")
    print("Download when done: modal volume get --force birthcert-outputs adapters/bc_lora ./outputs/adapters/bc_lora")


if __name__ == "__main__":
    main()
