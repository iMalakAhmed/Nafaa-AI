"""Spawn the case-study fine-tune job on Modal."""

from __future__ import annotations

import modal

APP_NAME = "casestudy-finetune"
FUNCTION_NAME = "run_finetune"

CALL_KWARGS = dict(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    epochs=14,
    lr=1e-4,
    lora_r=16,
    lora_alpha=32,
    max_pixels=1_800_000,
    max_seq_len=6144,
    adapter_name="case_study_lora",
)


def main() -> None:
    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    handle = fn.spawn(**CALL_KWARGS)
    print(f"Spawned casestudy fine-tune job: {handle.object_id}")
    print("Download when done:")
    print("  modal volume get --force birthcert-outputs adapters/case_study_lora ./outputs/adapters/case_study_lora")


if __name__ == "__main__":
    main()

