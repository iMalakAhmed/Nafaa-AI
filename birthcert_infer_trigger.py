"""Fire the birthcert inference job server-side, fully decoupled from this machine.

Steps:
  modal deploy birthcert_modal.py        # once; re-run after code changes
  python birthcert_infer_trigger.py      # fire-and-forget

Monitor:
  modal app logs <app-id>
  modal volume ls birthcert-outputs records_ft_lora

Download results when done:
  modal volume get --force birthcert-outputs records_ft_lora ./outputs/birthcert/records_ft_lora
  modal volume get --force birthcert-outputs raw_ft_lora ./outputs/birthcert/raw_ft_lora
"""

from __future__ import annotations

import modal

APP_NAME = "birthcert-parsing"
FUNCTION_NAME = "run_birthcert"

CALL_KWARGS = dict(
    ids=["BC_00001"],
    limit=None,
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    max_pixels=1_280_000,
    max_new_tokens=2048,
    skip_existing=False,
    enhance_image=True,
    tag="ft_lora",
    adapter_path="/root/project/outputs/adapters/bc_lora",
)


def main() -> None:
    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    handle = fn.spawn(**CALL_KWARGS)
    print(f"Spawned inference job: {handle.object_id}")
    print("Running entirely on Modal L4 — your connection cannot cancel it.")
    print("Download when done:")
    print("  modal volume get --force birthcert-outputs records_ft_lora/BC_00001.json ./outputs/birthcert/BC_00001.json")
    print("  modal volume get --force birthcert-outputs raw_ft_lora/BC_00001.txt ./outputs/birthcert/BC_00001_raw.txt")


if __name__ == "__main__":
    main()
