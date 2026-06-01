"""Fire the printeddoc inference job server-side, fully decoupled from this machine.

Steps:
  modal deploy printeddoc_modal.py        # once; re-run after code changes
  python printeddoc_infer_trigger.py      # fire-and-forget

Download when done:
  modal volume get --force birthcert-outputs printeddoc_records ./outputs/printeddoc/records
  modal volume get --force birthcert-outputs printeddoc_raw ./outputs/printeddoc/raw
"""

from __future__ import annotations

import modal

APP_NAME      = "printeddoc-parsing"
FUNCTION_NAME = "run_printeddoc"

CALL_KWARGS = dict(
    ids=None,
    limit=None,
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    max_pixels=1_280_000,
    max_new_tokens=2048,
    skip_existing=False,
    enhance_image=True,
    tag="ps_lora",
    adapter_path="/root/project/outputs/adapters/ps_lora",
)


def main() -> None:
    fn     = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    handle = fn.spawn(**CALL_KWARGS)
    print(f"Spawned printeddoc inference job: {handle.object_id}")
    print("Running entirely on Modal L4 — your connection cannot cancel it.")
    print("Download when done:")
    print("  modal volume get --force birthcert-outputs printeddoc_records_ps_lora ./outputs/printeddoc/records_ps_lora")
    print("  modal volume get --force birthcert-outputs printeddoc_raw_ps_lora ./outputs/printeddoc/raw_ps_lora")


if __name__ == "__main__":
    main()
