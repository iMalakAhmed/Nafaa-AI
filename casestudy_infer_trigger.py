"""Spawn the case-study inference job on Modal."""

from __future__ import annotations

import modal

APP_NAME = "casestudy-parsing"
FUNCTION_NAME = "run_casestudy"

CALL_KWARGS = dict(
    ids=None,
    limit=None,
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    max_pixels=1_800_000,
    max_new_tokens=4096,
    skip_existing=False,
    enhance_image=True,
    tag="case_study_lora",
    adapter_path="/root/project/outputs/adapters/case_study_lora",
)


def main() -> None:
    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    handle = fn.spawn(**CALL_KWARGS)
    print(f"Spawned casestudy inference job: {handle.object_id}")
    print("Download when done:")
    print("  modal volume get --force birthcert-outputs casestudy_records_case_study_lora ./outputs/casestudy")
    print("  modal volume get --force birthcert-outputs casestudy_raw_case_study_lora ./outputs/casestudy")


if __name__ == "__main__":
    main()

