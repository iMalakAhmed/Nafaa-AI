"""Fire the birth-certificate job server-side, decoupled from this machine.

Why: on a flaky connection `modal run` dies when the client drops. Deploying the
app once and then `spawn()`-ing the function submits in a single quick call; the
job then runs entirely on Modal and commits each record to the volume regardless
of what happens locally.

  modal deploy birthcert_modal.py        # once (re-run after code changes)
  python birthcert_trigger.py            # fire-and-forget the full 7B batch

Watch progress / download:
  modal volume ls birthcert-outputs records_7b
  modal volume get --force birthcert-outputs records_7b ./outputs/birthcert
"""

from __future__ import annotations

import modal

APP_NAME = "birthcert-parsing"
FUNCTION_NAME = "run_birthcert"

# AIN-7B (MBZUAI) is the strongest open Arabic VLM and the most honest about
# unreadable fields, so it is the default. Swap model_name/tag to compare others.
CALL_KWARGS = dict(
    ids=None,
    limit=None,
    model_name="MBZUAI/AIN",
    max_pixels=2_500_000,
    max_new_tokens=1024,
    skip_existing=False,
    enhance_image=True,
    tag="ain",
)


def main() -> None:
    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)
    handle = fn.spawn(**CALL_KWARGS)
    print(f"Spawned server-side job: {handle.object_id}")
    print("It runs on Modal and commits each record to the volume independently.")
    print("Progress:  modal volume ls birthcert-outputs records_7b")
    print("Download:  modal volume get --force birthcert-outputs records_7b ./outputs/birthcert")


if __name__ == "__main__":
    main()
