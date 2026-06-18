"""
Birth-certificate extractor on Modal (lightweight: Qwen2.5-VL-3B on a T4).

Run a few:
  modal run birthcert_modal.py --limit 5

Run everything (detached, survives a dropped connection):
  modal run --detach birthcert_modal.py --limit 0

Specific ids:
  modal run birthcert_modal.py --ids "BC_001,BC_002,BC_003"

Download results:
  modal volume get --force birthcert-outputs records ./outputs/birthcert
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "birthcert-parsing"
PROJECT_DIR = "/root/project"
OUTPUT_VOL = "birthcert-outputs"
HF_CACHE_VOL = "case-study-hf-cache"  # share the existing model cache
IMAGES_SUBDIR = "data/raw_images/DataSet/Birth Certificate"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("torchvision", "peft>=0.12.0")
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[
            ".venv",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "outputs",
            "notebooks",
            ".modal",
            "agent-tools",
        ],
    )
)

output_volume = modal.Volume.from_name(OUTPUT_VOL, create_if_missing=True)
hf_volume = modal.Volume.from_name(HF_CACHE_VOL, create_if_missing=True)

app = modal.App(APP_NAME)


def _select_images(ids: list[str] | None, limit: int | None) -> list[Path]:
    root = Path(PROJECT_DIR) / IMAGES_SUBDIR
    all_images = sorted(root.glob("*.jpeg")) + sorted(root.glob("*.jpg")) + sorted(root.glob("*.png"))
    if ids:
        wanted = {i.strip() for i in ids if i.strip()}
        all_images = [p for p in all_images if p.stem in wanted]
    if limit and limit > 0:
        all_images = all_images[:limit]
    return all_images


# L4: native bfloat16 (unlike the T4), which Qwen2.5-VL needs to stay numerically
# stable. Without bf16 the fp16 vision tower NaNs out and the model emits "!!!!".
@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 4,
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    volumes={
        f"{PROJECT_DIR}/outputs": output_volume,
        "/root/.cache/huggingface": hf_volume,
    },
)
def run_birthcert(
    ids: list[str] | None = None,
    limit: int | None = None,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_pixels: int = 1_280_000,
    max_new_tokens: int = 2048,
    skip_existing: bool = True,
    enhance_image: bool = True,
    tag: str = "",
    adapter_path: str | None = None,
) -> list[str]:
    import os
    import sys

    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    from birthcert.extract import run_batch

    images = _select_images(ids, limit)
    print(f"[birthcert] {len(images)} image(s) selected.", flush=True)
    # `tag` lets a stronger run (e.g. 7B) write to a separate folder so results can
    # be compared against the baseline instead of overwriting it.
    records_name = f"records_{tag}" if tag else "records"
    raw_name = f"raw_{tag}" if tag else "raw"
    out_dir = Path(PROJECT_DIR) / "outputs" / records_name
    raw_dir = Path(PROJECT_DIR) / "outputs" / raw_name

    written = run_batch(
        images,
        out_dir,
        model_name=model_name,
        torch_dtype="bfloat16",
        attn_implementation="sdpa",
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        enhance_image=enhance_image,
        skip_existing=skip_existing,
        raw_dir=raw_dir,
        adapter_path=adapter_path,
        commit_each=output_volume.commit,
    )
    output_volume.commit()
    return [str(Path(p).name) for p in written]


@app.local_entrypoint()
def main(
    ids: str = "",
    limit: int = 5,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_pixels: int = 1_280_000,
    max_new_tokens: int = 2048,
    skip_existing: bool = True,
    enhance_image: bool = True,
    tag: str = "",
    adapter_path: str = "",
    download: bool = True,
):
    id_list = [s.strip() for s in ids.split(",") if s.strip()] or None
    cap = None if limit <= 0 else limit
    print(f"[birthcert] submitting to Modal L4 (model={model_name}, limit={limit}, tag={tag!r}) …")
    written = run_birthcert.remote(
        ids=id_list,
        limit=cap,
        model_name=model_name,
        max_pixels=max_pixels,
        max_new_tokens=max_new_tokens,
        skip_existing=skip_existing,
        enhance_image=enhance_image,
        tag=tag,
        adapter_path=adapter_path or None,
    )
    print(f"[birthcert] wrote {len(written)} record(s): {written}")

    if download:
        import subprocess

        records_name = f"records_{tag}" if tag else "records"
        dest = Path("outputs") / "birthcert"
        dest.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["modal", "volume", "get", "--force", OUTPUT_VOL, records_name, str(dest)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            print(f"[birthcert] downloaded {records_name} -> {dest.resolve()}")
        else:
            print("[birthcert] download failed:", proc.stderr or proc.stdout)
            print(f"  download manually: modal volume get --force {OUTPUT_VOL} {records_name} ./outputs/birthcert")
