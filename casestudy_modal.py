"""Case-study form extractor on Modal.

Run all:
  modal deploy casestudy_modal.py
  python casestudy_infer_trigger.py
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "casestudy-parsing"
PROJECT_DIR = "/root/project"
OUTPUT_VOL = "birthcert-outputs"
HF_CACHE_VOL = "case-study-hf-cache"
IMAGES_SUBDIR = "data/raw_images/DataSet/cast study"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("torchvision", "peft>=0.12.0")
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[".venv", ".git", "__pycache__", ".pytest_cache", "outputs", "notebooks", ".modal", "agent-tools"],
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


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 5,
    env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    volumes={
        f"{PROJECT_DIR}/outputs": output_volume,
        "/root/.cache/huggingface": hf_volume,
    },
)
def run_casestudy(
    ids: list[str] | None = None,
    limit: int | None = None,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_pixels: int = 1_800_000,
    max_new_tokens: int = 4096,
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

    from casestudy.extract import run_batch

    images = _select_images(ids, limit)
    records_name = f"casestudy_records_{tag}" if tag else "casestudy_records"
    raw_name = f"casestudy_raw_{tag}" if tag else "casestudy_raw"
    out_dir = Path(PROJECT_DIR) / "outputs" / records_name
    raw_dir = Path(PROJECT_DIR) / "outputs" / raw_name
    print(f"[casestudy] {len(images)} image(s) selected.", flush=True)
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
    limit: int = 0,
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    max_pixels: int = 1_800_000,
    max_new_tokens: int = 4096,
    skip_existing: bool = True,
    enhance_image: bool = True,
    tag: str = "",
    adapter_path: str = "",
):
    id_list = [s.strip() for s in ids.split(",") if s.strip()] or None
    cap = None if limit <= 0 else limit
    written = run_casestudy.remote(
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
    print(f"[casestudy] wrote {len(written)} record(s): {written}")

