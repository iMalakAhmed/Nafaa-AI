"""Benchmark Arabic handwriting/OCR VLMs on case-study forms using Modal.

Examples:
  modal run casestudy_hwr_modal.py --model sherif1313/Arabic-English-handwritten-OCR-v3 --tag arabic_hwr_v3 --limit 3
  modal run casestudy_hwr_modal.py --model NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct --tag qari_v0221
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal


APP_NAME = "casestudy-hwr-benchmark"
PROJECT_DIR = "/root/project"
OUTPUT_VOL = "birthcert-outputs"
HF_CACHE_VOL = "case-study-hf-cache"

HF_MODEL_PRESETS = {
    "hf_hwr": [
        "sherif1313/Arabic-English-handwritten-OCR-v3",
        "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2",
        "sherif1313/Arabic-English-handwritten-OCR-Qwen3-VL-4B",
    ],
    "qari": [
        "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
    ],
    "open_arabic": [
        "sherif1313/Arabic-English-handwritten-OCR-v3",
        "sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2",
        "sherif1313/Arabic-English-handwritten-OCR-Qwen3-VL-4B",
        "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
    ],
}


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements-yolo-ocr.txt")
    .add_local_dir(
        ".",
        remote_path=PROJECT_DIR,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "node_modules",
            "outputs",
        ],
    )
)

output_volume = modal.Volume.from_name(OUTPUT_VOL, create_if_missing=True)
hf_volume = modal.Volume.from_name(HF_CACHE_VOL, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 6,
    volumes={
        f"{PROJECT_DIR}/outputs": output_volume,
        "/root/.cache/huggingface": hf_volume,
    },
)
def run_hwr_benchmark(
    model: str,
    tag: str,
    images: str = "data/raw_images/DataSet/cast study",
    ids: str = "",
    limit: int | None = None,
    fields: str = "",
) -> dict:
    import os
    import subprocess
    import sys

    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    image_root = Path(PROJECT_DIR) / images
    image_paths = sorted(
        list(image_root.glob("*.jpeg")) +
        list(image_root.glob("*.jpg")) +
        list(image_root.glob("*.png"))
    )
    if ids.strip():
        wanted = {item.strip() for item in ids.split(",") if item.strip()}
        image_paths = [path for path in image_paths if path.stem in wanted]
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No images selected from {images}")

    selected_dir = Path(PROJECT_DIR) / "outputs" / f"_casestudy_hwr_selected_{tag}"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for src in image_paths:
        dst = selected_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)

    out_name = f"casestudy_hwr_{tag}"
    records_dir = f"outputs/{out_name}/records"
    raw_dir = f"outputs/{out_name}/raw"

    extract_cmd = [
        sys.executable,
        "-m",
        "casestudy.yolo_ocr",
        "--images",
        str(selected_dir),
        "--model",
        model,
        "--out",
        records_dir,
        "--raw",
        raw_dir,
    ]
    if fields.strip():
        extract_cmd.extend(["--fields", fields])

    extract_proc = subprocess.run(extract_cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    if extract_proc.returncode != 0:
        error_dir = Path(PROJECT_DIR) / "outputs" / out_name
        error_dir.mkdir(parents=True, exist_ok=True)
        error_path = error_dir / "extract_error.txt"
        error_path.write_text(
            "COMMAND:\n"
            + " ".join(extract_cmd)
            + "\n\nSTDOUT:\n"
            + extract_proc.stdout
            + "\n\nSTDERR:\n"
            + extract_proc.stderr,
            encoding="utf-8",
        )
        output_volume.commit()
        raise RuntimeError(f"case-study HWR extraction failed. Error log: {error_path}\n{extract_proc.stderr[-4000:]}")

    eval_cmd = [
        sys.executable,
        "-m",
        "casestudy.evaluate",
        "--pred",
        records_dir,
        "--labels",
        "data/case_study_labels",
    ]
    eval_proc = subprocess.run(eval_cmd, cwd=PROJECT_DIR, check=True, capture_output=True, text=True)

    text_eval_cmd = [
        sys.executable,
        "-m",
        "casestudy.evaluate_qari_text",
        "--reads",
        raw_dir,
        "--labels",
        "data/case_study_labels",
    ]
    text_eval_proc = subprocess.run(text_eval_cmd, cwd=PROJECT_DIR, check=True, capture_output=True, text=True)

    summary = {
        "model": model,
        "tag": tag,
        "documents": [path.stem for path in image_paths],
        "records_dir": records_dir,
        "raw_dir": raw_dir,
        "evaluation": eval_proc.stdout,
        "ocr_text_evaluation": text_eval_proc.stdout,
    }
    summary_path = Path(PROJECT_DIR) / "outputs" / out_name / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_volume.commit()
    return summary


def _safe_model_tag(model: str) -> str:
    return model.replace("/", "_").replace(":", "_").replace(".", "_")


def _models_to_run(model: str, models: str, preset: str) -> list[str]:
    selected: list[str] = []
    if preset:
        if preset not in HF_MODEL_PRESETS:
            available = ", ".join(sorted(HF_MODEL_PRESETS))
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {available}")
        selected.extend(HF_MODEL_PRESETS[preset])
    if models:
        selected.extend(item.strip() for item in models.split(",") if item.strip())
    if model:
        selected.append(model)
    if not selected:
        available = ", ".join(sorted(HF_MODEL_PRESETS))
        raise ValueError(f"Pass --model, --models, or --preset. Available presets: {available}")
    return list(dict.fromkeys(selected))


@app.local_entrypoint()
def main(
    model: str = "",
    models: str = "",
    preset: str = "",
    tag: str = "",
    images: str = "data/raw_images/DataSet/cast study",
    ids: str = "",
    limit: int = 0,
    fields: str = "",
    download: bool = True,
):
    import subprocess

    results: list[dict] = []
    for index, model_id in enumerate(_models_to_run(model, models, preset), start=1):
        safe_tag = tag or _safe_model_tag(model_id)
        if tag and (models or preset):
            safe_tag = f"{safe_tag}_{index}"
        try:
            summary = run_hwr_benchmark.remote(
                model=model_id,
                tag=safe_tag,
                images=images,
                ids=ids,
                limit=limit or None,
                fields=fields,
            )
        except Exception as exc:
            result = {"model": model_id, "tag": safe_tag, "error": str(exc)}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            continue

        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if not download:
            continue

        out_name = f"casestudy_hwr_{safe_tag}"
        dest = Path("outputs") / out_name
        dest.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["modal", "volume", "get", "--force", OUTPUT_VOL, out_name, str(dest)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            print(f"[casestudy-hwr] downloaded outputs -> {dest.resolve()}")
        else:
            print("[casestudy-hwr] download failed:", proc.stderr or proc.stdout)
            print(f"  download manually: modal volume get --force {OUTPUT_VOL} {out_name} ./outputs/{out_name}")

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
