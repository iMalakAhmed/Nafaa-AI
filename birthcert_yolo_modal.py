"""Train/evaluate birth-certificate YOLO + OCR on Modal.

Run:
  modal run birthcert_yolo_modal.py --epochs 100 --imgsz 960 --batch 8

Download:
  modal volume get --force birthcert-outputs birthcert_yolo_ocr ./outputs/birthcert_yolo_ocr
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "birthcert-yolo-ocr"
PROJECT_DIR = "/root/project"
OUTPUT_VOL = "birthcert-outputs"
HF_CACHE_VOL = "case-study-hf-cache"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1")
    .pip_install_from_requirements("requirements-yolo-ocr.txt")
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


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 6,
    volumes={
        f"{PROJECT_DIR}/outputs": output_volume,
        "/root/.cache/huggingface": hf_volume,
    },
)
def run_birthcert_yolo_ocr(
    epochs: int = 100,
    imgsz: int = 960,
    batch: int = 8,
    name: str = "field_detector",
    images: str = "data/birthcert_yolo/images/val",
    output_name: str = "birthcert_yolo_ocr",
    ocr_backend: str = "easyocr",
    ocr_model: str | None = None,
    conf: float = 0.25,
    skip_train: bool = False,
    evaluate: bool = True,
) -> dict:
    import os
    import subprocess
    import sys

    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    weights = Path(PROJECT_DIR) / "outputs" / "birthcert_yolo" / name / "weights" / "best.pt"

    if not skip_train:
        train_cmd = [
            sys.executable,
            "tools/train_birthcert_yolo.py",
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--batch",
            str(batch),
            "--name",
            name,
        ]
        subprocess.run(train_cmd, cwd=PROJECT_DIR, check=True)
        output_volume.commit()

    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    extract_cmd = [
        sys.executable,
        "-m",
        "birthcert.yolo_ocr",
        "--weights",
        str(weights),
        "--images",
        images,
        "--out",
        f"outputs/{output_name}/records",
        "--raw",
        f"outputs/{output_name}/field_reads",
        "--crops",
        f"outputs/{output_name}/crops",
        "--ocr-backend",
        ocr_backend,
        "--conf",
        str(conf),
    ]
    if ocr_backend == "hf-vlm":
        if not ocr_model:
            raise ValueError("ocr_model is required when ocr_backend='hf-vlm'")
        extract_cmd.extend(["--ocr-model", ocr_model])

    extract_proc = subprocess.run(extract_cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    if extract_proc.returncode != 0:
        error_dir = Path(PROJECT_DIR) / "outputs" / output_name
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
        tail = (extract_proc.stderr or extract_proc.stdout)[-4000:]
        raise RuntimeError(f"YOLO/OCR extraction failed. Error log: {error_path}\n{tail}")

    eval_stdout = ""
    ocr_eval_stdout = ""
    if evaluate:
        eval_cmd = [
            sys.executable,
            "-m",
            "birthcert.evaluate",
            "--pred",
            f"outputs/{output_name}/records",
            "--labels",
            "data/birth_cert_labels",
        ]
        eval_proc = subprocess.run(eval_cmd, cwd=PROJECT_DIR, check=True, capture_output=True, text=True)
        ocr_eval_cmd = [
            sys.executable,
            "-m",
            "birthcert.evaluate_yolo_ocr_text",
            "--reads",
            f"outputs/{output_name}/field_reads",
            "--labels",
            "data/birth_cert_labels",
        ]
        ocr_eval_proc = subprocess.run(ocr_eval_cmd, cwd=PROJECT_DIR, check=True, capture_output=True, text=True)
        eval_stdout = eval_proc.stdout
        ocr_eval_stdout = ocr_eval_proc.stdout

    summary = {
        "weights": str(weights),
        "images": images,
        "records_dir": f"outputs/{output_name}/records",
        "field_reads_dir": f"outputs/{output_name}/field_reads",
        "crops_dir": f"outputs/{output_name}/crops",
        "evaluation": eval_stdout,
        "ocr_text_evaluation": ocr_eval_stdout,
    }
    summary_path = Path(PROJECT_DIR) / "outputs" / output_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    output_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    epochs: int = 100,
    imgsz: int = 960,
    batch: int = 8,
    name: str = "field_detector",
    images: str = "data/birthcert_yolo/images/val",
    output_name: str = "birthcert_yolo_ocr",
    ocr_backend: str = "easyocr",
    ocr_model: str = "",
    conf: float = 0.25,
    skip_train: bool = False,
    evaluate: bool = True,
    download: bool = True,
):
    summary = run_birthcert_yolo_ocr.remote(
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        images=images,
        output_name=output_name,
        ocr_backend=ocr_backend,
        ocr_model=ocr_model or None,
        conf=conf,
        skip_train=skip_train,
        evaluate=evaluate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if download:
        import subprocess

        dest = Path("outputs") / output_name
        dest.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["modal", "volume", "get", "--force", OUTPUT_VOL, output_name, str(dest)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            print(f"[birthcert-yolo-ocr] downloaded outputs -> {dest.resolve()}")
        else:
            print("[birthcert-yolo-ocr] download failed:", proc.stderr or proc.stdout)
            print(
                "  download manually: "
                f"modal volume get --force {OUTPUT_VOL} {output_name} ./outputs/{output_name}"
            )
