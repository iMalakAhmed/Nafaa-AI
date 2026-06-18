"""Train a YOLO detector for birth-certificate field boxes.

Run from repo root:
    python tools/train_birthcert_yolo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on birth-certificate field boxes.")
    parser.add_argument("--data", default=str(ROOT / "data" / "birthcert_yolo" / "data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint, e.g. yolov8n.pt or yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--project", default=str(ROOT / "outputs" / "birthcert_yolo"))
    parser.add_argument("--name", default="field_detector")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
