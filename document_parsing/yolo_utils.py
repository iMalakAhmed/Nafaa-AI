"""YOLO-assisted field cropping for document extraction.

When a trained YOLO model is provided, this module detects individual field
regions in the document image and returns PIL crops keyed by field path.
The caller (extractor) uses these crops for per-field VLM extraction instead
of running the VLM over the entire page.

Falls back gracefully: fields YOLO didn't detect are handled by the normal
full-image extraction pass, so the YOLO model doesn't need 100% recall to
be useful.

Usage:
    from yolo_utils import load_yolo, get_field_crops
    model = load_yolo("document_parsing/outputs/yolo/birthcert.pt")
    crops = get_field_crops("BC_001.jpeg", model, class_names)
    # crops = {"personal_and_other.child.name": <PIL Image>, ...}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CROP_PADDING = 15   # pixels added around each detected box
MIN_CONF     = 0.25 # ignore YOLO detections below this confidence


def load_yolo(model_path: str | Path):
    """Load an Ultralytics YOLO model from a .pt file."""
    from ultralytics import YOLO
    return YOLO(str(model_path))


def get_field_crops(
    image_path: str | Path,
    yolo_model,
    class_names: list[str],
    *,
    padding: int = CROP_PADDING,
    min_conf: float = MIN_CONF,
) -> dict[str, Any]:
    """
    Run YOLO detection and return a dict of {field_path: PIL crop}.

    When multiple boxes are detected for the same field, the highest-
    confidence one wins. Boxes below min_conf are discarded.
    """
    from PIL import Image

    results = yolo_model(str(image_path), verbose=False)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    best: dict[str, tuple[Any, float]] = {}  # field_path -> (crop, conf)

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < min_conf or class_id >= len(class_names):
            continue

        field_path = class_names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        crop = img.crop((
            max(0, int(x1) - padding),
            max(0, int(y1) - padding),
            min(w, int(x2) + padding),
            min(h, int(y2) + padding),
        ))

        if field_path not in best or conf > best[field_path][1]:
            best[field_path] = (crop, conf)

    return {k: v[0] for k, v in best.items()}
