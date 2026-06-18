"""Generate seed YOLO boxes for birth-certificate field detection.

This is intentionally a seed dataset, not final gold annotation. The certificate
layout is mostly fixed, so normalized template boxes give a useful first pass
that should be reviewed in CVAT, Label Studio, Roboflow, or another labeler.

Run from the repo root:
    python document_parsing/tools/generate_birthcert_yolo_seed.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = ROOT / "data" / "birth_cert_labels"
IMAGES_DIR = ROOT / "data" / "raw_images" / "DataSet" / "Birth Certificate"
OUT_DIR = ROOT / "data" / "birthcert_yolo"


CLASSES = [
    "child_national_id",
    "registration_number",
    "registration_date",
    "issue_date",
    "child_name",
    "child_gender",
    "child_religion",
    "child_nationality",
    "place_of_birth",
    "date_of_birth",
    "father_name",
    "father_religion",
    "father_nationality",
    "mother_name",
    "mother_religion",
    "mother_nationality",
    "health_office",
    "civil_registry",
    "serial_number",
]


# Boxes are YOLO-normalized: x_center, y_center, width, height.
# They target value regions, not whole rows. The coordinates are deliberately
# a little generous because these are seed labels for manual correction.
TEMPLATE_BOXES: dict[str, tuple[float, float, float, float]] = {
    "child_national_id": (0.52, 0.17, 0.38, 0.04),
    "child_nationality": (0.78, 0.28, 0.18, 0.04),
    "place_of_birth": (0.68, 0.32, 0.40, 0.04),
    "date_of_birth": (0.69, 0.35, 0.37, 0.04),
    "child_name": (0.56, 0.405, 0.36, 0.045),
    "child_religion": (0.23, 0.325, 0.20, 0.04),
    "child_gender": (0.22, 0.36, 0.18, 0.04),
    "father_name": (0.56, 0.515, 0.40, 0.045),
    "father_religion": (0.74, 0.485, 0.18, 0.04),
    "father_nationality": (0.76, 0.535, 0.17, 0.04),
    "mother_name": (0.56, 0.635, 0.42, 0.045),
    "mother_religion": (0.74, 0.60, 0.18, 0.04),
    "mother_nationality": (0.76, 0.66, 0.17, 0.04),
    "health_office": (0.63, 0.745, 0.32, 0.045),
    "civil_registry": (0.62, 0.785, 0.34, 0.045),
    "registration_number": (0.36, 0.74, 0.15, 0.04),
    "registration_date": (0.23, 0.78, 0.22, 0.04),
    "issue_date": (0.23, 0.82, 0.22, 0.04),
    "serial_number": (0.60, 0.865, 0.28, 0.045),
}


def field_is_present(record: dict, class_name: str) -> bool:
    mapping = {
        "child_national_id": ("ids", "child_national_id"),
        "registration_number": ("birth_certificate", "registration_number"),
        "registration_date": ("birth_certificate", "registration_date"),
        "issue_date": ("birth_certificate", "issue_date"),
        "health_office": ("birth_certificate", "health_office"),
        "civil_registry": ("birth_certificate", "civil_registry"),
        "serial_number": ("birth_certificate", "serial_number"),
    }
    nested_mapping = {
        "child_name": ("personal_and_other", "child", "name"),
        "child_gender": ("personal_and_other", "child", "gender"),
        "child_religion": ("personal_and_other", "child", "religion"),
        "child_nationality": ("personal_and_other", "child", "nationality"),
        "place_of_birth": ("personal_and_other", "child", "place_of_birth"),
        "date_of_birth": ("personal_and_other", "child", "date_of_birth"),
        "father_name": ("personal_and_other", "father", "name"),
        "father_religion": ("personal_and_other", "father", "religion"),
        "father_nationality": ("personal_and_other", "father", "nationality"),
        "mother_name": ("personal_and_other", "mother", "name"),
        "mother_religion": ("personal_and_other", "mother", "religion"),
        "mother_nationality": ("personal_and_other", "mother", "nationality"),
    }
    keys = mapping.get(class_name) or nested_mapping.get(class_name)
    if not keys:
        return False
    node = record
    for key in keys:
        if not isinstance(node, dict):
            return False
        node = node.get(key)
    return node not in (None, "")


def split_name(doc_id: str) -> str:
    number = int(doc_id.split("_", 1)[1])
    return "val" if number > 32 else "train"


def source_image_for(record: dict, doc_id: str) -> Path | None:
    for source in record.get("source_files") or []:
        path = ROOT / source
        if path.exists():
            return path
    fallback = IMAGES_DIR / f"{doc_id}.jpeg"
    return fallback if fallback.exists() else None


def clamp_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    w = min(max(w, 0.001), 1.0)
    h = min(max(h, 0.001), 1.0)
    x = min(max(x, w / 2), 1.0 - w / 2)
    y = min(max(y, h / 2), 1.0 - h / 2)
    return x, y, w, h


def main() -> None:
    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for label_path in sorted(LABELS_DIR.glob("BC_*.json")):
        record = json.loads(label_path.read_text(encoding="utf-8"))
        doc_id = record.get("document_id") or label_path.stem
        image_path = source_image_for(record, doc_id)
        if image_path is None:
            print(f"[warn] missing image for {doc_id}")
            continue

        split = split_name(doc_id)
        with Image.open(image_path) as img:
            width, height = img.size

        out_image = OUT_DIR / "images" / split / image_path.name
        if not out_image.exists():
            shutil.copy2(image_path, out_image)

        lines: list[str] = []
        for class_name, box in TEMPLATE_BOXES.items():
            if not field_is_present(record, class_name):
                continue
            class_id = CLASSES.index(class_name)
            x, y, w, h = clamp_box(box)
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        (OUT_DIR / "labels" / split / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        rows.append(f"{doc_id}: {split}, {width}x{height}, {len(lines)} boxes")

    (OUT_DIR / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    yaml = [
        f"path: {OUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASSES))
    (OUT_DIR / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} annotated image(s) to {OUT_DIR}")
    for row in rows:
        print("  " + row)
    print("\nReview/correct these seed boxes before serious YOLO training.")


if __name__ == "__main__":
    main()
