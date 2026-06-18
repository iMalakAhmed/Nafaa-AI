"""Render preview images for the birth-certificate YOLO seed boxes."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "birthcert_yolo"
OUT_DIR = DATASET / "previews"


def load_classes() -> list[str]:
    return (DATASET / "classes.txt").read_text(encoding="utf-8").splitlines()


def draw_one(image_path: Path, label_path: Path, classes: list[str]) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    width, height = img.size

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id_s, x_s, y_s, w_s, h_s = line.split()
        class_id = int(class_id_s)
        x, y, w, h = map(float, (x_s, y_s, w_s, h_s))
        left = (x - w / 2) * width
        top = (y - h / 2) * height
        right = (x + w / 2) * width
        bottom = (y + h / 2) * height
        name = classes[class_id]
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=3)
        draw.text((left + 2, max(0, top - 12)), name, fill=(255, 0, 0), font=font)

    split = image_path.parent.name
    out_split = OUT_DIR / split
    out_split.mkdir(parents=True, exist_ok=True)
    img.save(out_split / image_path.name)


def main() -> None:
    classes = load_classes()
    count = 0
    for split in ("train", "val"):
        for image_path in sorted((DATASET / "images" / split).glob("*.jpeg")):
            label_path = DATASET / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            draw_one(image_path, label_path, classes)
            count += 1
    print(f"Wrote {count} preview image(s) to {OUT_DIR}")


if __name__ == "__main__":
    main()
