"""Egyptian national ID reader based on ebrahimabdelghfar/National-ID-Reader.

The upstream project is a notebook. This module keeps the same pipeline but
returns a stable JSON-compatible dictionary:

1. YOLO finds the card.
2. YOLO finds fields on the cropped card.
3. EasyOCR reads Arabic text/ID digits from each field crop.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve


REPO_RAW = "https://raw.githubusercontent.com/ebrahimabdelghfar/National-ID-Reader/main"
CARD_FINDER_URL = f"{REPO_RAW}/card_finder_seg.pt"
CARD_DIVIDER_URL = f"{REPO_RAW}/card_divider_model.pt"


def _model_dir() -> Path:
    root = os.environ.get("NATIONAL_ID_MODEL_DIR", "/tmp/national_id_reader")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_weights() -> tuple[Path, Path]:
    model_dir = _model_dir()
    finder = model_dir / "card_finder_seg.pt"
    divider = model_dir / "card_divider_model.pt"
    if not finder.exists():
        urlretrieve(CARD_FINDER_URL, finder)
    if not divider.exists():
        urlretrieve(CARD_DIVIDER_URL, divider)
    return finder, divider


def _text(value: Any) -> str | None:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item)
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _digits(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    digits = re.sub(r"\D+", "", text)
    return digits or None


class NationalIdReader:
    def __init__(
        self,
        *,
        finder_model: str | Path | None = None,
        divider_model: str | Path | None = None,
        gpu: bool = True,
    ) -> None:
        import easyocr
        from ultralytics import YOLO

        if finder_model is None or divider_model is None:
            finder_path, divider_path = ensure_weights()
            finder_model = finder_model or finder_path
            divider_model = divider_model or divider_path

        self.card_segmentor = YOLO(str(finder_model))
        self.card_divider = YOLO(str(divider_model))
        self.reader = easyocr.Reader(["ar"], gpu=gpu)

    @staticmethod
    def _crop(frame, bbox: list[float]):
        x1, y1, x2, y2 = bbox[:4]
        return frame[int(y1):int(y2), int(x1):int(x2)]

    def _card_bbox(self, frame) -> list[float] | None:
        result = self.card_segmentor.predict(frame, conf=0.6, verbose=False)
        boxes = result[0].boxes.data.tolist() if result and result[0].boxes is not None else []
        if not boxes:
            return None
        boxes = sorted(boxes, key=lambda item: float(item[4]), reverse=True)
        return boxes[0][:4]

    def _slot_bboxes(self, frame) -> dict[str, list[float]]:
        result = self.card_divider.predict(frame, verbose=False)
        boxes = result[0].boxes.data.tolist() if result and result[0].boxes is not None else []
        slots: dict[str, list[float]] = {}
        for item in boxes:
            label = self.card_divider.names[int(item[5])]
            previous = slots.get(label)
            if previous is None or float(item[4]) > float(previous[4] if len(previous) > 4 else 0):
                slots[label] = item[:5]
        return {key: value[:4] for key, value in slots.items()}

    @staticmethod
    def _filter_text(frame):
        import cv2

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.GaussianBlur(frame, (3, 3), 3)
        frame = cv2.adaptiveThreshold(
            frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2
        )
        return cv2.medianBlur(frame, 3)

    @staticmethod
    def _filter_id(frame):
        import cv2

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.equalizeHist(frame)
        return cv2.inRange(frame, 0, 20)

    def _ocr(self, frame, *, is_id: bool = False) -> list[str]:
        import cv2

        if is_id:
            frame = cv2.resize(frame, (451, 53))
            frame = self._filter_id(frame)
            return self.reader.readtext(
                frame,
                detail=0,
                paragraph=True,
                rotation_info=list(range(0, 270, 10)),
                text_threshold=0.1,
                low_text=0.1,
                link_threshold=0.1,
                canvas_size=10000,
            )

        frame = cv2.resize(frame, (360, 70))
        frame = self._filter_text(frame)
        return self.reader.readtext(frame, detail=0, paragraph=True)

    def extract_image(self, image_path: str | Path, *, document_id: str | None = None) -> dict[str, Any]:
        import cv2

        image_path = Path(image_path)
        doc_id = document_id or image_path.stem
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        record: dict[str, Any] = {
            "document_id": doc_id,
            "document_type": "national_id",
            "source_files": [str(image_path).replace("\\", "/")],
            "full_name": None,
            "first_name": None,
            "remaining_name": None,
            "city": None,
            "governorate": None,
            "national_id": None,
            "raw": {},
            "detected_slots": [],
            "review_required": False,
            "review_notes": [],
        }

        card_bbox = self._card_bbox(frame)
        if card_bbox is None:
            record["review_required"] = True
            record["review_notes"].append("National ID card was not detected")
            return record

        card = self._crop(frame, card_bbox)
        slots = self._slot_bboxes(card)
        record["detected_slots"] = sorted(slots)

        for required in ["firstName", "name", "city", "gov", "idNo"]:
            if required not in slots:
                record["review_required"] = True
                record["review_notes"].append(f"Slot not detected: {required}")

        def read_slot(slot: str, *, is_id: bool = False) -> list[str]:
            bbox = slots.get(slot)
            if bbox is None:
                return []
            crop = self._crop(card, bbox)
            return self._ocr(crop, is_id=is_id)

        first_name = read_slot("firstName")
        remaining_name = read_slot("name")
        city = read_slot("city")
        governorate = read_slot("gov")
        national_id = read_slot("idNo", is_id=True)

        record["raw"] = {
            "firstName": first_name,
            "name": remaining_name,
            "city": city,
            "gov": governorate,
            "idNo": national_id,
        }
        record["first_name"] = _text(first_name)
        record["remaining_name"] = _text(remaining_name)
        record["full_name"] = _text([record["first_name"], record["remaining_name"]])
        record["city"] = _text(city)
        record["governorate"] = _text(governorate)
        record["national_id"] = _digits(national_id)

        if not record["national_id"] or len(record["national_id"]) != 14:
            record["review_required"] = True
            record["review_notes"].append("National ID number was not confidently read as 14 digits")

        return record


def extract_one(image_path: str | Path, *, document_id: str | None = None, gpu: bool = True) -> dict[str, Any]:
    return NationalIdReader(gpu=gpu).extract_image(image_path, document_id=document_id)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    record = extract_one(args.image, gpu=not args.cpu)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
