"""Deterministic crop regions for case-study forms.

The case-study pages are dense enough that full-page VLM extraction often reads
the layout but misses handwriting. These crops trade global context for larger
text in the most important zones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class Region:
    name: str
    description: str
    box: tuple[float, float, float, float]


FRONT_REGIONS: tuple[Region, ...] = (
    Region("front_header_applicant", "top header, office fields, applicant identity fields", (0.00, 0.00, 1.00, 0.38)),
    Region("front_family_table", "family-member table rows and columns", (0.00, 0.30, 1.00, 0.68)),
    Region("front_housing", "housing section and lower front-page checkboxes", (0.00, 0.58, 1.00, 1.00)),
)

BACK_REGIONS: tuple[Region, ...] = (
    Region("back_upper", "upper back-page housing/assets or checkbox fields", (0.00, 0.00, 1.00, 0.38)),
    Region("back_middle", "social, health, economic, needs, and checkbox middle sections", (0.00, 0.30, 1.00, 0.72)),
    Region("back_lower", "researcher summary, signatures, stamp, and lower checkboxes", (0.00, 0.62, 1.00, 1.00)),
)


def regions_for(page_side: str | None) -> tuple[Region, ...]:
    if page_side == "front":
        return FRONT_REGIONS
    if page_side == "back":
        return BACK_REGIONS
    return FRONT_REGIONS + BACK_REGIONS


def crop_region(image: Image.Image, region: Region) -> Image.Image:
    w, h = image.size
    x0, y0, x1, y1 = region.box
    box = (
        max(0, int(w * x0)),
        max(0, int(h * y0)),
        min(w, int(w * x1)),
        min(h, int(h * y1)),
    )
    return image.crop(box)


def iter_crops(image: Image.Image, page_side: str | None) -> Iterable[tuple[Region, Image.Image]]:
    for region in regions_for(page_side):
        yield region, crop_region(image, region)
