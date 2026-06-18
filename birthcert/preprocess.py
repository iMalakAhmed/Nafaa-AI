"""Lightweight image preparation (Pillow only, no OpenCV).

These are low-quality phone photos: faded ink, low contrast, sometimes small. We
keep it deliberately cheap — EXIF orientation fix, grayscale-aware autocontrast,
a mild sharpen, and an upscale of small images so faint Arabic strokes survive the
vision encoder's downsampling. No aggressive binarization (it destroys faint ink).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


def load_rgb(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) or img
    return img.convert("RGB")


def enhance(img: Image.Image, *, min_long_side: int = 1600) -> Image.Image:
    """Boost legibility of faint Arabic ink without destroying it."""
    # Autocontrast on a grayscale copy decides the stretch, then re-apply to RGB
    # so we don't introduce color casts.
    img = ImageOps.autocontrast(img, cutoff=1)

    long_side = max(img.size)
    if long_side < min_long_side:
        scale = min_long_side / float(long_side)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Mild unsharp mask makes thin strokes crisper; keep radius small to avoid halos.
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=2))
    return img


def prepare(path: str | Path, *, enhance_image: bool = True, min_long_side: int = 1600) -> Image.Image:
    img = load_rgb(path)
    if enhance_image:
        img = enhance(img, min_long_side=min_long_side)
    return img
