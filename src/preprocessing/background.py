from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageFilter


def remove_blue_screen(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([85, 45, 35]), np.array([140, 255, 255]))
    mask = cv2.medianBlur(mask, 5)
    alpha = Image.fromarray(255 - mask).filter(ImageFilter.GaussianBlur(radius=1.2))
    result = image.convert("RGBA")
    result.putalpha(alpha)
    return result


def remove_with_rembg(image: Image.Image) -> Image.Image:
    from rembg import remove

    result = remove(image.convert("RGBA"))
    return result if isinstance(result, Image.Image) else Image.open(BytesIO(result)).convert("RGBA")


def remove_background(image: Image.Image, backend: str) -> Image.Image:
    if backend == "none":
        return image.convert("RGBA")
    if backend == "blue_screen":
        return remove_blue_screen(image)
    if backend == "rembg":
        return remove_with_rembg(image)
    raise ValueError(f"Unknown background backend: {backend}")


def alpha_bbox(alpha: np.ndarray, threshold: int = 16) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(alpha > threshold)
    if cols.size == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def pad_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    pad_ratio: float = 0.08,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    pad_x = round((right - left) * pad_ratio)
    pad_y = round((bottom - top) * pad_ratio)
    max_w, max_h = size
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(max_w, right + pad_x),
        min(max_h, bottom + pad_y),
    )
