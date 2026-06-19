from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageFilter
import cv2
import numpy as np


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
