from __future__ import annotations

from PIL import Image
import cv2
import numpy as np


def rgb_to_hex(color: np.ndarray) -> str:
    red, green, blue = (int(value) for value in color)
    return f"#{red:02X}{green:02X}{blue:02X}"


def extract_palette(image: Image.Image, colors: int = 5) -> list[str]:
    rgba = np.asarray(image.convert("RGBA"))
    pixels = rgba[rgba[:, :, 3] > 32, :3]
    if pixels.size == 0:
        return []
    if len(pixels) > 30000:
        indices = np.linspace(0, len(pixels) - 1, 30000, dtype=np.int32)
        pixels = pixels[indices]
    samples = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(samples, colors, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=colors)
    ordered = centers[np.argsort(counts)[::-1]]
    return [rgb_to_hex(color) for color in ordered]
