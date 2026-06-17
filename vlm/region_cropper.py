from __future__ import annotations

from PIL import Image


def center_crop(image: Image.Image, ratio: float = 0.6) -> Image.Image:
    width, height = image.size
    crop_w, crop_h = int(width * ratio), int(height * ratio)
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))
