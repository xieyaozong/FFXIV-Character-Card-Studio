from __future__ import annotations

from PIL import Image


def build_review_crops(image: Image.Image) -> dict[str, Image.Image]:
    width, height = image.size
    return {
        "full_body": image.copy(),
        "face": image.crop((int(width * 0.25), 0, int(width * 0.75), int(height * 0.38))),
        "upper_body": image.crop((int(width * 0.12), 0, int(width * 0.88), int(height * 0.62))),
        "lower_body": image.crop((int(width * 0.12), int(height * 0.42), int(width * 0.88), height)),
    }
