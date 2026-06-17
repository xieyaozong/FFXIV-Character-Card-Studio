from __future__ import annotations

from io import BytesIO
from PIL import Image


def load_screenshot(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")
