from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("Image data is empty.")
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("File is not a readable image.") from exc


def load_image_path(path: Path) -> Image.Image:
    return load_image(path.read_bytes())
