from __future__ import annotations

from PIL import Image
from src.preprocessing.palette import extract_palette


def test_extract_palette() -> None:
    image = Image.new("RGB", (32, 32), (20, 180, 80))
    palette = extract_palette(image, colors=2)
    assert palette
    assert all(value.startswith("#") for value in palette)
