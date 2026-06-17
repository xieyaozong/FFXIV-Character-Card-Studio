from __future__ import annotations

from io import BytesIO
from PIL import Image
from vlm.image_captioner import caption_image_bytes


def test_caption_image_bytes() -> None:
    image = Image.new("RGB", (16, 8), "black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    caption = caption_image_bytes(buffer.getvalue())
    assert "16x8" in caption
