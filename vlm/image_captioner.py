from __future__ import annotations

from vlm.screenshot_preprocessor import load_screenshot


def caption_image_bytes(data: bytes) -> str:
    image = load_screenshot(data)
    width, height = image.size
    return f"Screenshot loaded ({width}x{height}). Connect a VLM provider to generate semantic UI captions."
