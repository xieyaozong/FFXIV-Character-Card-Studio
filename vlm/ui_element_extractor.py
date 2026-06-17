from __future__ import annotations


def extract_ui_elements(caption: str) -> list[str]:
    return [part.strip() for part in caption.split(",") if part.strip()]
