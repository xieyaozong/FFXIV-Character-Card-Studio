from __future__ import annotations


def build_multimodal_prompt(question: str, caption: str) -> str:
    return f"Question: {question}\nScreenshot context: {caption}"
