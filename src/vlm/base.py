from __future__ import annotations

from typing import Protocol

from PIL import Image

from src.domain.models import VLMFeatureResponse


class VLMBackend(Protocol):
    def analyze(self, images: list[Image.Image]) -> VLMFeatureResponse: ...
