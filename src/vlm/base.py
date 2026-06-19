from __future__ import annotations

from PIL import Image
from typing import Protocol
from src.domain.models import VLMFeatureResponse


class VLMBackend(Protocol):
    def analyze(self, images: list[Image.Image]) -> VLMFeatureResponse: ...
