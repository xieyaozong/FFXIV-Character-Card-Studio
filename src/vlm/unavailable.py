from __future__ import annotations

from PIL import Image

from src.domain.models import VLMFeatureResponse


class UnavailableVLMBackend:
    def analyze(self, images: list[Image.Image]) -> VLMFeatureResponse:
        del images
        return VLMFeatureResponse(uncertain=["VLM dependencies and model weights are not installed."])
