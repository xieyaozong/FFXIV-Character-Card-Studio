"""Image-embedding race recognition ("search by image").

Recognizes race by visual similarity to the maintainer's labeled reference DB (base_character),
instead of asking the VLM to name subtle/unusual anatomy in text. This sidesteps the local VLM's
blind spots (it misreads Au Ra fin-ears and scales); a CLIP head-crop embedding + kNN against the
reference index recognizes Au Ra reliably where the VLM cannot.

Pipeline: frame the character -> crop the head region -> CLIP embed -> kNN vote over the index.
Build the index once with scripts/build_race_index.py; classify at runtime.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# Default CLIP ViT-H image encoder (already downloaded for IP-Adapter).
DEFAULT_ENCODER = Path("models/ip-adapter/models/image_encoder")
DEFAULT_INDEX = Path("content_packs/ffxiv/race_index.npz")


def character_frame(image: Image.Image) -> Image.Image:
    """Fixed center crop for the centered character-creation reference shots (no segmentation)."""
    w, h = image.size
    return image.crop((int(w * 0.36), int(h * 0.12), int(w * 0.64), int(h * 0.99)))


def head_region(framed: Image.Image) -> Image.Image:
    """Crop the head/upper region of an already character-framed image (keeps side horns/fin-ears)."""
    w, h = framed.size
    return framed.crop((int(w * 0.05), 0, int(w * 0.95), int(h * 0.34)))


def on_white(image: Image.Image) -> Image.Image:
    """Background-remove and composite on white, so busy in-game scenes match the clean refs."""
    from src.preprocessing.background import remove_background

    foreground = remove_background(image, "rembg")
    canvas = Image.new("RGBA", foreground.size, "white")
    return Image.alpha_composite(canvas, foreground).convert("RGB")


@dataclass
class RacePrediction:
    race: str | None
    confidence: float
    needs_confirmation: bool
    ranked: list[tuple[str, float]] = field(default_factory=list)


class ClipEmbedder:
    """Wraps the CLIP vision encoder; returns L2-normalized image embeddings."""

    def __init__(self, encoder_dir: Path | str = DEFAULT_ENCODER, device: str = "cuda") -> None:
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = (
            CLIPVisionModelWithProjection.from_pretrained(str(encoder_dir), torch_dtype=torch.float16)
            .to(self.device)
            .eval()
        )
        self.processor = CLIPImageProcessor(
            size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224}
        )

    def embed(self, image: Image.Image) -> np.ndarray:
        torch = self._torch
        pixels = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values
        pixels = pixels.to(self.device, torch.float16)
        with torch.no_grad():
            vector = self.model(pixels).image_embeds[0].float().cpu().numpy()
        return vector / (np.linalg.norm(vector) + 1e-8)


class RaceIndex:
    """Labeled reference embeddings; classifies a query embedding by similarity-weighted kNN vote."""

    def __init__(self, embeddings: np.ndarray, labels: list[str]) -> None:
        self.embeddings = embeddings
        self.labels = labels

    def classify(self, embedding: np.ndarray, *, k: int = 7, min_confidence: float = 0.5,
                 min_similarity: float = 0.45) -> RacePrediction:
        sims = self.embeddings @ embedding
        order = np.argsort(-sims)[:k]
        weights: dict[str, float] = defaultdict(float)
        for idx in order:
            weights[self.labels[idx]] += float(sims[idx])
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        total = sum(w for _, w in ranked) or 1.0
        top_race, top_weight = ranked[0]
        confidence = top_weight / total
        confident = confidence >= min_confidence and float(sims[order[0]]) >= min_similarity
        return RacePrediction(
            race=top_race if confident else None,
            confidence=round(confidence, 3),
            needs_confirmation=not confident,
            ranked=[(r, round(w, 3)) for r, w in ranked],
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, embeddings=self.embeddings, labels=np.array(self.labels))

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX) -> RaceIndex:
        data = np.load(Path(path), allow_pickle=False)
        return cls(data["embeddings"], [str(label) for label in data["labels"]])
