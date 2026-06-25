"""Control-map preprocessors for ControlNet structural conditioning.

The user's screenshot already holds the character's true geometry. These turn it into a control
map so the redraw follows the real horn shape, hairstyle, and pose instead of inventing them — no
training, one image. Pick the preprocessor that matches the downloaded ControlNet:

- ``canny``  edge map (OpenCV, no extra model) — strong on horn/hair outlines; pair with a
  canny / lineart / union SDXL ControlNet.
- ``depth``  depth map (needs a depth estimator such as Depth-Anything) — body and pose; pairs
  with the depth-sdxl ControlNet already downloaded.
- ``none``   pass the image through unchanged (for a precomputed control map).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def canny_edges(image: Image.Image, *, low: int = 100, high: int = 200) -> Image.Image:
    """OpenCV Canny edges as a 3-channel control map. No model download required."""
    array = np.asarray(image.convert("RGB"))
    edges = cv2.Canny(array, low, high)
    return Image.fromarray(edges).convert("RGB")


def depth_map(image: Image.Image, model_path: Path, *, device: str = "cpu") -> Image.Image:
    """Monocular depth as a 3-channel control map via a transformers depth estimator."""
    from transformers import pipeline

    estimator = pipeline(
        "depth-estimation",
        model=str(model_path),
        device=0 if device == "cuda" else -1,
    )
    depth = estimator(image.convert("RGB"))["depth"]
    return depth.convert("RGB")


def build_control_image(
    image: Image.Image,
    preprocessor: str,
    *,
    depth_model: Path | None = None,
    device: str = "cpu",
) -> Image.Image:
    """Dispatch to the named preprocessor. ``depth`` requires ``depth_model``."""
    if preprocessor == "none":
        return image.convert("RGB")
    if preprocessor == "canny":
        return canny_edges(image)
    if preprocessor == "depth":
        if depth_model is None:
            raise ValueError("depth preprocessor needs --depth-model (a depth estimator path)")
        return depth_map(image, depth_model, device=device)
    raise ValueError(f"unknown control preprocessor: {preprocessor!r}")
