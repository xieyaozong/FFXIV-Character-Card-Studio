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


def canny_edges(
    image: Image.Image,
    *,
    low: int = 100,
    high: int = 200,
    keep_mask: Image.Image | None = None,
    dilate_frac: float = 0.04,
) -> Image.Image:
    """OpenCV Canny edges as a 3-channel control map. No model download required.

    When ``keep_mask`` is given, edges are computed on the full image (so a held weapon and the
    hand gripping it survive even if background removal would have erased them) but only kept inside
    the mask dilated by ``dilate_frac`` of the image's short side. The dilation reaches just past the
    subject silhouette to catch an adjacent weapon, while distant background edges are dropped.
    """
    array = np.asarray(image.convert("RGB"))
    edges = cv2.Canny(array, low, high)
    if keep_mask is not None:
        mask = np.asarray(keep_mask.convert("L").resize(image.size, Image.Resampling.NEAREST))
        # Touch test: the subject silhouette lightly grown, so a held weapon/hand whose edges sit just
        # outside the rembg mask still counts as connected to the subject.
        radius = max(1, round(min(image.size) * dilate_frac))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        subject = cv2.dilate(mask, kernel) > 10
        # Keep WHOLE connected edge components that touch the subject (the weapon connects to the hand
        # so it is kept in full, even past the mask); drop background components that touch nothing.
        count, labels = cv2.connectedComponents(edges)
        touching = set(np.unique(labels[subject])) - {0}
        edges = np.where(np.isin(labels, list(touching)), edges, 0).astype(np.uint8)
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
    keep_mask: Image.Image | None = None,
) -> Image.Image:
    """Dispatch to the named preprocessor. ``depth`` requires ``depth_model``.

    ``keep_mask`` (canny only) restricts edges to the dilated subject so a with-background source can
    be used — keeping the held weapon/hand edges that background removal would otherwise erase.
    """
    if preprocessor == "none":
        return image.convert("RGB")
    if preprocessor == "canny":
        return canny_edges(image, keep_mask=keep_mask)
    if preprocessor == "depth":
        if depth_model is None:
            raise ValueError("depth preprocessor needs --depth-model (a depth estimator path)")
        return depth_map(image, depth_model, device=device)
    raise ValueError(f"unknown control preprocessor: {preprocessor!r}")
