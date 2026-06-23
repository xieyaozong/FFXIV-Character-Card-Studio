from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from src.preprocessing.background import alpha_bbox, pad_bbox, remove_background


@dataclass
class TriageThresholds:
    """Gates for deciding whether a screenshot is worth sending to the VLM."""

    min_brightness: float = 40.0       # mean luminance (0-255) inside the subject
    min_subject_height: int = 360      # subject bbox height in original pixels
    min_coverage: float = 0.012        # subject area / whole frame (too small = too far)
    max_coverage: float = 0.97         # near full frame = background removal failed
    min_sharpness: float = 10.0        # variance of Laplacian inside the subject crop


@dataclass
class SubjectMetrics:
    coverage: float
    subject_height_px: int
    brightness: float
    sharpness: float
    bbox: tuple[int, int, int, int] | None  # original-image coordinates


@dataclass
class TriageResult:
    usable: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    metrics: SubjectMetrics | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        return self.metrics.bbox if self.metrics else None


def _downscale(image: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    work = image.convert("RGB")
    if max(work.size) <= max_side:
        return work, 1.0
    scale = max(work.size) / max_side
    work = work.copy()
    work.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return work, scale


def analyze_subject(
    image: Image.Image,
    *,
    mask_backend: str = "rembg",
    mask_max_side: int = 1280,
    alpha_threshold: int = 16,
) -> SubjectMetrics:
    """Segment the character once and derive cheap usability metrics from the mask."""
    work, scale = _downscale(image, mask_max_side)
    alpha = np.asarray(remove_background(work, mask_backend).convert("RGBA"))[:, :, 3]
    coverage = float((alpha > alpha_threshold).mean())
    gray = cv2.cvtColor(np.asarray(work), cv2.COLOR_RGB2GRAY)

    small_bbox = alpha_bbox(alpha, alpha_threshold)
    if small_bbox is None:
        brightness = float(gray.mean())
        return SubjectMetrics(coverage, 0, brightness, 0.0, None)

    mask = alpha > alpha_threshold
    brightness = float(gray[mask].mean())
    left, top, right, bottom = small_bbox
    crop = gray[top:bottom, left:right]
    sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0

    bbox = tuple(round(value * scale) for value in small_bbox)
    subject_height = bbox[3] - bbox[1]
    return SubjectMetrics(coverage, subject_height, brightness, sharpness, bbox)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def evaluate_metrics(
    metrics: SubjectMetrics,
    thresholds: TriageThresholds | None = None,
) -> TriageResult:
    """Apply gates and produce a 0-1 usability score. Pure: no image work here."""
    thresholds = thresholds or TriageThresholds()
    reasons: list[str] = []
    if metrics.bbox is None:
        return TriageResult(False, 0.0, ["no_subject_detected"], metrics)
    if metrics.brightness < thresholds.min_brightness:
        reasons.append("too_dark")
    if metrics.subject_height_px < thresholds.min_subject_height:
        reasons.append("subject_too_small")
    if metrics.coverage < thresholds.min_coverage:
        reasons.append("subject_too_far")
    if metrics.coverage > thresholds.max_coverage:
        reasons.append("mask_covers_whole_frame")
    if metrics.sharpness < thresholds.min_sharpness:
        reasons.append("too_blurry")

    score = (
        0.45 * _clamp01(metrics.brightness / 160.0)
        + 0.35 * _clamp01(metrics.subject_height_px / 900.0)
        + 0.20 * _clamp01(metrics.sharpness / 120.0)
    )
    return TriageResult(not reasons, round(score, 4), reasons, metrics)


def triage_image(
    image: Image.Image,
    *,
    thresholds: TriageThresholds | None = None,
    mask_backend: str = "rembg",
    mask_max_side: int = 1280,
) -> TriageResult:
    metrics = analyze_subject(image, mask_backend=mask_backend, mask_max_side=mask_max_side)
    return evaluate_metrics(metrics, thresholds)


def crop_subject(
    image: Image.Image,
    result: TriageResult | SubjectMetrics,
    *,
    pad_ratio: float = 0.08,
) -> Image.Image:
    """Crop the original image to the detected subject. Returns the input unchanged if no bbox."""
    if result.bbox is None:
        return image
    return image.crop(pad_bbox(result.bbox, image.size, pad_ratio))
