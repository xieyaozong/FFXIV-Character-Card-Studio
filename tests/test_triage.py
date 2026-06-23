from __future__ import annotations

import numpy as np
from PIL import Image

from src.preprocessing.background import alpha_bbox, pad_bbox
from src.preprocessing.triage import (
    SubjectMetrics,
    TriageThresholds,
    crop_subject,
    evaluate_metrics,
)


def test_alpha_bbox_finds_tight_box() -> None:
    alpha = np.zeros((100, 80), dtype=np.uint8)
    alpha[20:60, 10:50] = 255
    assert alpha_bbox(alpha) == (10, 20, 50, 60)


def test_alpha_bbox_returns_none_when_empty() -> None:
    assert alpha_bbox(np.zeros((10, 10), dtype=np.uint8)) is None


def test_pad_bbox_clamps_to_image_bounds() -> None:
    padded = pad_bbox((0, 0, 50, 50), size=(60, 60), pad_ratio=0.5)
    assert padded == (0, 0, 60, 60)


def test_evaluate_metrics_accepts_clear_subject() -> None:
    metrics = SubjectMetrics(
        coverage=0.3, subject_height_px=900, brightness=150.0, sharpness=80.0, bbox=(0, 0, 400, 900)
    )
    result = evaluate_metrics(metrics)
    assert result.usable
    assert result.reasons == []
    assert result.score > 0.7


def test_evaluate_metrics_rejects_dark_and_small() -> None:
    metrics = SubjectMetrics(coverage=0.005, subject_height_px=80, brightness=12.0, sharpness=3.0, bbox=(0, 0, 40, 80))
    result = evaluate_metrics(metrics, TriageThresholds())
    assert not result.usable
    assert {"too_dark", "subject_too_small", "subject_too_far", "too_blurry"} <= set(result.reasons)


def test_evaluate_metrics_rejects_missing_subject() -> None:
    metrics = SubjectMetrics(coverage=0.0, subject_height_px=0, brightness=0.0, sharpness=0.0, bbox=None)
    result = evaluate_metrics(metrics)
    assert not result.usable
    assert result.reasons == ["no_subject_detected"]


def test_crop_subject_uses_padded_bbox() -> None:
    image = Image.new("RGB", (200, 300))
    metrics = SubjectMetrics(
        coverage=0.2, subject_height_px=100, brightness=120.0, sharpness=40.0, bbox=(50, 100, 150, 200)
    )
    cropped = crop_subject(image, metrics, pad_ratio=0.0)
    assert cropped.size == (100, 100)


def test_crop_subject_returns_input_without_bbox() -> None:
    image = Image.new("RGB", (40, 40))
    metrics = SubjectMetrics(coverage=0.0, subject_height_px=0, brightness=0.0, sharpness=0.0, bbox=None)
    assert crop_subject(image, metrics) is image
