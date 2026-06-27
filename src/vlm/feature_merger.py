from __future__ import annotations

from collections import defaultdict

from src.domain.models import FeatureCandidate, RaceTraits

# Traits a head-and-shoulders crop can judge better than a full-body shot.
HEAD_TRAITS = ("ear_type", "horns", "scales", "face_type")


def merge_head_traits(full: RaceTraits, head: RaceTraits) -> RaceTraits:
    """Let the zoomed head pass override head traits, keeping the full-body tail/stature.

    The head crop makes small/pale horns and faint scales legible, so a concrete reading there
    (anything but "occluded") wins over the full-body pass — fixing its confident false negatives.
    """
    merged = full.model_copy(deep=True)
    for field in HEAD_TRAITS:
        head_value = getattr(head, field)
        if head_value != "occluded":
            setattr(merged, field, head_value)
    return merged


def merge_feature_candidates(groups: list[list[FeatureCandidate]]) -> list[FeatureCandidate]:
    buckets: dict[tuple[str, str], list[FeatureCandidate]] = defaultdict(list)
    for group in groups:
        for candidate in group:
            buckets[(candidate.key.casefold(), candidate.value.casefold())].append(candidate)

    merged = []
    for candidates in buckets.values():
        best = max(candidates, key=lambda item: item.confidence).model_copy(deep=True)
        best.confidence = sum(item.confidence for item in candidates) / len(candidates)
        best.evidence = [evidence for item in candidates for evidence in item.evidence]
        merged.append(best)
    return sorted(merged, key=lambda item: item.confidence, reverse=True)
