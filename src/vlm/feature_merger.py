from __future__ import annotations

from collections import defaultdict

from src.domain.models import (
    EvidenceStatus,
    FeatureCandidate,
    OptionalEntity,
    RaceTraits,
    VLMFeatureResponse,
)

# Traits a head-and-shoulders crop can judge better than a full-body shot.
HEAD_TRAITS = ("ear_type", "horns", "scales", "face_type")

TRAIT_FIELDS = ("ear_type", "horns", "scales", "tail_type", "stature", "face_type")
# Values that carry no positive information (region hidden), and ones that are a definite negative.
_TRAIT_OCCLUDED = {"occluded", ""}
_TRAIT_NEGATIVE = {"absent", "none"}
# Identity/outfit values that mean "nothing visible" — never let one override a real description.
_NON_VISIBLE = {"occluded", "none", "absent", "not visible", "n/a", "unknown", ""}


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


def _is_visible(value: str) -> bool:
    normalized = value.strip().strip(".").lower().replace("_", " ")
    return normalized not in _NON_VISIBLE and "not visible" not in normalized


def _trait_rank(value: str) -> int:
    """Informativeness of a trait reading: a concrete positive beats a definite negative beats hidden."""
    v = (value or "").strip().lower()
    if v in _TRAIT_OCCLUDED:
        return 0
    if v in _TRAIT_NEGATIVE:
        return 1
    return 2


def merge_traits(traits_list: list[RaceTraits]) -> RaceTraits:
    """Per field, take the most informative reading across angles (a back shot can't unsee horns)."""
    merged: dict[str, str] = {}
    for field in TRAIT_FIELDS:
        best_value, best_rank = "occluded", -1
        for traits in traits_list:
            value = getattr(traits, field)
            rank = _trait_rank(value)
            if rank > best_rank:
                best_value, best_rank = value, rank
        merged[field] = best_value
    return RaceTraits.model_validate(merged)


def _merge_keyed(groups: list[list[FeatureCandidate]]) -> list[FeatureCandidate]:
    """Union identity/outfit entries by key; per key keep the richest VISIBLE reading across angles.

    So a glove seen plainly in one shot ("white quilted") loses to the angle that reads it fully
    ("green fingerless gloves, exposed fingers"). Visible always beats occluded for the same slot.
    """

    def score(candidate: FeatureCandidate) -> tuple[bool, float, int]:
        return (_is_visible(candidate.value), candidate.confidence, len(candidate.value))

    by_key: dict[str, FeatureCandidate] = {}
    for group in groups:
        for candidate in group:
            key = candidate.key.casefold()
            if key not in by_key or score(candidate) > score(by_key[key]):
                by_key[key] = candidate
    return list(by_key.values())


def _merge_optional(entities: list[OptionalEntity]) -> OptionalEntity:
    include = any(entity.include for entity in entities)
    statuses = [entity.status for entity in entities]
    status = EvidenceStatus.DETECTED if EvidenceStatus.DETECTED in statuses else statuses[0]
    candidates = merge_feature_candidates([entity.candidates for entity in entities])
    return OptionalEntity(include=include, status=status, candidates=candidates)


def merge_feature_responses(responses: list[VLMFeatureResponse]) -> VLMFeatureResponse:
    """Aggregate per-image VLM readings of the SAME character into one complete feature set.

    A single screenshot rarely shows everything (a raised arm hides a glove, a hat hides horns, smoke
    VFX hides a hand). Running the extractor on 2-3 angles and taking the union per feature fills those
    gaps without forcing one shot to carry it all.
    """
    if not responses:
        raise ValueError("merge_feature_responses needs at least one response")
    if len(responses) == 1:
        return responses[0]
    return VLMFeatureResponse(
        traits=merge_traits([r.traits for r in responses]),
        identity=_merge_keyed([r.identity for r in responses]),
        outfit=_merge_keyed([r.outfit for r in responses]),
        job=_merge_optional([r.job for r in responses]),
        weapon=_merge_optional([r.weapon for r in responses]),
        uncertain=sorted({u for r in responses for u in r.uncertain}),
    )
