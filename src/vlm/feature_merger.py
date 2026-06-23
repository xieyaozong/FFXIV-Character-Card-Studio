from __future__ import annotations

from collections import defaultdict

from src.domain.models import FeatureCandidate


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
