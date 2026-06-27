"""Ensemble race recognition: combine VLM traits with image-embedding kNN.

Neither signal is reliable alone (the VLM misreads Au Ra's subtle anatomy; embedding kNN degrades
across the creation-shot -> in-game domain gap). They are complementary: the VLM nails the obvious
races (cat / rabbit / muzzle / tiny / tall), embedding ranks Au Ra-type highest. This combines them:

  agree            -> that race, boosted confidence
  one side only    -> trust the confident side (each covers the other's blind spot)
  both but differ  -> defer to one-tap confirmation
  neither          -> defer, with a blended best-guess suggestion
"""

from __future__ import annotations

from src.catalog.race_classifier import RaceIndex
from src.catalog.race_recognizer import RaceMatch, RaceSignature, recognize_race
from src.domain.models import RaceTraits


def to_race_id(name: str) -> str:
    """Normalize a base_character folder label ('Au Ra', \"Miqo'te\") to a race_id ('au_ra')."""
    return name.strip().lower().replace("'", "").replace(" ", "_")


def _distribution(ranked: list[tuple[str, float]]) -> dict[str, float]:
    positive = {race: max(0.0, score) for race, score in ranked}
    total = sum(positive.values()) or 1.0
    return {race: value / total for race, value in positive.items()}


def _blend(
    vlm_ranked: list[tuple[str, float]],
    emb_ranked: list[tuple[str, float]],
    w_vlm: float = 0.5,
    w_emb: float = 0.5,
) -> list[tuple[str, float]]:
    vlm_dist = _distribution(vlm_ranked)
    emb_dist = _distribution([(to_race_id(race), score) for race, score in emb_ranked])
    races = set(vlm_dist) | set(emb_dist)
    blended = {race: w_vlm * vlm_dist.get(race, 0.0) + w_emb * emb_dist.get(race, 0.0) for race in races}
    return sorted(blended.items(), key=lambda item: round(item[1], 6), reverse=True)


def recognize_race_ensemble(
    traits: RaceTraits,
    signatures: dict[str, RaceSignature],
    *,
    embedding=None,
    index: RaceIndex | None = None,
) -> RaceMatch:
    """Combine the trait recognizer with the image-embedding classifier. VLM-only if no index."""
    vlm = recognize_race(traits, signatures)
    if embedding is None or index is None:
        return vlm

    emb = index.classify(embedding)
    emb_race = to_race_id(emb.race) if emb.race else None
    merged = _blend(vlm.ranked, emb.ranked)

    if vlm.race_id and emb_race:
        if vlm.race_id == emb_race:
            confidence = round(min(1.0, 0.6 + 0.4 * max(vlm.confidence, emb.confidence)), 3)
            return RaceMatch(vlm.race_id, confidence, False, merged, ["vlm+embedding agree"])
        return RaceMatch(None, 0.4, True, merged, [f"disagree: vlm={vlm.race_id} embedding={emb_race}"])
    if vlm.race_id:
        return RaceMatch(vlm.race_id, vlm.confidence, False, merged, ["vlm only"])
    if emb_race:
        return RaceMatch(emb_race, emb.confidence, False, merged, ["embedding only"])

    suggestion = merged[0][0] if merged else None
    confidence = round(merged[0][1], 3) if merged else 0.0
    return RaceMatch(None, confidence, True, merged, ["neither confident", f"suggestion={suggestion}"])
