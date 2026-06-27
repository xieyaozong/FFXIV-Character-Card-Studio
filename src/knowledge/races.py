from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from src.domain.models import RaceTraits

# Matching weights. Decisive traits dominate and can eliminate a race on contradiction;
# minor traits only nudge the score. "occluded" observations carry no information.
DECISIVE_MATCH = 5.0
DECISIVE_CONTRADICTION = -4.0
MINOR_MATCH = 1.0
MINOR_MISMATCH = -1.0
MIN_SCORE = 2.0
MIN_MARGIN = 2.0

# Traits the VLM under-detects (small/pale/easily-missed), mapped to their "nothing here" value.
# A negative reading on these is likely a MISS, not a real contradiction, so it earns no penalty —
# this stops the all-negative Hyur signature from soaking up false negatives like a missed horn.
# ear_type / stature / face_type are deliberately excluded: ears, body height, and a muzzle are
# obvious, so "human" / "average" there are reliable discriminators that should rule races out.
UNDERDETECTED_NEGATIVE = {"horns": "absent", "scales": "absent", "tail_type": "none"}


class RaceSignature(BaseModel):
    """A race's defining traits, used both to recognize the race and to correct perception."""

    names: dict[str, str] = Field(default_factory=dict)      # FFXIV nouns: ja-JP / zh-TW / en-US
    signature: dict[str, str] = Field(default_factory=dict)  # race-defining trait -> canonical value
    decisive: list[str] = Field(default_factory=list)        # traits weighted highest when matching


def load_race_signatures(path: Path | str) -> dict[str, RaceSignature]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    races = data.get("races") or {}
    return {race_id: RaceSignature.model_validate(record) for race_id, record in races.items()}


@dataclass
class RaceMatch:
    race_id: str | None
    confidence: float
    needs_confirmation: bool
    ranked: list[tuple[str, float]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _score_race(observed: dict[str, str], sig: RaceSignature) -> float:
    score = 0.0
    for trait, canonical in sig.signature.items():
        value = observed.get(trait, "occluded")
        if value == "occluded":
            continue
        decisive = trait in sig.decisive
        if value == canonical:
            score += DECISIVE_MATCH if decisive else MINOR_MATCH
        elif UNDERDETECTED_NEGATIVE.get(trait) == value:
            # VLM saw "nothing here" on a subtle trait — likely a miss, not a real contradiction.
            continue
        else:
            # A real disagreement (two distinct features, or a reliably-seen value): penalize.
            score += DECISIVE_CONTRADICTION if decisive else MINOR_MISMATCH
    return score


def recognize_race(traits: RaceTraits, signatures: dict[str, RaceSignature]) -> RaceMatch:
    """Match observed traits against race signatures. Low score or a close runner-up → confirm."""
    observed = traits.model_dump()
    ranked = sorted(
        ((race_id, _score_race(observed, sig)) for race_id, sig in signatures.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return RaceMatch(None, 0.0, True, [], ["no_signatures"])

    top_id, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else float("-inf")
    confident = top >= MIN_SCORE and (top - second) >= MIN_MARGIN
    confidence = max(0.0, min(1.0, top / DECISIVE_MATCH))
    reasons = [] if confident else ["low_score_or_margin"]
    return RaceMatch(top_id if confident else None, round(confidence, 3), not confident, ranked, reasons)


def apply_canonical_traits(
    race_id: str,
    traits: RaceTraits,
    signatures: dict[str, RaceSignature],
) -> RaceTraits:
    """Lock the recognized race's defining traits to canonical values (the anatomy invariant).

    Corrects race-defining perception mistakes once the race is known — e.g. a scaled tail the
    VLM called furred, or scales it missed.
    """
    corrected = traits.model_copy()
    for trait, canonical in signatures[race_id].signature.items():
        if hasattr(corrected, trait):
            setattr(corrected, trait, canonical)
    return corrected
