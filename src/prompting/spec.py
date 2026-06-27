from __future__ import annotations

from dataclasses import dataclass, field

from src.catalog.race_recognizer import RaceSignature, recognize_race
from src.domain.models import RaceTraits


@dataclass
class GenerationSpec:
    """Two-channel generation spec: user content + lore guardrails, plus a constraint checklist."""

    positive_prompt: str
    negative_prompt: str
    constraints: dict[str, list[str]] = field(default_factory=lambda: {"required": [], "forbidden": []})
    race_id: str | None = None


def anatomy_profile_for(race_id: str, anatomy_rules: dict) -> dict | None:
    """Find the anatomy profile for a race. Profiles are keyed race[_clan[_gender]]; match by prefix."""
    profiles = anatomy_rules.get("profiles") or {}
    if race_id in profiles:
        return profiles[race_id]
    for key, profile in profiles.items():
        if key.startswith(race_id):
            return profile
    return None


def compile_generation_spec(
    *,
    content_terms: list[str],
    style_prompt: str,
    base_negative: str,
    traits: RaceTraits,
    race_signatures: dict[str, RaceSignature],
    anatomy_rules: dict,
    extra_prompt: str = "",
    race_id: str | None = None,
    recognized: bool = False,
) -> GenerationSpec:
    """Recognize the race, then assemble the two channels as separate blocks (see knowledge-layer.md §9).

    The lore-required tokens sit ahead of user content so they survive CLIP truncation; the forbidden
    tokens go to the negative. When no race is recognized, only the content channel is used.

    Pass ``recognized=True`` with a ``race_id`` (which may be None) to use an externally decided race
    (e.g. the ensemble recognizer) instead of running the trait recognizer here.
    """
    if not recognized:
        race_id = recognize_race(traits, race_signatures).race_id
    required_tokens: list[str] = []
    forbidden_tokens: list[str] = []
    constraints: dict[str, list[str]] = {"required": [], "forbidden": []}

    if race_id:
        profile = anatomy_profile_for(race_id, anatomy_rules)
        if profile:
            tokens = profile.get("generation_tokens") or {}
            required_tokens = list(tokens.get("positive") or [])
            forbidden_tokens = list(tokens.get("negative") or [])
            constraints = {
                "required": list(profile.get("required_traits") or []),
                "forbidden": list(profile.get("forbidden_traits") or []),
            }

    positive = ", ".join(part for part in [extra_prompt, style_prompt, *required_tokens, *content_terms] if part)
    negative = ", ".join(part for part in [*forbidden_tokens, base_negative] if part)
    return GenerationSpec(positive, negative, constraints, race_id)
