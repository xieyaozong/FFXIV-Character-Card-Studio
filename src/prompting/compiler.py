from __future__ import annotations

from src.domain.models import CharacterProfile, PanelRequest, PromptPlan


def confirmed_values(profile: CharacterProfile) -> list[str]:
    return [feature.value for feature in profile.identity_features if feature.confirmed]


def compile_prompt(profile: CharacterProfile, panel: PanelRequest) -> PromptPlan:
    positive = [
        *confirmed_values(profile),
        panel.product,
        panel.style,
        panel.view,
        panel.pose,
        panel.expression,
    ]
    if panel.custom_direction.strip():
        positive.append(panel.custom_direction.strip())
    if profile.job.include and profile.job.canonical_id:
        positive.append(profile.job.canonical_id)
    if profile.weapon.include and profile.weapon.canonical_id:
        positive.append(profile.weapon.canonical_id)

    negative = ["text", "watermark", "extra character", "duplicate body"]
    if not profile.weapon.include:
        negative.extend(["weapon", "sword", "gun", "bow", "staff"])

    return PromptPlan(
        positive_prompt=", ".join(value for value in positive if value),
        negative_prompt=", ".join(negative),
        panel=panel,
    )
