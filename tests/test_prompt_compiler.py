from __future__ import annotations

from src.domain.models import CharacterProfile, FeatureCandidate, PanelRequest
from src.prompting.compiler import compile_prompt


def test_prompt_uses_only_confirmed_features_and_omits_weapon() -> None:
    profile = CharacterProfile(
        profile_id="demo",
        identity_features=[
            FeatureCandidate(key="hair", value="long black hair", confidence=0.9, confirmed=True),
            FeatureCandidate(key="eyes", value="purple eyes", confidence=0.6, confirmed=False),
        ],
    )
    panel = PanelRequest(
        product="avatar portrait",
        style="hand drawn character sheet",
        view="bust portrait",
        pose="facing viewer",
        expression="sleepy",
    )
    plan = compile_prompt(profile, panel)
    assert "long black hair" in plan.positive_prompt
    assert "purple eyes" not in plan.positive_prompt
    assert "weapon" in plan.negative_prompt
