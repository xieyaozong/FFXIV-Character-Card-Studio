from __future__ import annotations

from src.domain.models import (
    CharacterProfile,
    CompatibilityMode,
    EvidenceStatus,
    VLMFeatureResponse,
)


def test_optional_entities_default_to_hidden() -> None:
    profile = CharacterProfile(profile_id="demo")
    assert profile.anatomy.confirmed is False
    assert profile.anatomy.required_traits == []
    assert profile.anatomy.compatibility is CompatibilityMode.ADVISORY
    assert profile.job.include is False
    assert profile.job.status == EvidenceStatus.NOT_VISIBLE
    assert profile.weapon.include is False


def test_vlm_response_parses_discriminating_traits() -> None:
    payload = {
        "traits": {"ear_type": "feline", "horns": "absent", "tail_type": "feline_furred"},
        "identity": [{"key": "hair_color", "value": "black", "confidence": 0.9}],
    }
    response = VLMFeatureResponse.model_validate(payload)
    assert response.traits.ear_type == "feline"
    assert response.traits.tail_type == "feline_furred"
    assert response.traits.scales == "occluded"  # omitted field falls back to occluded
    assert response.identity[0].value == "black"


def test_vlm_response_traits_default_to_occluded_when_absent() -> None:
    response = VLMFeatureResponse.model_validate({"identity": []})
    assert response.traits.horns == "occluded"
    assert response.traits.face_type == "occluded"
