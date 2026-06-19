from __future__ import annotations

from src.domain.models import CharacterProfile, EvidenceStatus


def test_optional_entities_default_to_hidden() -> None:
    profile = CharacterProfile(profile_id="demo")
    assert profile.job.include is False
    assert profile.job.status == EvidenceStatus.NOT_VISIBLE
    assert profile.weapon.include is False
