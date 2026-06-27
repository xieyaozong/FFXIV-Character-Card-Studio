from __future__ import annotations

import numpy as np

from src.domain.models import RaceTraits
from src.knowledge.race_index import RaceIndex
from src.knowledge.race_matcher import recognize_race_ensemble, to_race_id
from src.knowledge.races import RaceSignature

SIGNATURES = {
    "au_ra": RaceSignature(signature={"horns": "present", "ear_type": "scaled_fin"}, decisive=["horns", "ear_type"]),
    "miqote": RaceSignature(signature={"ear_type": "feline", "tail_type": "feline_furred"}, decisive=["ear_type"]),
}

# Tiny index: orthogonal vectors for Au Ra vs Miqo'te.
AU = np.array([1.0, 0.0, 0.0, 0.0])
MI = np.array([0.0, 1.0, 0.0, 0.0])
INDEX = RaceIndex(np.stack([AU, AU, MI, MI]), ["Au Ra", "Au Ra", "Miqo'te", "Miqo'te"])


def test_folder_label_normalizes_to_race_id() -> None:
    assert to_race_id("Au Ra") == "au_ra"
    assert to_race_id("Miqo'te") == "miqote"


def test_embedding_recovers_au_ra_when_vlm_blind() -> None:
    # The real failure: VLM sees nothing decisive (all occluded) but the head embedding is Au Ra.
    match = recognize_race_ensemble(RaceTraits(), SIGNATURES, embedding=AU, index=INDEX)
    assert match.race_id == "au_ra"
    assert not match.needs_confirmation
    assert "embedding only" in match.reasons


def test_agreement_is_confident() -> None:
    traits = RaceTraits(ear_type="feline", tail_type="feline_furred")
    match = recognize_race_ensemble(traits, SIGNATURES, embedding=MI, index=INDEX)
    assert match.race_id == "miqote"
    assert "vlm+embedding agree" in match.reasons


def test_disagreement_defers_to_confirmation() -> None:
    # VLM says Miqo'te (feline), embedding says Au Ra — conflict -> ask.
    traits = RaceTraits(ear_type="feline", tail_type="feline_furred")
    match = recognize_race_ensemble(traits, SIGNATURES, embedding=AU, index=INDEX)
    assert match.race_id is None
    assert match.needs_confirmation


def test_vlm_only_when_no_index() -> None:
    traits = RaceTraits(ear_type="feline", tail_type="feline_furred")
    match = recognize_race_ensemble(traits, SIGNATURES)
    assert match.race_id == "miqote"
