from __future__ import annotations

from pathlib import Path

from src.domain.models import RaceTraits
from src.knowledge.races import (
    RaceSignature,
    apply_canonical_traits,
    load_race_signatures,
    recognize_race,
)

AU_RA = RaceSignature(
    signature={"horns": "present", "scales": "face", "tail_type": "scaled"},
    decisive=["horns"],
)
MIQOTE = RaceSignature(
    signature={"ear_type": "feline", "horns": "absent", "tail_type": "feline_furred"},
    decisive=["ear_type"],
)
SIGNATURES = {"au_ra": AU_RA, "miqote": MIQOTE}

# Reproduces the real-world failure: the VLM false-negatived the pale horns / faint scales / tail
# but the head-zoom pass caught the scaled fin-ears. The all-baseline Hyur signature must NOT win.
AU_RA_FIN = RaceSignature(
    signature={"horns": "present", "ear_type": "scaled_fin", "scales": "face", "tail_type": "scaled"},
    decisive=["horns", "ear_type"],
)
HYUR = RaceSignature(
    signature={"ear_type": "human", "horns": "absent", "scales": "absent", "tail_type": "none", "stature": "average"},
    decisive=[],
)
FIN_SIGNATURES = {"au_ra": AU_RA_FIN, "hyur": HYUR}


def test_fin_ears_beat_hyur_despite_false_negatives() -> None:
    traits = RaceTraits(
        ear_type="scaled_fin", horns="absent", scales="absent", tail_type="none", stature="average", face_type="human"
    )
    match = recognize_race(traits, FIN_SIGNATURES)
    assert match.race_id == "au_ra"
    assert not match.needs_confirmation


def test_plain_human_still_recognized_as_hyur() -> None:
    traits = RaceTraits(
        ear_type="human", horns="absent", scales="absent", tail_type="none", stature="average", face_type="human"
    )
    match = recognize_race(traits, FIN_SIGNATURES)
    assert match.race_id == "hyur"


def test_decisive_trait_recognizes_au_ra() -> None:
    # The VLM mis-typed the tail and missed scales, but horns=present is decisive.
    traits = RaceTraits(
        ear_type="human", horns="present", scales="absent",
        tail_type="feline_furred", stature="average", face_type="human",
    )
    match = recognize_race(traits, SIGNATURES)
    assert match.race_id == "au_ra"
    assert not match.needs_confirmation


def test_feline_ears_recognize_miqote() -> None:
    traits = RaceTraits(ear_type="feline", horns="absent", tail_type="feline_furred")
    match = recognize_race(traits, SIGNATURES)
    assert match.race_id == "miqote"


def test_all_occluded_needs_confirmation() -> None:
    match = recognize_race(RaceTraits(), SIGNATURES)
    assert match.race_id is None
    assert match.needs_confirmation


def test_correction_locks_race_defining_traits() -> None:
    traits = RaceTraits(horns="present", scales="absent", tail_type="feline_furred")
    corrected = apply_canonical_traits("au_ra", traits, SIGNATURES)
    assert corrected.tail_type == "scaled"  # corrected from feline_furred
    assert corrected.scales == "face"       # filled in from absent
    assert corrected.horns == "present"


def test_example_signatures_file_loads() -> None:
    signatures = load_race_signatures(Path("knowledge/ffxiv/race_signatures.example.yaml"))
    assert "example_race" in signatures
    assert signatures["example_race"].signature["horns"] == "present"
