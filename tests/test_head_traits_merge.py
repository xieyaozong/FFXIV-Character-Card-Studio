from __future__ import annotations

from src.domain.models import RaceTraits
from src.vlm.feature_merger import merge_head_traits


def test_head_pass_overrides_full_body_false_negative() -> None:
    # The real failure: full-body pass missed the Au Ra horns/scales; the head zoom catches them.
    full = RaceTraits(ear_type="human", horns="absent", scales="absent", tail_type="none", stature="average")
    head = RaceTraits(ear_type="long_pointed", horns="present", scales="face", face_type="human")
    merged = merge_head_traits(full, head)
    assert merged.horns == "present"
    assert merged.scales == "face"
    assert merged.ear_type == "long_pointed"
    # Body traits the head crop cannot see are kept from the full-body pass.
    assert merged.tail_type == "none"
    assert merged.stature == "average"


def test_occluded_head_values_do_not_clobber_full() -> None:
    full = RaceTraits(ear_type="feline", horns="absent", scales="absent")
    head = RaceTraits()  # all occluded (e.g. head crop was unusable)
    merged = merge_head_traits(full, head)
    assert merged.ear_type == "feline"
    assert merged.horns == "absent"


def test_head_can_confirm_absence() -> None:
    full = RaceTraits(horns="occluded")
    head = RaceTraits(horns="absent")
    assert merge_head_traits(full, head).horns == "absent"
