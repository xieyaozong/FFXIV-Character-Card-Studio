from __future__ import annotations

from src.domain.models import RaceTraits
from src.knowledge.races import RaceSignature
from src.prompting.spec import compile_generation_spec

SIGNATURES = {
    "au_ra": RaceSignature(
        signature={"horns": "present", "scales": "face", "tail_type": "scaled"},
        decisive=["horns"],
    )
}
ANATOMY = {
    "profiles": {
        "au_ra_raen_female": {
            "required_traits": ["au_ra_horns", "facial_scales", "au_ra_tail"],
            "forbidden_traits": ["human_ears", "animal_ears"],
            "generation_tokens": {
                "positive": ["REAN", "type3rean-white_horns"],
                "negative": ["human ears", "cat ears"],
            },
        }
    }
}


def test_injects_guardrails_with_required_before_content() -> None:
    traits = RaceTraits(horns="present", scales="absent", tail_type="feline_furred")
    spec = compile_generation_spec(
        content_terms=["black hair", "black cap"],
        style_prompt="sketch",
        base_negative="text, watermark",
        traits=traits,
        race_signatures=SIGNATURES,
        anatomy_rules=ANATOMY,
    )
    assert spec.race_id == "au_ra"
    assert "REAN" in spec.positive_prompt
    # lore-required tokens sit ahead of user content so they survive CLIP truncation
    assert spec.positive_prompt.index("REAN") < spec.positive_prompt.index("black hair")
    assert "human ears" in spec.negative_prompt
    assert "au_ra_horns" in spec.constraints["required"]
    assert "human_ears" in spec.constraints["forbidden"]


def test_no_guardrails_when_race_unrecognized() -> None:
    spec = compile_generation_spec(
        content_terms=["black hair"],
        style_prompt="sketch",
        base_negative="text",
        traits=RaceTraits(),  # all occluded -> no race
        race_signatures=SIGNATURES,
        anatomy_rules=ANATOMY,
    )
    assert spec.race_id is None
    assert "REAN" not in spec.positive_prompt
    assert "black hair" in spec.positive_prompt
