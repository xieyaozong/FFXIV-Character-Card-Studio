"""Slot-aware GenerationPlan: decompose generation into per-element intents instead of one global pass.

Why this exists: a single global img2img strength + one ControlNet scale cannot satisfy conflicting
needs at once — the weapon and race head must be REPRODUCED (lock to the screenshot) while the pose,
body figure and face 五官 must be GENERATED (free, style-driven). Locking everything globally also
locks bad source poses (a hand lost in smoke); freeing everything loses the gear. The plan splits the
image into slots, each with an intent that maps to concrete diffusion controls, so a free base can be
followed by targeted reproduce passes only where fidelity matters.

Intents (from docs/project-research-plan.md, adopted) mapped to our reproduce-vs-generate rules:
- preserve_exact      keep a visible detail pixel-close      -> reference image / low denoise
- preserve_structure  keep silhouette/shape                  -> ControlNet at a HIGH per-slot scale
- preserve_semantic   keep the concept, not exact pixels     -> prompt tokens + LoRA + required traits
- generate            create a new look (free)               -> style prompt / style LoRA, LOW control
- forbid              prevent invalid anatomy/artifacts      -> negative tokens (+ future validator)

Deliberate divergences from that doc (see the reproduction-vs-generation rules): pose = generate (NOT
preserve_structure), face = generate within the race head envelope (NOT preserve_exact/IP-Adapter),
body figure = generate (low priority) with only worldview caps as forbid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.domain.models import RaceTraits
from src.knowledge.races import RaceSignature
from src.prompting.spec import compile_generation_spec

INTENTS = ("preserve_exact", "preserve_structure", "preserve_semantic", "generate", "forbid")


@dataclass
class Slot:
    """One element of the character with how faithfully it must be rendered."""

    name: str
    intent: str
    region: str = "full"  # full | head | weapon | hands | feet
    control_scale: float | None = None  # per-slot ControlNet scale when this slot gets its own pass
    strength: float | None = None  # per-slot img2img/inpaint denoise when it gets a pass
    note: str = ""


@dataclass
class GenerationPlan:
    """Compiled, slot-aware plan the runner executes as: free base -> targeted reproduce passes."""

    positive_prompt: str
    negative_prompt: str
    constraints: dict[str, list[str]]
    race_id: str | None
    base_control_scale: float  # ControlNet scale for the FREE base pass (keeps pose/body/face free)
    slots: list[Slot] = field(default_factory=list)

    def slot(self, name: str) -> Slot | None:
        return next((s for s in self.slots if s.name == name), None)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data


def compile_generation_plan(
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
    base_control_scale: float = 0.5,
    slot_control_scale: float = 0.85,
    slot_strength: float = 0.45,
) -> GenerationPlan:
    """Build the slot plan. Reuses compile_generation_spec for the prompt/negative/constraints channels,
    then attaches the slot structure + per-slot controls. ``base_control_scale`` frees the base pass;
    ``slot_control_scale`` is the HIGH scale used by reproduce passes (race head, weapon)."""
    spec = compile_generation_spec(
        content_terms=content_terms,
        style_prompt=style_prompt,
        base_negative=base_negative,
        traits=traits,
        race_signatures=race_signatures,
        anatomy_rules=anatomy_rules,
        extra_prompt=extra_prompt,
        race_id=race_id,
        recognized=recognized,
    )

    slots = [
        Slot("pose", "generate", region="full", note="action/angle are the user's; never locked"),
        Slot("style", "generate", note="output look = the user's style LoRA + style prompt"),
        Slot("face", "generate", region="head", note="五官 generated within the race head envelope"),
        Slot(
            "race_head", "preserve_semantic", region="head",
            control_scale=slot_control_scale, strength=slot_strength,
            note="lock head shape + horns/ears/face scales to the screenshot",
        ),
        Slot("outfit", "preserve_semantic", region="full", note="gear from init+prompt at the base scale"),
        Slot(
            "weapon", "preserve_structure", region="weapon",
            control_scale=slot_control_scale, strength=slot_strength,
            note="reproduce shape/colors; needs a weapon region to get its own pass (not yet localized)",
        ),
        Slot("forbid", "forbid", note="forbidden anatomy/artifacts -> negative prompt"),
    ]
    return GenerationPlan(
        positive_prompt=spec.positive_prompt,
        negative_prompt=spec.negative_prompt,
        constraints=spec.constraints,
        race_id=spec.race_id,
        base_control_scale=base_control_scale,
        slots=slots,
    )
