from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class EvidenceStatus(StrEnum):
    DETECTED = "detected"
    UNCERTAIN = "uncertain"
    NOT_VISIBLE = "not_visible"
    CONFIRMED_NONE = "confirmed_none"
    USER_ADDED = "user_added"


class CompatibilityMode(StrEnum):
    STRICT = "strict"
    ADVISORY = "advisory"
    FREEFORM = "freeform"


class EvidenceRef(BaseModel):
    source_image: str
    evidence_box: tuple[int, int, int, int] | None = None


class FeatureCandidate(BaseModel):
    key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confirmed: bool = False


class OptionalEntity(BaseModel):
    include: bool = False
    status: EvidenceStatus = EvidenceStatus.NOT_VISIBLE
    canonical_id: str | None = None
    candidates: list[FeatureCandidate] = Field(default_factory=list)


class OutfitProfile(BaseModel):
    profile_id: str
    name: str
    features: list[FeatureCandidate] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)


class AnatomyProfile(BaseModel):
    race_id: str = ""
    clan_id: str = ""
    compatibility: CompatibilityMode = CompatibilityMode.ADVISORY
    required_traits: list[str] = Field(default_factory=list)
    conditional_traits: list[str] = Field(default_factory=list)
    forbidden_traits: list[str] = Field(default_factory=list)
    confirmed: bool = False


class CharacterProfile(BaseModel):
    profile_id: str
    display_name: str = ""
    locale: str = "zh-TW"
    anatomy: AnatomyProfile = Field(default_factory=AnatomyProfile)
    identity_features: list[FeatureCandidate] = Field(default_factory=list)
    outfits: list[OutfitProfile] = Field(default_factory=list)
    job: OptionalEntity = Field(default_factory=OptionalEntity)
    weapon: OptionalEntity = Field(default_factory=OptionalEntity)
    personality: list[str] = Field(default_factory=list)
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    quote: str = ""
    palette: list[str] = Field(default_factory=list)


class PanelRequest(BaseModel):
    product: str
    style: str
    view: str
    pose: str
    expression: str
    custom_direction: str = ""


class PromptPlan(BaseModel):
    positive_prompt: str
    negative_prompt: str
    panel: PanelRequest


class ScreenshotSummary(BaseModel):
    filename: str
    width: int
    height: int
    palette: list[str]


class RaceTraits(BaseModel):
    """Discriminating anatomical traits the recognizer matches against race signatures.

    Each field carries one of its allowed values, or "occluded" when the part is hidden
    (horns under a hat, a tail out of frame, a back-facing shot). The VLM reports what it
    sees; it never names the race. Allowed values are listed beside each field.
    """

    ear_type: str = "occluded"   # human | long_pointed | feline | rabbit_long | leonine | scaled_fin | occluded
    horns: str = "occluded"      # present | absent | occluded
    scales: str = "occluded"     # face | body | absent | occluded
    tail_type: str = "occluded"  # scaled | feline_furred | none | occluded
    stature: str = "occluded"    # child_short | average | large_tall | occluded
    face_type: str = "occluded"  # human | feline_muzzle | occluded

    @field_validator("ear_type", mode="before")
    @classmethod
    def _normalize_ear_type(cls, value: object) -> object:
        """Map free-text VLM variants to canonical values (e.g. 'fin-shaped' -> 'scaled_fin')."""
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if "fin" in normalized:
            return "scaled_fin"
        return normalized or "occluded"


class VLMFeatureResponse(BaseModel):
    traits: RaceTraits = Field(default_factory=RaceTraits)
    identity: list[FeatureCandidate] = Field(default_factory=list)
    outfit: list[FeatureCandidate] = Field(default_factory=list)
    job: OptionalEntity = Field(default_factory=OptionalEntity)
    weapon: OptionalEntity = Field(default_factory=OptionalEntity)
    uncertain: list[str] = Field(default_factory=list)
