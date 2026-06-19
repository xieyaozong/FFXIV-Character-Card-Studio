from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, Field


class EvidenceStatus(StrEnum):
    DETECTED = "detected"
    UNCERTAIN = "uncertain"
    NOT_VISIBLE = "not_visible"
    CONFIRMED_NONE = "confirmed_none"
    USER_ADDED = "user_added"


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


class CharacterProfile(BaseModel):
    profile_id: str
    display_name: str = ""
    locale: str = "zh-TW"
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


class VLMFeatureResponse(BaseModel):
    identity: list[FeatureCandidate] = Field(default_factory=list)
    outfit: list[FeatureCandidate] = Field(default_factory=list)
    job: OptionalEntity = Field(default_factory=OptionalEntity)
    weapon: OptionalEntity = Field(default_factory=OptionalEntity)
    uncertain: list[str] = Field(default_factory=list)
