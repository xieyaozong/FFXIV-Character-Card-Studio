from __future__ import annotations

from src.catalog.entity_catalog import EntityCatalog, EntityRecord
from src.catalog.gear_recognizer import recognize_gear

CATALOG = EntityCatalog(
    [
        EntityRecord(
            canonical_id="equipment.silver_set",
            entity_type="equipment",
            names={"en-US": "Silver Set"},
            aliases={"en-US": ["silver jacket", "cargo pants"]},
            visual_prompt="sleek silver bomber jacket",
            reference_image="refs/silver.png",
        ),
        EntityRecord(
            canonical_id="job.example",
            entity_type="job",
            names={"en-US": "Example Job"},
        ),
    ]
)


def test_recognizes_equipment_by_keyword_in_outfit() -> None:
    match = recognize_gear(["white silver jacket", "black boots"], CATALOG)
    assert match is not None
    assert match.record.canonical_id == "equipment.silver_set"
    assert match.record.reference_image == "refs/silver.png"
    assert "silver jacket" in match.matched


def test_ignores_non_equipment_entities() -> None:
    # The job entity must never be returned even if its name appears.
    assert recognize_gear(["example job outfit"], CATALOG) is None


def test_no_match_returns_none() -> None:
    assert recognize_gear(["plain red dress"], CATALOG) is None


def test_prefers_more_keyword_hits() -> None:
    match = recognize_gear(["silver jacket and cargo pants"], CATALOG)
    assert match is not None
    assert match.hits == 2
