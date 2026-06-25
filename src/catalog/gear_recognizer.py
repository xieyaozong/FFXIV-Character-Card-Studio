"""Recognize the gear/glamour set from the VLM's outfit description.

Extends the recognition loop to the equipment layer: match the free-text outfit terms against the
maintainer's curated equipment entities, so the DB can backstop appearance with that set's verified
visual tokens (and, later, a canonical reference image) instead of the user prompting it.

Matching is keyword containment — an equipment entity is recognized when its names/aliases appear in
the outfit text. The maintainer gives each set distinctive aliases; until they do, this is inert.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.catalog.entity_catalog import EntityCatalog, EntityRecord


@dataclass
class GearMatch:
    record: EntityRecord
    hits: int
    matched: list[str]


def _keywords(record: EntityRecord) -> list[str]:
    values = list(record.names.values())
    values.extend(alias for aliases in record.aliases.values() for alias in aliases)
    return [value.casefold().strip() for value in values if value.strip()]


def recognize_gear(
    outfit_terms: list[str],
    catalog: EntityCatalog,
    *,
    entity_types: tuple[str, ...] = ("equipment",),
    min_hits: int = 1,
) -> GearMatch | None:
    """Return the best equipment entity whose keywords appear in the outfit text, or None."""
    text = " , ".join(outfit_terms).casefold()
    best: GearMatch | None = None
    for record in catalog.records:
        if record.entity_type not in entity_types:
            continue
        matched = sorted({keyword for keyword in _keywords(record) if keyword in text})
        if len(matched) >= min_hits and (best is None or len(matched) > best.hits):
            best = GearMatch(record, len(matched), matched)
    return best
