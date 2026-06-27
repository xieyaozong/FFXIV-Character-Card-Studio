from __future__ import annotations

from src.knowledge.entities import EntityRecord, EntityStore


def test_resolve_names_and_aliases() -> None:
    record = EntityRecord(
        canonical_id="job.example",
        entity_type="job",
        names={"en-US": "Example Job"},
        aliases={"en-US": ["example", "sample job"]},
    )
    store = EntityStore([record])
    assert store.resolve("Example Job") == record
    assert store.resolve("example") == record
    assert store.resolve("sample job") == record
