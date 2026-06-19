from __future__ import annotations

from src.catalog.entity_catalog import EntityCatalog, EntityRecord


def test_resolve_multilingual_alias() -> None:
    record = EntityRecord(
        canonical_id="job.example",
        entity_type="job",
        names={"en-US": "Example", "ja-JP": "サンプル", "zh-TW": "範例"},
    )
    catalog = EntityCatalog([record])
    assert catalog.resolve("Example") == record
    assert catalog.resolve("サンプル") == record
    assert catalog.resolve("範例") == record
