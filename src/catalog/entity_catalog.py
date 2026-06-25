from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EntityRecord(BaseModel):
    canonical_id: str
    entity_type: str
    names: dict[str, str] = Field(default_factory=dict)
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    visual_prompt: str = ""
    negative_prompt: list[str] = Field(default_factory=list)
    reference_image: str = ""           # canonical glamour/gear reference (for later image injection)
    game_version: str = ""
    source: str = ""
    reviewed_at: str = ""
    notes: str = ""


class EntityCatalog:
    def __init__(self, records: list[EntityRecord]) -> None:
        self.records = records
        self.alias_index: dict[str, EntityRecord] = {}
        for record in records:
            values = list(record.names.values())
            values.extend(alias for aliases in record.aliases.values() for alias in aliases)
            for value in values:
                self.alias_index[value.casefold().strip()] = record

    @classmethod
    def load(cls, path: Path | str) -> EntityCatalog:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        records = [EntityRecord.model_validate(item) for item in data.get("entities", [])]
        return cls(records)

    def resolve(self, value: str) -> EntityRecord | None:
        return self.alias_index.get(value.casefold().strip())
