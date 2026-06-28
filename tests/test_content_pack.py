from pathlib import Path

import yaml

from src.knowledge.entities import EntityStore


def test_content_pack_loads() -> None:
    # Validate the public templates; local maintained YAML can be stricter or larger.
    root = Path("knowledge/ffxiv")
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    anatomy = yaml.safe_load((root / "anatomy_rules.example.yaml").read_text(encoding="utf-8"))

    EntityStore.load(root / "entities.example.yaml")
    assert anatomy["default_compatibility"] == manifest["default_compatibility"]
