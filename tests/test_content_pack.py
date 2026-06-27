from pathlib import Path

import yaml

from src.knowledge.entities import EntityStore


def test_content_pack_loads() -> None:
    # The real anatomy_rules.yaml / entities.yaml are git-ignored (maintainer's private data),
    # so the format is validated against the published *.example.yaml templates.
    root = Path("knowledge/ffxiv")
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    anatomy = yaml.safe_load((root / "anatomy_rules.example.yaml").read_text(encoding="utf-8"))

    EntityStore.load(root / "entities.example.yaml")
    assert anatomy["default_compatibility"] == manifest["default_compatibility"]
