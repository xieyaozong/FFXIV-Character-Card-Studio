from pathlib import Path

import yaml

from src.catalog.entity_catalog import EntityCatalog


def test_content_pack_loads() -> None:
    root = Path("content_packs/ffxiv")
    manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
    anatomy = yaml.safe_load((root / manifest["anatomy_file"]).read_text(encoding="utf-8"))

    EntityCatalog.load(root / manifest["entity_file"])
    assert anatomy["default_compatibility"] == manifest["default_compatibility"]
