from __future__ import annotations

from src.knowledge.assets import resolve_race_assets

ANATOMY = {
    "profiles": {
        "au_ra_raen_female": {
            "required_traits": ["au_ra_horns"],
            "assets": {
                "loras": ["models/loras/au-ra.safetensors=0.75"],
                "ip_adapter_image": "knowledge/ffxiv/refs/au_ra.png",
            },
        },
        "miqote_seeker_female": {"required_traits": ["cat_ears"]},
    }
}


def test_resolves_assets_by_race_prefix() -> None:
    assets = resolve_race_assets("au_ra", ANATOMY)
    assert assets["loras"] == ["models/loras/au-ra.safetensors=0.75"]
    assert assets["ip_adapter_image"] == "knowledge/ffxiv/refs/au_ra.png"


def test_missing_assets_block_returns_empty() -> None:
    assets = resolve_race_assets("miqote", ANATOMY)
    assert assets["loras"] == []
    assert assets["ip_adapter_image"] is None


def test_unknown_race_returns_empty() -> None:
    assets = resolve_race_assets("lalafell", ANATOMY)
    assert assets["loras"] == []
    assert assets["ip_adapter_image"] is None
