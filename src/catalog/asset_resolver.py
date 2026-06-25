"""Map a recognized race to the maintainer's curated generation assets.

Closes the zero-prompt loop: once the recognizer names the race, the knowledge DB supplies the
matching LoRA(s) and an optional canonical reference image, so the user never hand-wires them.
Assets are read from the optional ``assets`` block of the race's anatomy profile.
"""

from __future__ import annotations

from src.prompting.spec import anatomy_profile_for


def resolve_race_assets(race_id: str, anatomy_rules: dict) -> dict:
    """Return ``{"loras": [...], "ip_adapter_image": str | None}`` for a recognized race."""
    profile = anatomy_profile_for(race_id, anatomy_rules) or {}
    assets = profile.get("assets") or {}
    return {
        "loras": list(assets.get("loras") or []),
        "ip_adapter_image": assets.get("ip_adapter_image"),
    }
