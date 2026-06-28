from __future__ import annotations

import pytest
from PIL import Image

from src.rendering.card_layout import FONT_REGULAR, CardSpec, Panel, render_card

pytestmark = pytest.mark.skipif(not FONT_REGULAR.exists(), reason="needs a local CJK font")


def test_render_card_returns_expected_canvas() -> None:
    hero = Image.new("RGB", (512, 900), (200, 180, 160))
    spec = CardSpec(
        fields={"種族": "敖龍族", "武器": "先鋒雙牙"},
        personality=["安靜", "愛睏"],
        quotes=["今天也想睡覺…"],
        likes=["墨鏡"],
        hobbies=["FF14"],
        mood="今天超開心！",
        date="2026年6月28日",
        hero=hero,
        expressions=[Panel(None, "開心"), Panel(None, "生氣")],
        chibis=[Panel(None, "睡覺")],
        palette=["#AABBCC", "#112233"],
    )
    card = render_card(spec, size=(1240, 1754))
    assert card.size == (1240, 1754)
    assert card.mode == "RGB"


def test_render_card_handles_empty_spec() -> None:
    card = render_card(CardSpec(), size=(800, 1000))
    assert card.size == (800, 1000)
