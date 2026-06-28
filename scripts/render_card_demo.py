"""Render a demo character card from existing assets to preview the Path B layout.

Hero = a generated full-body; portrait = its head crop; palette from the foreground; text is a
sample profile. Expression/chibi slots are placeholders until those panels are generated.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from src.preprocessing.palette import extract_palette
    from src.rendering.card_layout import CardSpec, Panel, render_card

    hero_path = PROJECT_ROOT / "outputs/experiments/esrgan-on/result.png"
    fg_path = PROJECT_ROOT / "outputs/experiments/esrgan-on/input_foreground.png"
    hero = Image.open(hero_path).convert("RGB")
    w, h = hero.size
    portrait = hero.crop((int(w * 0.28), int(h * 0.04), int(w * 0.72), int(h * 0.34)))
    palette = extract_palette(Image.open(fg_path)) if fg_path.exists() else extract_palette(hero.convert("RGBA"))

    expression_labels = ["開心", "生氣", "愛睏", "害羞", "偷笑", "無言"]
    spec = CardSpec(
        title="角色檔案",
        fields={"種族": "敖龍族 / アウラ", "職業": "毒蛇劍士 (Viper)", "武器": "先鋒雙牙"},
        personality=["安靜", "冷面笑匠", "愛睏", "喜歡小動物", "戰鬥時很帥！"],
        quotes=["今天也想睡覺…", "不要逼我上班！", "我只是路過啦！"],
        likes=["漂亮的墨鏡", "好看的武器"],
        dislikes=["囉嗦的人"],
        hobbies=["FF14", "冒險！"],
        mood="今天跳舞成功了！新衣服超開心！想去新地圖冒險！",
        date="2026年6月28日",
        hero=hero,
        portrait=portrait,
        expressions=[Panel(portrait, label) for label in expression_labels],  # placeholder image
        chibis=[Panel(None, "睡覺"), Panel(None, "小跑步"), Panel(None, "坐下")],
        palette=palette,
    )

    card = render_card(spec)
    out = PROJECT_ROOT / "outputs/cards/demo-card.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    card.save(out)
    print(out)


if __name__ == "__main__":
    main()
