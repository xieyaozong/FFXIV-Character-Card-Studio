"""Program-composed multi-panel character card (Path B renderer).

Takes locally-generated panels (hero, portrait, expressions, chibi) + the character's profile
text + palette and lays them out into a structured card with embedded CJK text. This is the
controllable/faithful renderer; the organic hand-drawn collage look is the cloud Path A renderer.
The deliverable is the spec (CardSpec); the renderer is swappable (see docs/knowledge-layer.md §10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# zh-TW primary (covers kanji/kana too); bold for headers.
FONT_REGULAR = Path("C:/Windows/Fonts/msjh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msjhbd.ttc")

INK = (40, 40, 48)
MUTED = (110, 110, 120)
ACCENT = (90, 165, 90)
PAPER = (252, 251, 248)
LINE = (175, 175, 185)


@dataclass
class Panel:
    image: Image.Image | None = None
    label: str = ""


@dataclass
class CardSpec:
    title: str = "角色檔案"
    fields: dict[str, str] = field(default_factory=dict)      # 種族 / 職業 / 武器 …
    personality: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    hobbies: list[str] = field(default_factory=list)
    mood: str = ""
    date: str = ""
    hero: Image.Image | None = None
    portrait: Image.Image | None = None
    expressions: list[Panel] = field(default_factory=list)
    chibis: list[Panel] = field(default_factory=list)
    palette: list[str] = field(default_factory=list)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Character-level wrap (CJK has no spaces); honor explicit newlines."""
    lines: list[str] = []
    for raw in text.split("\n"):
        current = ""
        for char in raw:
            if font.getlength(current + char) <= max_width:
                current += char
            else:
                lines.append(current)
                current = char
        lines.append(current)
    return lines


def _fit(image: Image.Image, box: tuple[int, int], pad_color=(255, 255, 255)) -> Image.Image:
    """Contain the image within box (letterbox), preserving aspect."""
    from PIL import ImageOps

    return ImageOps.contain(image.convert("RGB"), box, Image.Resampling.LANCZOS)


def _box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, body_font, title_font) -> int:
    """Draw a rounded titled box outline; return the inner content y-start."""
    x0, y0, x1, _ = xy
    draw.rounded_rectangle(xy, radius=10, outline=LINE, width=2)
    if title:
        draw.text((x0 + 14, y0 + 8), title, font=title_font, fill=ACCENT)
        return y0 + 8 + title_font.size + 8
    return y0 + 12


def _lines(draw, x, y, items, font, max_width, *, lh=8, color=INK, bullet="") -> int:
    for item in items:
        for ln in _wrap((bullet + item) if bullet else item, font, max_width):
            draw.text((x, y), ln, font=font, fill=color)
            y += font.size + lh
    return y


def _panel_grid(canvas, draw, panels, region, cols, label_font, *, cell_gap=10):
    """Lay out labeled panels in a grid inside region (x0,y0,x1,y1)."""
    x0, y0, x1, y1 = region
    rows = (len(panels) + cols - 1) // cols
    cw = (x1 - x0 - cell_gap * (cols - 1)) // cols
    ch = (y1 - y0 - cell_gap * (rows - 1)) // max(rows, 1)
    for i, panel in enumerate(panels):
        cx = x0 + (i % cols) * (cw + cell_gap)
        cy = y0 + (i // cols) * (ch + cell_gap)
        img_h = ch - (label_font.size + 6 if panel.label else 0)
        if panel.image is not None:
            thumb = _fit(panel.image, (cw, img_h))
            canvas.paste(thumb, (cx + (cw - thumb.width) // 2, cy + (img_h - thumb.height) // 2))
        else:
            draw.rounded_rectangle((cx, cy, cx + cw, cy + img_h), radius=8, outline=LINE, width=1)
            draw.text((cx + cw // 2, cy + img_h // 2), "—", font=label_font, fill=MUTED, anchor="mm")
        if panel.label:
            draw.text((cx + cw // 2, cy + img_h + 3), panel.label, font=label_font, fill=INK, anchor="ma")


def render_card(spec: CardSpec, *, size: tuple[int, int] = (1240, 1754)) -> Image.Image:
    W, H = size
    canvas = Image.new("RGB", size, PAPER)
    draw = ImageDraw.Draw(canvas)
    m = 28

    f_title = _font(FONT_BOLD, 30)
    f_head = _font(FONT_BOLD, 22)
    f_body = _font(FONT_REGULAR, 19)
    f_small = _font(FONT_REGULAR, 16)

    # Header
    draw.text((m, m), spec.title, font=f_title, fill=INK)
    if spec.date:
        draw.text((W - m, m + 6), spec.date, font=f_body, fill=MUTED, anchor="ra")
    draw.line((m, m + 46, W - m, m + 46), fill=LINE, width=2)

    top = m + 60
    col_l = (m, 300)                      # left column x, width
    col_c = (348, 540)                    # center (hero)
    col_r = (912, W - m - 912 + m)        # right column

    # ---- Left column: profile / personality / quotes ----
    x, w = col_l
    y = top
    box_b = y
    # profile box
    fields_lines = [f"{k}：{v}" for k, v in spec.fields.items()]
    bh = 16 + f_head.size + 8 + len(fields_lines) * (f_body.size + 8) + 12
    inner = _box(draw, (x, y, x + w, y + bh), spec.title, f_body, f_head)
    _lines(draw, x + 14, inner, fields_lines, f_body, w - 28)
    y += bh + 14
    # personality
    if spec.personality:
        bh = 16 + f_head.size + 8 + len(spec.personality) * (f_body.size + 8) + 12
        inner = _box(draw, (x, y, x + w, y + bh), "個性", f_body, f_head)
        _lines(draw, x + 14, inner, spec.personality, f_body, w - 28, bullet="・")
        y += bh + 14
    # quotes
    if spec.quotes:
        qlines = sum(len(_wrap("「" + q + "」", f_body, w - 28)) for q in spec.quotes)
        bh = 16 + f_head.size + 8 + qlines * (f_body.size + 8) + 12
        inner = _box(draw, (x, y, x + w, y + bh), "口頭禪", f_body, f_head)
        _lines(draw, x + 14, inner, [f"「{q}」" for q in spec.quotes], f_body, w - 28)
        y += bh + 14
    box_b = max(box_b, y)

    # ---- Center: hero ----
    hx, hw = col_c
    if spec.hero is not None:
        hero = _fit(spec.hero, (hw, 920))
        canvas.paste(hero, (hx + (hw - hero.width) // 2, top))
    if spec.fields.get("武器"):
        draw.text((hx + hw // 2, top + 928), f"武器：{spec.fields['武器']}", font=f_small, fill=MUTED, anchor="ma")

    # ---- Right column: likes / dislikes / hobbies / mood ----
    x, w = col_r
    y = top
    for title, items in (("喜歡的東西", spec.likes), ("討厭的東西", spec.dislikes), ("興趣", spec.hobbies)):
        if not items:
            continue
        bh = 16 + f_head.size + 8 + len(items) * (f_body.size + 8) + 12
        inner = _box(draw, (x, y, x + w, y + bh), title, f_body, f_head)
        _lines(draw, x + 14, inner, items, f_body, w - 28, bullet="・")
        y += bh + 14
    if spec.mood:
        mlines = _wrap(spec.mood, f_body, w - 28)
        bh = 16 + f_head.size + 8 + len(mlines) * (f_body.size + 8) + 12
        inner = _box(draw, (x, y, x + w, y + bh), "今日心情", f_body, f_head)
        _lines(draw, x + 14, inner, [spec.mood], f_body, w - 28)
        y += bh + 14
    # portrait thumb under right column
    if spec.portrait is not None and y < top + 700:
        ph = _fit(spec.portrait, (w, 320))
        canvas.paste(ph, (x + (w - ph.width) // 2, y + 4))

    # ---- Expression grid ----
    band = max(box_b, top + 940) + 20
    if spec.expressions:
        draw.text((m, band), "表情包（隨機！）", font=f_head, fill=INK)
        _panel_grid(canvas, draw, spec.expressions, (m, band + 34, W // 2 + 80, band + 34 + 280), 3, f_small)

    # ---- Chibi row ----
    chibi_y = band + 34 + 300
    if spec.chibis:
        draw.text((m, chibi_y), "動作", font=f_head, fill=INK)
        _panel_grid(canvas, draw, spec.chibis, (m, chibi_y + 34, W - m, chibi_y + 34 + 220), len(spec.chibis), f_small)

    # ---- Footer: palette + id ----
    fy = H - m - 40
    if spec.palette:
        sw = 46
        for i, hexv in enumerate(spec.palette):
            cx = m + i * (sw + 8)
            color = tuple(int(hexv.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))
            draw.rounded_rectangle((cx, fy, cx + sw, fy + 40), radius=6, fill=color, outline=LINE)
        draw.text((m, fy - 24), "配色", font=f_small, fill=MUTED)
    if spec.date:
        draw.text((W - m, fy + 14), spec.date, font=f_body, fill=INK, anchor="ra")

    return canvas
