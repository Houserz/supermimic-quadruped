"""On-screen joystick display: the arrow pad with the pressed keys lit, and the stick / robot velocities.

Drawn with Pillow as an RGBA image, so the same panel goes on the live viewer (``as_rgb``, over a
solid background: the viewer's image overlay has no alpha) and on rendered video (``composite``).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACCENT = (46, 122, 250)  # the planner ghost's blue
KEY_FILL, KEY_EDGE, KEY_GLYPH = (236, 238, 243), (150, 155, 166), (96, 101, 112)
TEXT, TEXT_DIM = (34, 37, 43), (110, 115, 125)
PANEL = (255, 255, 255, 205)


@lru_cache(maxsize=8)
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf") if bold else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _arrow(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], direction: str, fill) -> None:
    x0, y0, x1, y1 = box
    cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.24
    pts = {"up": [(cx, cy - r), (cx - r, cy + 0.7 * r), (cx + r, cy + 0.7 * r)],
           "down": [(cx, cy + r), (cx - r, cy - 0.7 * r), (cx + r, cy - 0.7 * r)],
           "left": [(cx - r, cy), (cx + 0.7 * r, cy - r), (cx + 0.7 * r, cy + r)],
           "right": [(cx + r, cy), (cx - 0.7 * r, cy - r), (cx - 0.7 * r, cy + r)]}[direction]
    d.polygon(pts, fill=fill)


def joystick_panel(lit: set[str], stick, robot=None, level: float | None = None, scale: float = 1.0) -> np.ndarray:
    """(H, W, 4) uint8 RGBA panel. ``lit``: pressed keys among "up", "down", "left", "right", "shift";
    ``stick`` / ``robot``: (vx m/s, vy m/s, wz rad/s); ``level``: speed level as a fraction."""
    s = scale
    k, g, pad = round(32 * s), round(5 * s), round(10 * s)
    shift_w = round(50 * s)
    pad_w = shift_w + g + 3 * k + 2 * g
    text_x = pad + pad_w + round(14 * s)
    col = [round(x * s) for x in (98, 160, 228)]  # right edges of the three value columns
    W, H = text_x + col[-1] + pad, pad * 2 + 2 * k + g
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=round(10 * s), fill=PANEL)
    keys = {"shift": (pad, pad + k + g, pad + shift_w, pad + 2 * k + g)}
    x = pad + shift_w + g
    keys["left"] = (x, pad + k + g, x + k, pad + 2 * k + g)
    keys["down"] = (x + k + g, pad + k + g, x + 2 * k + g, pad + 2 * k + g)
    keys["right"] = (x + 2 * (k + g), pad + k + g, x + 3 * k + 2 * g, pad + 2 * k + g)
    keys["up"] = (x + k + g, pad, x + 2 * k + g, pad + k)
    for name, box in keys.items():
        on = name in lit
        d.rounded_rectangle(box, radius=round(6 * s), fill=ACCENT if on else KEY_FILL, outline=ACCENT if on else KEY_EDGE, width=max(1, round(1.5 * s)))
        glyph = (255, 255, 255) if on else KEY_GLYPH
        if name == "shift":
            f = _font(round(13 * s), bold=True)
            d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "Shift", font=f, fill=glyph, anchor="mm")
        else:
            _arrow(d, box, name, glyph)
    f, fb, fs = _font(round(14 * s)), _font(round(14 * s), bold=True), _font(round(12 * s))
    rows = [("", ("vx m/s", "vy m/s", "wz rad/s"), fs, TEXT_DIM), ("stick", tuple(f"{v:+.2f}" for v in stick), fb, ACCENT)]
    if robot is not None:
        rows.append(("robot", tuple(f"{v:+.2f}" for v in robot), f, TEXT))
    y = pad + round(2 * s)
    for label, vals, font, colour in rows:
        d.text((text_x, y), label, font=fb if label else fs, fill=colour if label else TEXT_DIM)
        for c, v in zip(col, vals):
            d.text((text_x + c, y), v, font=font, fill=colour, anchor="ra")
        y += round((16 if not label else 21) * s)
    if level is not None:
        d.text((pad, pad + round(2 * s)), f"speed {round(100 * level)}%", font=fs, fill=TEXT_DIM)
    return np.asarray(img)


def legend(items: list[tuple[str, tuple[int, int, int]]], scale: float = 1.0) -> np.ndarray:
    """(H, W, 4) RGBA: one coloured square and label per item, in a row."""
    s = scale
    f = _font(round(14 * s))
    pad, sq, gap = round(9 * s), round(13 * s), round(16 * s)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    widths = [sq + round(6 * s) + int(probe.textlength(t, font=f)) for t, _ in items]
    W, H = 2 * pad + sum(widths) + gap * (len(items) - 1), 2 * pad + round(16 * s)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=round(8 * s), fill=PANEL)
    x = pad
    for (t, c), w in zip(items, widths):
        cy = H / 2
        d.rounded_rectangle((x, cy - sq / 2, x + sq, cy + sq / 2), radius=round(3 * s), fill=c, outline=(120, 125, 135))
        d.text((x + sq + round(6 * s), cy), t, font=f, fill=TEXT, anchor="lm")
        x += w + gap
    return np.asarray(img)


def composite(frame: np.ndarray, rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-blend ``rgba`` onto the RGB ``frame`` (in place) with its top-left corner at (x, y)."""
    h, w = rgba.shape[:2]
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    region = frame[y : y + h, x : x + w].astype(np.float32)
    frame[y : y + h, x : x + w] = (a * rgba[..., :3] + (1 - a) * region).astype(np.uint8)
    return frame


def as_rgb(rgba: np.ndarray, background=(222, 226, 233)) -> np.ndarray:
    """RGBA -> RGB over a solid background (the MuJoCo viewer's image overlay takes RGB)."""
    out = np.empty(rgba.shape[:2] + (3,), np.uint8)
    out[:] = background
    return composite(out, rgba, 0, 0)
