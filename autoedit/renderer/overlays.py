"""Text, emoji, and flash overlay building.

All overlay clip construction lives here — static and karaoke text, emoji
rasterized via Pillow, and the flash effect overlay (which is composited
like a text/emoji overlay rather than applied as a per-clip transform).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._fonts import EMOJI_FONT_PATH
from .plan import EmojiOverlay, TextOverlay

_ANCHOR_POSITIONS: dict[str, tuple[str, str]] = {
    "top": ("center", "top"),
    "middle": ("center", "center"),
    "bottom": ("center", "bottom"),
}

# EMOJI_FONT_PATH embeds exactly ONE bitmap (CBDT/CBLC) strike, at this pixel
# size — there is no "regular size" to ask for, this is it. MoviePy's
# TextClip can't use this font at all: it always probes the font via
# `PIL.ImageFont.truetype(font)` with NO size argument before ever looking at
# our `font_size`, which raises "invalid pixel size" for any single-strike
# bitmap font. So emoji are rasterized directly via Pillow (at this, the
# font's one real size) in `_build_emoji_clip` instead of going through
# `TextClip` — see that function's docstring.
_EMOJI_STRIKE_PIXEL_SIZE = 109
# Visual size (pixels, tallest dimension) emoji are downscaled to on the
# output frame after rasterizing at the strike's native (much larger) size.
_EMOJI_DISPLAY_SIZE_PX = 72

_FLASH_DURATION_SEC = 0.05


def _build_text_clip(overlay: TextOverlay, *, font_path: str | Path) -> Any:
    from moviepy import TextClip

    position = _ANCHOR_POSITIONS.get(overlay.anchor, _ANCHOR_POSITIONS["bottom"])
    clip = TextClip(font=str(font_path), text=overlay.content, font_size=48, color="white")
    return clip.with_duration(overlay.end - overlay.start).with_position(position)


def _build_karaoke_clips(overlay: TextOverlay, *, font_path: str | Path) -> list[Any]:
    """Split `overlay.content` into words, each shown for an even slice of `[overlay.start, overlay.end)`.

    "karaoke" (per AGENTS.md's `TextStyle`: "rapid word-by-word") means one
    word on screen at a time, not the whole phrase held for the whole span —
    the static-caption path (`_build_text_clip`) already covers "the whole
    phrase, the whole span." An empty/whitespace-only `content` yields no
    clips at all rather than dividing by zero.
    """
    from moviepy import TextClip

    words = overlay.content.split()
    if not words:
        return []

    position = _ANCHOR_POSITIONS.get(overlay.anchor, _ANCHOR_POSITIONS["bottom"])
    per_word_duration = (overlay.end - overlay.start) / len(words)

    clips = []
    for index, word in enumerate(words):
        clip = TextClip(font=str(font_path), text=word, font_size=48, color="white")
        clip = clip.with_duration(per_word_duration).with_position(position)
        clips.append(clip.with_start(overlay.start + index * per_word_duration))
    return clips


def _build_emoji_clip(overlay: EmojiOverlay) -> Any:
    """Rasterize `overlay.glyph` via Pillow directly and wrap it in an ImageClip.

    Deliberately does NOT use MoviePy's `TextClip` (unlike `_build_text_clip`):
    `EMOJI_FONT_PATH` is a bitmap-strike color emoji font with only ONE
    embedded size (`_EMOJI_STRIKE_PIXEL_SIZE`), and `TextClip.__init__`
    unconditionally probes any font via `PIL.ImageFont.truetype(font)` with
    no size argument (Pillow's default size, 10) before it ever looks at our
    `font_size` — which raises `ValueError: ... invalid pixel size` for a
    single-strike font no matter what size we ask for. Pillow can still draw
    this font perfectly well directly (`embedded_color=True`) when we
    instantiate it ourselves at its one real size, so we do that and hand
    MoviePy the resulting RGBA bitmap as an `ImageClip` instead.
    """
    import numpy as np
    from moviepy import ImageClip
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(EMOJI_FONT_PATH), _EMOJI_STRIKE_PIXEL_SIZE)
    canvas_size = _EMOJI_STRIKE_PIXEL_SIZE * 2
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text((0, 0), overlay.glyph, font=font, embedded_color=True)
    glyph_bbox = canvas.getbbox() or (0, 0, 1, 1)  # fall back to a 1x1 pixel if nothing was drawn
    glyph_image = canvas.crop(glyph_bbox)

    clip = ImageClip(np.array(glyph_image)).resized(height=_EMOJI_DISPLAY_SIZE_PX)
    return clip.with_duration(overlay.end - overlay.start).with_position(("center", "center"))


def _build_flash_clip(*, size: tuple[int, int]) -> Any:
    """A `_FLASH_DURATION_SEC` solid-white frame, composited at the flashing shot's start time.

    Unlike the other effects, `flash` isn't a per-clip transform (it doesn't
    change how the shot itself plays) — it's an extra overlay, exactly like
    a text/emoji overlay, which is why it's built and composited in `compose.render`
    rather than living in `_EFFECT_FUNCS`/`_apply_effects_to_clip`.
    """
    from moviepy import ColorClip

    return ColorClip(size=size, color=(255, 255, 255)).with_duration(_FLASH_DURATION_SEC)
