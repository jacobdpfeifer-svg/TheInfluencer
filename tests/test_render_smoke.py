"""Render-path smoke tests — the one place the REAL MoviePy composite runs in CI.

The rest of the suite deliberately never renders (see AGENTS.md testing rule),
which is why two real defects slipped through: an OpenCV 5.0 shadow that broke
face detection, and emoji overlays drawn with a font that has no emoji glyphs.
These smoke tests close that gap cheaply:

- `test_render_smoke_produces_a_valid_mp4` exercises the actual
  `renderer.render` path end to end on a real fixture clip, so a broken render
  (bad MoviePy API, font path, effect func, etc.) fails loudly instead of
  silently.
- `test_emoji_font_covers_common_emoji_glyphs` is a fast, render-free font
  coverage check. It is the test that would have caught the blank-emoji bug:
  Roboto has no emoji glyphs, so `EMOJI_FONT_PATH` must point at an
  emoji-capable font. It is marked `xfail(strict=True)` until that font is
  added — when you fix it, the strict xfail flips to a failure telling you to
  remove this marker.

Run just these with:  pytest -m render
Skip them in a fast unit loop with:  pytest -m "not render"
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from autoedit.models.timeline import Timeline, TimelineItem, Track
from autoedit import renderer

pytestmark = pytest.mark.render

_FIXTURE_CLIP = Path(__file__).resolve().parent.parent / "fixtures" / "media" / "multi_shot.mp4"

# A handful of emoji the heuristic/director actually emit or plausibly will.
_SAMPLE_EMOJI = {"fire": 0x1F525, "sparkles": 0x2728, "hundred": 0x1F4AF}


@pytest.mark.skipif(not _FIXTURE_CLIP.exists(), reason="multi_shot fixture missing")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_render_smoke_produces_a_valid_mp4(tmp_path: Path) -> None:
    """A minimal video+text timeline renders to a real, non-empty, probeable mp4."""
    timeline = Timeline(
        tracks=[
            Track(
                name="v1",
                kind="video",
                items=[
                    TimelineItem(
                        id="clip-s1",
                        start=0.0,
                        end=1.0,
                        payload={"shot": "s1", "source": str(_FIXTURE_CLIP), "in": 0.0, "out": 1.0},
                    )
                ],
            ),
            Track(
                name="captions",
                kind="text",
                items=[
                    TimelineItem(
                        id="cap-1",
                        start=0.0,
                        end=1.0,
                        payload={"content": "smoke test", "style": "static", "anchor": "bottom"},
                    )
                ],
            ),
        ]
    )

    out = tmp_path / "smoke.mp4"
    result = renderer.render(timeline, out, fps=24.0)

    assert result.exists(), "renderer did not create an output file"
    assert result.stat().st_size > 0, "renderer produced an empty file"

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, f"ffprobe rejected the output: {probe.stderr}"
    assert float(probe.stdout.strip()) > 0, "rendered mp4 has zero duration"


@pytest.mark.render
@pytest.mark.xfail(
    strict=True,
    reason="BLANK-EMOJI BUG: emoji overlays render with Roboto, which has no "
    "emoji glyphs. Add an emoji-capable font, point renderer.EMOJI_FONT_PATH at "
    "it, then remove this xfail marker.",
)
def test_emoji_font_covers_common_emoji_glyphs() -> None:
    """The font used to render emoji overlays must actually contain emoji glyphs."""
    from fontTools.ttLib import TTFont

    # The font the renderer uses for emoji. Today the renderer reuses the text
    # font (Roboto) for emoji; once fixed this should be a dedicated emoji font.
    emoji_font_path = getattr(renderer, "EMOJI_FONT_PATH", renderer.DEFAULT_FONT_PATH)
    cmap = TTFont(str(emoji_font_path)).getBestCmap()

    missing = [name for name, cp in _SAMPLE_EMOJI.items() if cp not in cmap]
    assert not missing, f"emoji font {emoji_font_path} is missing glyphs for: {missing}"
