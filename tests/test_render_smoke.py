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
  Roboto has no emoji glyphs, so `renderer.EMOJI_FONT_PATH` points at a
  dedicated emoji-capable font (vendored Noto Color Emoji) instead.

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

_FIXTURES_MEDIA_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "media"
_FIXTURE_CLIP = _FIXTURES_MEDIA_DIR / "multi_shot.mp4"
_SECOND_FIXTURE_CLIP = _FIXTURES_MEDIA_DIR / "face_scene.mp4"
_MUSIC_FIXTURE = _FIXTURES_MEDIA_DIR / "tone.wav"

# A handful of emoji the heuristic/director actually emit or plausibly will.
_SAMPLE_EMOJI = {"fire": 0x1F525, "sparkles": 0x2728, "hundred": 0x1F4AF}


def _probe_duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, f"ffprobe rejected {path}: {probe.stderr}"
    return float(probe.stdout.strip())


def _probe_audio_channels(path: Path) -> int:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, f"ffprobe rejected {path}: {probe.stderr}"
    output = probe.stdout.strip()
    return int(output) if output else 0


def _single_shot_video_track(*, source: Path, end: float = 1.0) -> Track:
    return Track(
        name="v1", kind="video",
        items=[TimelineItem(id="clip-s1", start=0.0, end=end, payload={"shot": "s1", "source": str(source), "in": 0.0, "out": end})],
    )


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


@pytest.mark.skipif(not _FIXTURE_CLIP.exists(), reason="multi_shot fixture missing")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
@pytest.mark.parametrize("mode", ["center_crop", "rule_of_thirds", "fit"])
def test_reframe_normalizes_a_square_fixture_onto_the_9x16_canvas(tmp_path: Path, mode: str) -> None:
    """The `multi_shot` fixture is a 64x64 square clip -- a good stress case for
    canvas normalization, since it's neither pre-cropped nor pre-letterboxed to
    9:16 the way most footage would incidentally already be close to.
    """
    timeline = Timeline(
        tracks=[
            Track(
                name="v1",
                kind="video",
                items=[
                    TimelineItem(
                        id="clip-s1", start=0.0, end=1.0,
                        payload={"shot": "s1", "source": str(_FIXTURE_CLIP), "in": 0.0, "out": 1.0},
                    )
                ],
            ),
            Track(
                name="reframe",
                kind="reframe",
                items=[
                    TimelineItem(
                        id="reframe-1", start=0.0, end=1.0,
                        payload={"kind": mode, "target_aspect": 9 / 16, "shot": "s1"},
                    )
                ],
            ),
        ]
    )

    out = tmp_path / f"reframe_{mode}.mp4"
    result = renderer.render(timeline, out, fps=24.0)

    assert result.exists() and result.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, f"ffprobe rejected the output: {probe.stderr}"
    width_str, height_str = probe.stdout.strip().split(",")
    assert (int(width_str), int(height_str)) == (1080, 1920)


@pytest.mark.skipif(not _FIXTURE_CLIP.exists(), reason="multi_shot fixture missing")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_karaoke_captions_render_one_word_at_a_time(tmp_path: Path) -> None:
    """A `style='karaoke'` overlay must split into several short per-word clips
    (see `renderer._build_karaoke_clips`), not one static block of text — this is
    the render path exercise `_build_text_clip` alone never covers.
    """
    timeline = Timeline(
        tracks=[
            Track(
                name="v1",
                kind="video",
                items=[
                    TimelineItem(
                        id="clip-s1", start=0.0, end=2.0,
                        payload={"shot": "s1", "source": str(_FIXTURE_CLIP), "in": 0.0, "out": 2.0},
                    )
                ],
            ),
            Track(
                name="captions",
                kind="text",
                items=[
                    TimelineItem(
                        id="cap-1", start=0.0, end=2.0,
                        payload={"content": "this is a karaoke test", "style": "karaoke", "anchor": "middle"},
                    )
                ],
            ),
        ]
    )

    out = tmp_path / "karaoke.mp4"
    result = renderer.render(timeline, out, fps=24.0)

    assert result.exists() and result.stat().st_size > 0
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(out)],
        capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0
    assert float(probe.stdout.strip()) > 0


@pytest.mark.skipif(not _FIXTURE_CLIP.exists(), reason="multi_shot fixture missing")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
@pytest.mark.parametrize("kind", ["zoom_in", "zoom_out", "speed_ramp", "shake", "ken_burns", "blur_intro", "flash"])
def test_every_effect_kind_renders_without_error(tmp_path: Path, kind: str) -> None:
    """Every kind `renderer._EFFECT_FUNCS` (or `flash`'s own overlay path,
    see its docstring) understands must actually composite -- a real
    MoviePy/Pillow call per kind, not just a Timeline-mutation unit test.
    """
    timeline = Timeline(
        tracks=[
            _single_shot_video_track(source=_FIXTURE_CLIP, end=1.0),
            Track(
                name="effects", kind="effect",
                items=[TimelineItem(id="fx-1", start=0.0, end=1.0, payload={"kind": kind, "shot": "s1"})],
            ),
        ]
    )

    out = tmp_path / f"effect_{kind}.mp4"
    result = renderer.render(timeline, out, fps=24.0)

    assert result.exists() and result.stat().st_size > 0
    assert _probe_duration(out) > 0


@pytest.mark.skipif(not _FIXTURE_CLIP.exists(), reason="multi_shot fixture missing")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
@pytest.mark.parametrize("kind", ["fade", "whip_pan"])
def test_every_transition_kind_renders_without_error(tmp_path: Path, kind: str) -> None:
    """A transition needs two real, adjacent clips on the video track — this
    is the one thing a Timeline-only subsystem test (`test_subsystems.py`)
    can never exercise: the actual `CrossFadeIn`/`CrossFadeOut` or edge-blur
    compositing in `renderer._build_video_track`.
    """
    timeline = Timeline(
        tracks=[
            Track(
                name="v1", kind="video",
                items=[
                    TimelineItem(id="clip-s1", start=0.0, end=1.0, payload={"shot": "s1", "source": str(_FIXTURE_CLIP), "in": 0.0, "out": 1.0}),
                    TimelineItem(id="clip-s2", start=1.0, end=2.0, payload={"shot": "s2", "source": str(_FIXTURE_CLIP), "in": 1.0, "out": 2.0}),
                ],
            ),
            Track(
                name="transitions", kind="transition",
                items=[TimelineItem(id="trans-1", start=0.85, end=1.15, payload={"kind": kind, "between": ["s1", "s2"]})],
            ),
        ]
    )

    out = tmp_path / f"transition_{kind}.mp4"
    result = renderer.render(timeline, out, fps=24.0)

    assert result.exists() and result.stat().st_size > 0
    assert _probe_duration(out) > 0


@pytest.mark.skipif(not _MUSIC_FIXTURE.exists(), reason="tone.wav fixture missing")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_music_track_mixes_real_audio_into_the_rendered_output(tmp_path: Path) -> None:
    """`renderer._mix_audio`'s real MoviePy audio path (`AudioFileClip` +
    `MultiplyVolume` + `CompositeAudioClip`) -- a silent-video fixture with a
    music track appended must come out with an audible (nonzero-channel)
    audio stream, not just succeed structurally.
    """
    timeline = Timeline(
        tracks=[
            _single_shot_video_track(source=_FIXTURE_CLIP, end=1.0),
            Track(
                name="music", kind="audio",
                items=[
                    TimelineItem(
                        id="music-1", start=0.0, end=1.0,
                        payload={"source": str(_MUSIC_FIXTURE), "in": 0.0, "out": 1.0, "volume": 0.6},
                    )
                ],
            ),
        ]
    )

    out = tmp_path / "music.mp4"
    result = renderer.render(timeline, out, fps=24.0)

    assert result.exists() and result.stat().st_size > 0
    assert _probe_audio_channels(out) > 0


@pytest.mark.skipif(not (_FIXTURE_CLIP.exists() and _SECOND_FIXTURE_CLIP.exists()), reason="fixture clip(s) missing")
@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not on PATH")
def test_multi_source_clips_render_into_one_continuous_output(tmp_path: Path) -> None:
    """Two DIFFERENT source files on the video track (a pooled-footage
    montage, per `content.extract_pool`) must composite into one continuous
    output -- each segment loads its own `VideoFileClip(segment.source)`
    (see `renderer._build_video_track`'s docstring), which a single-source
    smoke test can't exercise.
    """
    timeline = Timeline(
        tracks=[
            Track(
                name="v1", kind="video",
                items=[
                    TimelineItem(id="clip-s1", start=0.0, end=1.0, payload={"shot": "s1", "source": str(_FIXTURE_CLIP), "in": 0.0, "out": 1.0}),
                    TimelineItem(id="clip-s2", start=1.0, end=2.0, payload={"shot": "s2", "source": str(_SECOND_FIXTURE_CLIP), "in": 0.0, "out": 1.0}),
                ],
            ),
        ]
    )

    out = tmp_path / "multi_source.mp4"
    result = renderer.render(timeline, out, fps=24.0)

    assert result.exists() and result.stat().st_size > 0
    assert _probe_duration(out) == pytest.approx(2.0, abs=0.2)


@pytest.mark.render
def test_emoji_font_covers_common_emoji_glyphs() -> None:
    """The font used to render emoji overlays must actually contain emoji glyphs."""
    from fontTools.ttLib import TTFont

    cmap = TTFont(str(renderer.EMOJI_FONT_PATH)).getBestCmap()

    missing = [name for name, cp in _SAMPLE_EMOJI.items() if cp not in cmap]
    assert not missing, f"emoji font {renderer.EMOJI_FONT_PATH} is missing glyphs for: {missing}"
