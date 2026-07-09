"""Shared test helpers.

No test in this suite may require a real video or a render (see AGENTS.md
testing rule). Everything here loads plain JSON fixtures.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture(name: str) -> Any:
    """Load and deep-copy a fixture JSON file by filename (e.g. 'shot.json')."""
    with (FIXTURES_DIR / name).open() as f:
        data = json.load(f)
    return copy.deepcopy(data)


@pytest.fixture
def media_asset_data() -> dict:
    return load_fixture("media_asset.json")


@pytest.fixture
def shot_data() -> dict:
    return load_fixture("shot.json")


@pytest.fixture
def style_profile_data() -> dict:
    return load_fixture("style_profile.json")


@pytest.fixture
def content_features_data() -> dict:
    return load_fixture("content_features.json")


@pytest.fixture
def edit_plan_data() -> dict:
    return load_fixture("edit_plan.json")


@pytest.fixture
def timeline_data() -> dict:
    return load_fixture("timeline.json")


@pytest.fixture
def tiny_clip_path() -> Path:
    """Path to a tiny (~12KB, 1s) real mp4 fixture for exercising `ingest.probe`.

    Generated with:
        ffmpeg -f lavfi -i "testsrc=size=64x64:rate=10:duration=1" \\
               -f lavfi -i "sine=frequency=440:duration=1" -shortest \\
               -c:v libx264 -pix_fmt yuv420p -c:a aac -movflags +faststart \\
               fixtures/media/clip_tiny.mp4
    """
    return FIXTURES_DIR / "media" / "clip_tiny.mp4"


@pytest.fixture
def multi_shot_clip_path() -> Path:
    """Path to a tiny 3s/3-shot mp4 fixture for exercising `extractors.pacing`.

    Three 1s hard-cut segments (static red -> animated testsrc -> static
    blue), generated with:
        ffmpeg -f lavfi -i "color=c=red:size=64x64:rate=10:d=1" \\
               -f lavfi -i "testsrc=size=64x64:rate=10:d=1" \\
               -f lavfi -i "color=c=blue:size=64x64:rate=10:d=1" \\
               -filter_complex "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]" \\
               -map "[v]" -c:v libx264 -pix_fmt yuv420p -movflags +faststart \\
               fixtures/media/multi_shot.mp4
    """
    return FIXTURES_DIR / "media" / "multi_shot.mp4"


@pytest.fixture
def face_scene_clip_path() -> Path:
    """Path to a tiny 2s/2-shot mp4 fixture for exercising `extractors.framing`.

    Shot 1 (0-1s): a tight, public-domain NASA astronaut headshot (Chris
    Hadfield, `PD-USGov`) looped as a still frame — a real, detectable face
    filling most of the frame (a "close" shot). Shot 2 (1-2s): an animated
    `testsrc` pattern with no face (a "wide", moving shot). Generated with:
        ffmpeg -loop 1 -t 1 -i hadfield_crop.jpg \\
               -f lavfi -i "testsrc=size=200x200:rate=5:d=1" \\
               -filter_complex "[0:v]scale=200:200,fps=5,setsar=1[a];
                                [a][1:v]concat=n=2:v=1:a=0[v]" \\
               -map "[v]" -c:v libx264 -pix_fmt yuv420p -movflags +faststart \\
               fixtures/media/face_scene.mp4
    """
    return FIXTURES_DIR / "media" / "face_scene.mp4"


@pytest.fixture
def click_track_path() -> Path:
    """Path to a synthetic 4s, 120 BPM percussive click-track wav fixture.

    Exercises `extractors.audio`'s BPM/beat detection with a clean, tonal
    (low zero-crossing-rate) signal that should NOT trip the speech heuristic.
    Generated with a short exponentially-decaying 1kHz click repeated every
    0.5s via numpy + soundfile (see git history / this docstring for the
    exact synthesis if regeneration is ever needed).
    """
    return FIXTURES_DIR / "media" / "tone.wav"


@pytest.fixture
def speech_clip_path() -> Path:
    """Path to a ~5s real speech wav fixture, synthesized with macOS `say`.

    Exercises `extractors.audio`'s zero-crossing-rate speech heuristic with
    real (if synthetic-sounding) speech. Generated with:
        say -o speech.aiff "This is a test of the audio extractor..."
        ffmpeg -i speech.aiff -ar 22050 -ac 1 fixtures/media/speech.wav
    """
    return FIXTURES_DIR / "media" / "speech.wav"


@pytest.fixture
def static_caption_clip_path() -> Path:
    """Path to a 2s mp4 fixture with a sustained ('static') caption.

    Rendered with OpenCV (`cv2.putText` + `cv2.VideoWriter`, no `ffmpeg
    drawtext` dependency needed) so it's self-contained: "SUBSCRIBE NOW" in
    white on black, anchored near the bottom of the frame for the whole clip.
    """
    return FIXTURES_DIR / "media" / "static_caption.mp4"


@pytest.fixture
def karaoke_caption_clip_path() -> Path:
    """Path to a 2s mp4 fixture with rapid word-by-word ('karaoke') captions.

    Rendered with OpenCV: a new word drawn near the vertical middle of the
    frame every 0.2s (10 words over 2s at 5fps), exercising the text
    extractor's karaoke-vs-static style classifier.
    """
    return FIXTURES_DIR / "media" / "karaoke_caption.mp4"


@pytest.fixture
def video_features_fast_cuts_data() -> dict:
    """Fast-cut (0.5s shots), beat-synced, karaoke-captioned, music-only video."""
    return load_fixture("video_features_fast_cuts.json")


@pytest.fixture
def video_features_slow_static_data() -> dict:
    """Slow (3s shots) narrated video with sustained static captions, no music."""
    return load_fixture("video_features_slow_static.json")


@pytest.fixture
def video_features_mixed_data() -> dict:
    """Moderate-pace narrated video: one uncaptioned shot, two static captions."""
    return load_fixture("video_features_mixed.json")
