"""Tests for `autoedit.content.extract.extract`.

Two layers, matching how the module is built:
- End-to-end tests against real tiny fixtures from `tests/conftest.py` (shared
  with the P2 extractor tests), confirming the real wiring produces a valid
  `ContentFeatures`.
- Unit tests that monkeypatch `ingest.probe` + the three extractors with
  fixed, synthetic feature objects, so the assembly/quantization logic
  itself (motion bucket boundaries, is_vertical, has_face, shot building) is
  exercised deterministically without depending on any particular real clip.
"""

from __future__ import annotations

import pytest

import importlib

from autoedit import ingest
from autoedit.extractors.audio import AudioFeatures
from autoedit.extractors.framing import FramingFeatures, ShotFraming
from autoedit.extractors.pacing import PacingFeatures
from autoedit.models import ContentFeatures, MediaAsset

# `autoedit.content`'s __init__ re-exports `extract` (the function) under the
# same name as this submodule, shadowing it as an attribute — so grab the
# actual submodule via `sys.modules` (through `importlib`) to monkeypatch its
# internals, rather than `from autoedit.content import extract`.
extract_module = importlib.import_module("autoedit.content.extract")
extract = extract_module.extract


def _media(width: int, height: int, duration: float, path: str) -> MediaAsset:
    return MediaAsset(
        path=path, type="video", duration=duration, width=width, height=height, fps=30.0, codec="h264", audio_channels=0
    )


def _patch_pipeline(monkeypatch, *, width, height, duration, pacing, framing, audio, path="raw/clip.mp4"):
    monkeypatch.setattr(extract_module.ingest, "probe", lambda p: _media(width, height, duration, path=str(p)))
    monkeypatch.setattr(extract_module, "extract_pacing", lambda p: pacing)
    monkeypatch.setattr(extract_module, "extract_framing", lambda p, shot_bounds=None: framing)
    monkeypatch.setattr(extract_module, "extract_audio", lambda p: audio)


# --- End-to-end, against real fixtures (shared with the P2 extractor tests) ---


def test_end_to_end_shape_on_multi_shot_fixture(multi_shot_clip_path):
    features = extract(multi_shot_clip_path)

    assert isinstance(features, ContentFeatures)
    assert len(features.shots) == 3
    assert [shot.id for shot in features.shots] == ["s1", "s2", "s3"]
    assert features.aspect == pytest.approx(1.0, abs=0.05)
    assert features.is_vertical is False
    assert features.motion in ("low", "med", "high")
    assert features.has_speech is False
    assert features.music_bpm is None


def test_end_to_end_detects_face_on_face_scene_fixture(face_scene_clip_path):
    features = extract(face_scene_clip_path)

    assert len(features.shots) == 2
    assert features.has_face is True
    assert features.shots[0].faces == 1
    assert features.shots[0].scale == "close"
    assert features.shots[1].faces == 0


def test_end_to_end_shot_spans_match_pacing_bounds(multi_shot_clip_path):
    from autoedit.extractors.pacing import extract_pacing

    pacing = extract_pacing(multi_shot_clip_path)
    features = extract(multi_shot_clip_path)

    for shot, (start, end) in zip(features.shots, pacing.shot_bounds):
        assert shot.in_ == pytest.approx(start)
        assert shot.out_ == pytest.approx(end)
        assert shot.dur == pytest.approx(end - start)


def test_propagates_typed_ingest_error_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(ingest.MediaNotFoundError):
        extract(missing)


# --- Unit tests: assembly + quantization logic via monkeypatched extractors ---


def _one_shot_pacing(motion: float, duration: float = 2.0) -> PacingFeatures:
    return PacingFeatures(
        duration=duration, cuts=[], shot_bounds=[(0.0, duration)], shot_lengths=[duration],
        shot_len_median=duration, motion_curve=[motion],
    )


def _one_shot_framing(aspect: float, faces: int = 0, scale: str = "wide") -> FramingFeatures:
    return FramingFeatures(
        aspect=aspect,
        shots=[
            ShotFraming(
                start=0.0, end=2.0, faces=faces, face_positions=[[0.5, 0.5]] * faces, scale=scale,
                camera="static", brightness=0.5, sharpness=100.0,
            )
        ],
    )


def _audio(bpm: float | None = None, has_speech: bool = False) -> AudioFeatures:
    return AudioFeatures(duration=2.0, bpm=bpm, beat_times=[], rms_curve=[], rms_times=[], has_speech=has_speech)


@pytest.mark.parametrize(
    "motion, expected_bucket",
    [(0.0, "low"), (0.005, "low"), (0.03, "med"), (0.05, "high"), (0.2, "high")],
)
def test_motion_bucket_quantization_boundaries(monkeypatch, motion, expected_bucket):
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=_one_shot_pacing(motion), framing=_one_shot_framing(0.5625), audio=_audio(),
    )
    features = extract("whatever.mp4")
    assert features.motion == expected_bucket


def test_is_vertical_true_when_height_exceeds_width(monkeypatch):
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=_one_shot_pacing(0.0), framing=_one_shot_framing(1080 / 1920), audio=_audio(),
    )
    features = extract("whatever.mp4")
    assert features.is_vertical is True
    assert features.aspect == pytest.approx(1080 / 1920)


def test_is_vertical_false_for_landscape(monkeypatch):
    _patch_pipeline(
        monkeypatch, width=1920, height=1080, duration=2.0,
        pacing=_one_shot_pacing(0.0), framing=_one_shot_framing(1920 / 1080), audio=_audio(),
    )
    features = extract("whatever.mp4")
    assert features.is_vertical is False


def test_has_face_true_when_any_shot_has_a_face(monkeypatch):
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=_one_shot_pacing(0.0), framing=_one_shot_framing(0.5625, faces=2, scale="close"), audio=_audio(),
    )
    features = extract("whatever.mp4")
    assert features.has_face is True
    assert features.shots[0].faces == 2
    assert features.shots[0].scale == "close"


def test_music_bpm_and_has_speech_pass_through_from_audio(monkeypatch):
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=_one_shot_pacing(0.0), framing=_one_shot_framing(0.5625), audio=_audio(bpm=128.0, has_speech=True),
    )
    features = extract("whatever.mp4")
    assert features.music_bpm == pytest.approx(128.0)
    assert features.has_speech is True


def test_shot_ids_are_sequential_and_one_indexed(monkeypatch):
    pacing = PacingFeatures(
        duration=3.0, cuts=[1.0, 2.0], shot_bounds=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
        shot_lengths=[1.0, 1.0, 1.0], shot_len_median=1.0, motion_curve=[0.0, 0.0, 0.0],
    )
    framing = FramingFeatures(
        aspect=0.5625,
        shots=[
            ShotFraming(start=s, end=e, faces=0, face_positions=[], scale="wide", camera="static", brightness=0.5, sharpness=50.0)
            for s, e in pacing.shot_bounds
        ],
    )
    _patch_pipeline(monkeypatch, width=1080, height=1920, duration=3.0, pacing=pacing, framing=framing, audio=_audio())

    features = extract("whatever.mp4")
    assert [shot.id for shot in features.shots] == ["s1", "s2", "s3"]
    assert all(shot.source == features.shots[0].source for shot in features.shots)
