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
from autoedit.extractors.audio import AudioFeatures, extract_audio
from autoedit.extractors.framing import FramingFeatures, ShotFraming
from autoedit.extractors.pacing import PacingFeatures
from autoedit.extractors.text import ShotText, TextFeatures
from autoedit.models import ContentFeatures, MediaAsset

# `autoedit.content`'s __init__ re-exports `extract` (the function) under the
# same name as this submodule, shadowing it as an attribute — so grab the
# actual submodule via `sys.modules` (through `importlib`) to monkeypatch its
# internals, rather than `from autoedit.content import extract`.
extract_module = importlib.import_module("autoedit.content.extract")
extract = extract_module.extract
extract_pool = extract_module.extract_pool


def _media(width: int, height: int, duration: float, path: str) -> MediaAsset:
    return MediaAsset(
        path=path, type="video", duration=duration, width=width, height=height, fps=30.0, codec="h264", audio_channels=0
    )


def _text(style: str = "none") -> TextFeatures:
    return TextFeatures(shots=[ShotText(start=0.0, end=1.0, events=[], style=style)])


def _patch_pipeline(monkeypatch, *, width, height, duration, pacing, framing, audio, text=None, path="raw/clip.mp4"):
    monkeypatch.setattr(extract_module.ingest, "probe", lambda p: _media(width, height, duration, path=str(p)))
    monkeypatch.setattr(extract_module, "extract_pacing", lambda p: pacing)
    monkeypatch.setattr(extract_module, "extract_framing", lambda p, shot_bounds=None: framing)
    monkeypatch.setattr(extract_module, "extract_audio", lambda p: audio)
    monkeypatch.setattr(extract_module, "extract_text", lambda p, shot_bounds=None: text if text is not None else _text())


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


def test_beat_times_pass_through_from_a_real_musical_audio_track(monkeypatch, click_track_path):
    """`ContentFeatures.beat_times` should carry through the audio extractor's real
    beat detection (tone.wav is a real 120 BPM click track — see conftest.py), not
    just an empty placeholder — this is what makes `cutter`'s beat-sync usable."""
    real_audio = extract_audio(click_track_path)
    assert len(real_audio.beat_times) >= 4  # sanity: tone.wav really has detectable beats

    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=_one_shot_pacing(0.0), framing=_one_shot_framing(0.5625), audio=real_audio,
    )
    features = extract("whatever.mp4")
    assert features.beat_times == real_audio.beat_times


def test_beat_times_defaults_to_empty_when_audio_has_none(monkeypatch):
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=_one_shot_pacing(0.0), framing=_one_shot_framing(0.5625), audio=_audio(),
    )
    features = extract("whatever.mp4")
    assert features.beat_times == []


def test_has_text_and_text_style_pass_through_when_static_captions_detected(monkeypatch):
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=_one_shot_pacing(0.0), framing=_one_shot_framing(0.5625), audio=_audio(), text=_text("static"),
    )
    features = extract("whatever.mp4")
    assert features.has_text is True
    assert features.text_style == "static"


def test_has_text_false_and_style_none_with_no_on_screen_text(monkeypatch):
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=_one_shot_pacing(0.0), framing=_one_shot_framing(0.5625), audio=_audio(), text=_text("none"),
    )
    features = extract("whatever.mp4")
    assert features.has_text is False
    assert features.text_style == "none"


def test_text_style_is_the_most_common_non_none_style_across_shots(monkeypatch):
    pacing = _one_shot_pacing(0.0)
    text = TextFeatures(
        shots=[
            ShotText(start=0.0, end=1.0, events=[], style="karaoke"),
            ShotText(start=1.0, end=2.0, events=[], style="karaoke"),
            ShotText(start=2.0, end=3.0, events=[], style="static"),
        ]
    )
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=pacing, framing=_one_shot_framing(0.5625), audio=_audio(), text=text,
    )
    features = extract("whatever.mp4")
    assert features.text_style == "karaoke"


def test_text_extractor_is_called_with_the_same_shot_bounds_as_framing(monkeypatch):
    pacing = _one_shot_pacing(0.0)
    calls = []
    _patch_pipeline(
        monkeypatch, width=1080, height=1920, duration=2.0,
        pacing=pacing, framing=_one_shot_framing(0.5625), audio=_audio(),
    )
    monkeypatch.setattr(extract_module, "extract_text", lambda p, shot_bounds=None: calls.append(shot_bounds) or _text())
    extract("whatever.mp4")
    assert calls == [pacing.shot_bounds]


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


# --- extract_pool: merging a pool of clips into one ContentFeatures ----------


def _shot_dict(shot_id: str, source: str, *, faces: int = 0, scale: str = "wide", motion: float = 0.1) -> dict:
    return {
        "id": shot_id, "source": source, "in": 0.0, "out": 2.0, "dur": 2.0,
        "motion": motion, "brightness": 0.5, "sharpness": 50.0, "faces": faces, "scale": scale,
    }


def _clip_features(**overrides) -> ContentFeatures:
    base = dict(
        aspect=0.5625, has_speech=False, music_bpm=None,
        shots=[_shot_dict("s1", "a.mp4")], motion="low",
        is_vertical=True, has_face=False, beat_times=[],
    )
    base.update(overrides)
    return ContentFeatures.model_validate(base)


def _patch_extract_by_path(monkeypatch, mapping: dict[str, ContentFeatures]) -> None:
    """Make `extract_pool`'s internal per-clip `extract` return canned features keyed by path."""
    monkeypatch.setattr(extract_module, "extract", lambda path: mapping[str(path)])


class TestExtractPool:
    def test_empty_paths_raises(self):
        with pytest.raises(ValueError, match="at least one clip"):
            extract_pool([])

    def test_concatenates_all_shots_across_clips(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(shots=[_shot_dict("s1", "a.mp4"), _shot_dict("s2", "a.mp4")]),
                "b.mp4": _clip_features(shots=[_shot_dict("s1", "b.mp4")]),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert len(features.shots) == 3

    def test_reindexes_shot_ids_globally_unique_in_clip_then_shot_order(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(shots=[_shot_dict("s1", "a.mp4"), _shot_dict("s2", "a.mp4")]),
                "b.mp4": _clip_features(shots=[_shot_dict("s1", "b.mp4"), _shot_dict("s2", "b.mp4")]),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert [shot.id for shot in features.shots] == ["s1", "s2", "s3", "s4"]
        assert len({shot.id for shot in features.shots}) == 4

    def test_preserves_each_shots_own_source_and_span(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(shots=[_shot_dict("s1", "a.mp4")]),
                "b.mp4": _clip_features(shots=[_shot_dict("s1", "b.mp4")]),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert [shot.source for shot in features.shots] == ["a.mp4", "b.mp4"]

    def test_has_face_and_has_speech_are_any_across_clips(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(has_face=False, has_speech=False),
                "b.mp4": _clip_features(has_face=True, has_speech=True),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert features.has_face is True
        assert features.has_speech is True

    def test_aspect_and_is_vertical_come_from_the_dominant_clip(self, monkeypatch):
        # b.mp4 has more shots, so it dominates the (single) output aspect.
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(aspect=0.5625, is_vertical=True, shots=[_shot_dict("s1", "a.mp4")]),
                "b.mp4": _clip_features(
                    aspect=16 / 9, is_vertical=False,
                    shots=[_shot_dict("s1", "b.mp4"), _shot_dict("s2", "b.mp4")],
                ),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert features.aspect == pytest.approx(16 / 9)
        assert features.is_vertical is False

    def test_music_bpm_is_the_first_music_y_clip_ignoring_speech_clips(self, monkeypatch):
        # a.mp4 has a BPM but also speech (not a music bed); b.mp4 is the real music.
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(music_bpm=90.0, has_speech=True),
                "b.mp4": _clip_features(music_bpm=128.0, has_speech=False),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert features.music_bpm == pytest.approx(128.0)

    def test_music_bpm_none_when_no_music_y_clip(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(music_bpm=90.0, has_speech=True),
                "b.mp4": _clip_features(music_bpm=None, has_speech=False),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert features.music_bpm is None

    def test_motion_is_the_highest_bucket_any_clip_reached(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(motion="low"),
                "b.mp4": _clip_features(motion="high"),
                "c.mp4": _clip_features(motion="med"),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4", "c.mp4"])
        assert features.motion == "high"

    def test_has_text_is_any_across_clips(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(has_text=False, text_style="none"),
                "b.mp4": _clip_features(has_text=True, text_style="static"),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert features.has_text is True

    def test_text_style_is_the_most_common_non_none_style_across_clips(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(has_text=True, text_style="karaoke"),
                "b.mp4": _clip_features(has_text=True, text_style="karaoke"),
                "c.mp4": _clip_features(has_text=True, text_style="static"),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4", "c.mp4"])
        assert features.text_style == "karaoke"

    def test_text_style_is_none_when_no_clip_has_text(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(has_text=False, text_style="none"),
                "b.mp4": _clip_features(has_text=False, text_style="none"),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert features.text_style == "none"
        assert features.has_text is False

    def test_beat_times_is_empty_for_a_pool(self, monkeypatch):
        _patch_extract_by_path(
            monkeypatch,
            {
                "a.mp4": _clip_features(beat_times=[0.5, 1.0, 1.5]),
                "b.mp4": _clip_features(beat_times=[0.25, 0.75]),
            },
        )
        features = extract_pool(["a.mp4", "b.mp4"])
        assert features.beat_times == []

    def test_single_clip_pool_matches_a_plain_extract(self, monkeypatch):
        one = _clip_features(shots=[_shot_dict("s1", "a.mp4"), _shot_dict("s2", "a.mp4")])
        _patch_extract_by_path(monkeypatch, {"a.mp4": one})
        features = extract_pool(["a.mp4"])
        assert [shot.id for shot in features.shots] == ["s1", "s2"]
        assert features.aspect == pytest.approx(one.aspect)

    def test_end_to_end_over_three_real_fixture_clips(
        self, multi_shot_clip_path, face_scene_clip_path, static_caption_clip_path
    ):
        features = extract_pool([multi_shot_clip_path, face_scene_clip_path, static_caption_clip_path])

        assert isinstance(features, ContentFeatures)
        # 3 + 2 + 1 shots across the three fixtures, re-IDed contiguously.
        assert len(features.shots) == 6
        assert [shot.id for shot in features.shots] == ["s1", "s2", "s3", "s4", "s5", "s6"]
        # face_scene has a detectable face, so the pool does too.
        assert features.has_face is True
        # Every shot still points at its own source file.
        sources = {shot.source for shot in features.shots}
        assert len(sources) == 3
