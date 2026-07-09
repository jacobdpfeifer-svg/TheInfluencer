"""Tests for `autoedit.extractors.audio.extract_audio`.

Exercised against two tiny real audio fixtures (see `tests/conftest.py`):
a 120 BPM percussive click track (music-like, low zero-crossing rate) and a
short real speech clip synthesized with macOS `say` (high, variable
zero-crossing rate).
"""

from __future__ import annotations

import pytest

from autoedit.extractors.audio import AudioFeatures, extract_audio


def test_detects_tempo_and_music_on_click_track(click_track_path):
    features = extract_audio(click_track_path)

    assert isinstance(features, AudioFeatures)
    assert features.duration == pytest.approx(4.0, abs=0.05)
    assert features.bpm is not None
    assert features.bpm == pytest.approx(120.0, rel=0.1)
    assert len(features.beat_times) >= 4
    assert features.has_speech is False


def test_detects_speech_on_spoken_clip(speech_clip_path):
    features = extract_audio(speech_clip_path)

    assert features.has_speech is True
    assert features.duration > 0


def test_rms_curve_is_aligned_and_nonnegative(click_track_path):
    features = extract_audio(click_track_path)

    assert len(features.rms_curve) == len(features.rms_times)
    assert len(features.rms_curve) > 0
    assert all(v >= 0.0 for v in features.rms_curve)
    assert features.rms_times == sorted(features.rms_times)


def test_beat_times_fall_within_duration(click_track_path):
    features = extract_audio(click_track_path)

    for beat in features.beat_times:
        assert 0.0 <= beat <= features.duration + 0.5


def test_empty_audio_file_returns_safe_defaults(tmp_path):
    silent = tmp_path / "silent.wav"
    import numpy as np
    import soundfile as sf

    sf.write(silent, np.zeros(0, dtype="float32"), 22050)

    features = extract_audio(silent)
    assert features.duration == 0.0
    assert features.bpm is None
    assert features.beat_times == []
    assert features.has_speech is False


def test_video_with_no_audio_stream_returns_safe_defaults(multi_shot_clip_path):
    # A raw video clip with no audio track at all (e.g. our silent pacing
    # fixture) is a normal shape for this pipeline, not an error.
    features = extract_audio(multi_shot_clip_path)
    assert features.duration == 0.0
    assert features.bpm is None
    assert features.has_speech is False


def test_rejects_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises(FileNotFoundError):
        extract_audio(missing)
