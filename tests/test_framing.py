"""Tests for `autoedit.extractors.framing.extract_framing`.

Exercised against a tiny real fixture (see `tests/conftest.py::face_scene_clip_path`)
since face/scale/camera detection is inherently a pixel-decoding measurement.
"""

from __future__ import annotations

import pytest

from autoedit.extractors.framing import FramingFeatures, extract_framing


def test_detects_a_close_face_shot_and_a_wide_faceless_shot(face_scene_clip_path):
    features = extract_framing(face_scene_clip_path, shot_bounds=[(0.0, 1.0), (1.0, 2.0)])

    assert isinstance(features, FramingFeatures)
    assert features.aspect == pytest.approx(1.0, abs=0.05)
    assert len(features.shots) == 2

    face_shot, faceless_shot = features.shots
    assert face_shot.faces == 1
    assert face_shot.scale == "close"
    assert len(face_shot.face_positions) == 1
    fx, fy = face_shot.face_positions[0]
    assert 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0

    assert faceless_shot.faces == 0
    assert faceless_shot.face_positions == []
    assert faceless_shot.scale == "wide"


def test_static_face_shot_vs_moving_faceless_shot(face_scene_clip_path):
    features = extract_framing(face_scene_clip_path, shot_bounds=[(0.0, 1.0), (1.0, 2.0)])

    face_shot, faceless_shot = features.shots
    assert face_shot.camera == "static"
    assert faceless_shot.camera == "moving"


def test_brightness_and_sharpness_are_in_expected_ranges(face_scene_clip_path):
    features = extract_framing(face_scene_clip_path, shot_bounds=[(0.0, 1.0), (1.0, 2.0)])

    for shot in features.shots:
        assert 0.0 <= shot.brightness <= 1.0
        assert shot.sharpness >= 0.0


def test_defaults_to_a_single_whole_video_shot_when_no_bounds_given(face_scene_clip_path):
    features = extract_framing(face_scene_clip_path)

    assert len(features.shots) == 1
    assert features.shots[0].start == 0.0
    assert features.shots[0].end == pytest.approx(2.0, abs=0.05)


def test_rejects_zero_duration_video(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(Exception):
        extract_framing(empty)
