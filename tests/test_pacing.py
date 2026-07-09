"""Tests for `autoedit.extractors.pacing.extract_pacing`.

Like ingest, pacing decodes real frames (via PySceneDetect + OpenCV), so it
is exercised against a tiny real fixture rather than pure JSON — see
`tests/conftest.py::multi_shot_clip_path` for how that fixture was generated.
"""

from __future__ import annotations

import pytest

from autoedit.extractors.pacing import PacingFeatures, extract_pacing


def test_detects_hard_cuts_in_multi_shot_fixture(multi_shot_clip_path):
    features = extract_pacing(multi_shot_clip_path)

    assert isinstance(features, PacingFeatures)
    assert features.duration == pytest.approx(3.0, abs=0.05)
    assert features.cuts == pytest.approx([1.0, 2.0], abs=0.15)
    assert len(features.shot_bounds) == 3
    assert len(features.shot_lengths) == 3
    assert len(features.motion_curve) == 3
    assert features.shot_len_median == pytest.approx(1.0, abs=0.15)


def test_motion_curve_is_higher_for_the_animated_middle_shot(multi_shot_clip_path):
    features = extract_pacing(multi_shot_clip_path)

    static_red, animated, static_blue = features.motion_curve
    assert animated > static_red
    assert animated > static_blue
    assert static_red == pytest.approx(0.0, abs=1e-6)
    assert static_blue == pytest.approx(0.0, abs=1e-6)


def test_single_shot_clip_has_no_cuts(tiny_clip_path):
    features = extract_pacing(tiny_clip_path)

    assert features.cuts == []
    assert len(features.shot_bounds) == 1
    assert features.shot_bounds[0] == pytest.approx((0.0, features.duration), abs=0.05)


def test_shot_bounds_cover_the_full_duration(multi_shot_clip_path):
    features = extract_pacing(multi_shot_clip_path)

    assert features.shot_bounds[0][0] == 0.0
    assert features.shot_bounds[-1][1] == pytest.approx(features.duration, abs=1e-6)
    for (_, end), (next_start, _) in zip(features.shot_bounds, features.shot_bounds[1:]):
        assert end == pytest.approx(next_start, abs=1e-6)


def test_rejects_zero_duration_video(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(Exception):
        extract_pacing(empty)
