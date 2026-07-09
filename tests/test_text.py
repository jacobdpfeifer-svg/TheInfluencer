"""Tests for `autoedit.extractors.text.extract_text`.

Exercised against two tiny real fixtures rendered with OpenCV (see
`tests/conftest.py`): a sustained ("static") caption and a rapid
word-by-word ("karaoke") caption.
"""

from __future__ import annotations

import pytest

from autoedit.extractors.text import TextFeatures, extract_text


def test_static_caption_is_a_single_sustained_event(static_caption_clip_path):
    features = extract_text(static_caption_clip_path)

    assert isinstance(features, TextFeatures)
    assert len(features.shots) == 1
    shot = features.shots[0]

    assert shot.style == "static"
    assert len(shot.events) == 1
    event = shot.events[0]
    assert "SUBSCRIBE" in event.text
    assert event.anchor == "bottom"
    assert event.end - event.start > 1.0


def test_karaoke_caption_is_many_short_events(karaoke_caption_clip_path):
    features = extract_text(karaoke_caption_clip_path)

    shot = features.shots[0]
    assert shot.style == "karaoke"
    assert len(shot.events) >= 5
    for event in shot.events:
        assert event.anchor == "middle"
        assert event.end - event.start <= 0.6


def test_karaoke_words_appear_in_order(karaoke_caption_clip_path):
    features = extract_text(karaoke_caption_clip_path)

    texts = [e.text for e in features.shots[0].events]
    assert texts == sorted(texts, key=lambda t: features.shots[0].events[texts.index(t)].start)
    assert "HELLO" in texts
    assert texts.index("HELLO") < texts.index("DEMO")


def test_respects_explicit_shot_bounds(static_caption_clip_path):
    features = extract_text(static_caption_clip_path, shot_bounds=[(0.0, 1.0), (1.0, 2.0)])

    assert len(features.shots) == 2
    for shot in features.shots:
        assert shot.style == "static"
        assert len(shot.events) == 1


def test_rejects_zero_duration_video(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(Exception):
        extract_text(empty)
