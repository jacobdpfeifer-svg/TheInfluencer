"""Tests for autoedit/templates/captions.py.

Pure fixture-JSON tests (per AGENTS.md's testing rule): `generate_caption_copy`
is rule-based text SELECTION over `ContentFeatures`, no LLM/media involved.
"""

from __future__ import annotations

from autoedit.models.content_features import ContentFeatures
from autoedit.templates.captions import generate_caption_copy


def _features(**overrides) -> ContentFeatures:
    base = dict(
        aspect=0.5625,
        has_speech=False,
        music_bpm=None,
        shots=[
            {
                "id": "s1", "source": "raw.mp4", "in": 0.0, "out": 2.0, "dur": 2.0,
                "motion": 0.1, "brightness": 0.5, "sharpness": 50.0, "faces": 0, "scale": "wide",
            }
        ],
        motion="low",
        is_vertical=True,
        has_face=False,
    )
    base.update(overrides)
    return ContentFeatures.model_validate(base)


class TestUnrecognizedTagsPassThrough:
    def test_unknown_placeholder_is_returned_verbatim(self) -> None:
        features = _features()
        assert generate_caption_copy(features, "3 months in") == "3 months in"

    def test_case_insensitive_tag_matching(self) -> None:
        features = _features(has_speech=True)
        assert generate_caption_copy(features, "title") == generate_caption_copy(features, "TITLE")


class TestTitleTag:
    def test_high_motion_face_wins_top_priority(self) -> None:
        features = _features(has_face=True, motion="high", has_speech=True, music_bpm=130.0)
        assert generate_caption_copy(features, "TITLE") == "wait for it \U0001f440"

    def test_high_bpm_without_face_or_motion(self) -> None:
        features = _features(music_bpm=128.0)
        assert generate_caption_copy(features, "TITLE") == "this hits different \U0001f525"

    def test_speech_without_face_or_music(self) -> None:
        features = _features(has_speech=True)
        assert generate_caption_copy(features, "TITLE") == "the story so far..."

    def test_face_alone_falls_through_to_face_rule(self) -> None:
        features = _features(has_face=True)
        assert generate_caption_copy(features, "TITLE") == "you had to be there \u2728"

    def test_neutral_features_use_the_fallback(self) -> None:
        features = _features()
        assert generate_caption_copy(features, "TITLE") == "New content \U0001f4a5"


class TestCtaTag:
    def test_music_present_wins_top_priority(self) -> None:
        features = _features(music_bpm=100.0, has_face=True)
        assert generate_caption_copy(features, "CTA") == "turn the sound on \U0001f3b6"

    def test_face_without_music(self) -> None:
        features = _features(has_face=True)
        assert generate_caption_copy(features, "CTA") == "follow for more like this \u2728"

    def test_neutral_features_use_the_fallback(self) -> None:
        features = _features()
        assert generate_caption_copy(features, "CTA") == "follow for more \u2728"


class TestDeterminism:
    def test_same_input_always_yields_the_same_output(self) -> None:
        features = _features(has_speech=True)
        results = {generate_caption_copy(features, "TITLE") for _ in range(5)}
        assert len(results) == 1
