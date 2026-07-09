"""Tests for autoedit/templates/matcher.py.

Pure fixture-JSON tests (per AGENTS.md's testing rule): `match_template`
never touches media or a Timeline, only ContentFeatures/StyleProfile.
"""

from __future__ import annotations

from autoedit.models.content_features import ContentFeatures
from autoedit.models.style_profile import StyleProfile
from autoedit.templates.matcher import match_template, score_template

_NEUTRAL_STYLE = StyleProfile.model_validate(
    {
        "aspect": 0.5625,
        "shot_len_median": 2.0,
        "shot_len_spread": 0.5,
        "cut_on_beat": False,
        "caption_style_freq": {"karaoke": 0.5, "static": 0.5},
        "caption_density": 0.0,
        "text_amount": 0.0,
        "effect_freq": 0.0,
        "sample_count": 1,
    }
)


def _shot(shot_id: str, *, motion: float = 0.1, faces: int = 0, scale: str = "wide", dur: float = 2.0) -> dict:
    return {
        "id": shot_id,
        "source": "raw.mp4",
        "in": 0.0,
        "out": dur,
        "dur": dur,
        "motion": motion,
        "brightness": 0.5,
        "sharpness": 50.0,
        "faces": faces,
        "scale": scale,
    }


def _features(**overrides) -> ContentFeatures:
    base = dict(
        aspect=0.5625,
        has_speech=False,
        music_bpm=None,
        shots=[_shot("s1")],
        motion="low",
        is_vertical=True,
        has_face=False,
    )
    base.update(overrides)
    return ContentFeatures.model_validate(base)


class TestMatchTemplate:
    def test_picks_talking_head_template_for_speech_and_face_footage(self) -> None:
        features = _features(
            has_speech=True,
            has_face=True,
            shots=[_shot("s1", faces=1, scale="close", dur=8.0), _shot("s2", faces=1, scale="close", dur=6.0)],
        )

        winner = match_template(features, _NEUTRAL_STYLE)

        assert winner.name == "talking_head_with_broll"

    def test_picks_punchy_beat_montage_for_high_motion_music_footage(self) -> None:
        features = _features(
            has_speech=False,
            has_face=False,
            music_bpm=128.0,
            motion="high",
            shots=[_shot(f"s{i}", motion=0.8, dur=1.0) for i in range(1, 6)],
        )

        winner = match_template(features, _NEUTRAL_STYLE)

        assert winner.name == "punchy_beat_montage"

    def test_returns_a_template_from_the_given_registry(self) -> None:
        features = _features()
        winner = match_template(features, _NEUTRAL_STYLE)
        assert winner.name in {"punchy_beat_montage", "talking_head_with_broll", "quick_montage"}


class TestScoreTemplate:
    def test_aspect_mismatch_lowers_the_score(self) -> None:
        from autoedit.templates import TEMPLATE_REGISTRY

        template = TEMPLATE_REGISTRY["quick_montage"]  # aspect_ratio "9:16"
        matching_features = _features(aspect=9 / 16)
        mismatched_features = _features(aspect=16 / 9)

        assert score_template(template, matching_features, _NEUTRAL_STYLE) > score_template(
            template, mismatched_features, _NEUTRAL_STYLE
        )

    def test_music_required_template_is_penalized_without_music(self) -> None:
        from autoedit.templates import TEMPLATE_REGISTRY

        template = TEMPLATE_REGISTRY["quick_montage"]  # music.required=True
        with_music = _features(music_bpm=128.0)
        without_music = _features(music_bpm=None)

        assert score_template(template, with_music, _NEUTRAL_STYLE) > score_template(
            template, without_music, _NEUTRAL_STYLE
        )
