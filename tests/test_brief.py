"""Tests for `autoedit.director.brief.build_brief`.

Pure fixture-JSON tests (per AGENTS.md's testing rule) — `build_brief` never
touches media or calls an LLM; it's a lossy Timeline/ContentFeatures ->
compact-JSON translation only.
"""

from __future__ import annotations

from autoedit.director.brief import build_brief
from autoedit.models.content_features import ContentFeatures
from autoedit.models.style_profile import StyleProfile
from autoedit.subsystems import TOOL_MANIFEST


def _features(content_features_data: dict, **overrides) -> ContentFeatures:
    data = {**content_features_data, **overrides}
    return ContentFeatures.model_validate(data)


def _style(style_profile_data: dict) -> StyleProfile:
    return StyleProfile.model_validate(style_profile_data)


def test_brief_has_features_style_tools_and_template_keys(content_features_data, style_profile_data):
    brief = build_brief(_features(content_features_data), _style(style_profile_data), TOOL_MANIFEST)
    assert set(brief.keys()) == {"features", "style", "tools", "template"}
    assert brief["tools"] == sorted(TOOL_MANIFEST.keys())
    assert isinstance(brief["template"], str)


def test_brief_features_includes_has_text_and_text_style(content_features_data, style_profile_data):
    features = _features(content_features_data, has_text=True, text_style="karaoke")
    brief = build_brief(features, _style(style_profile_data), TOOL_MANIFEST)
    assert brief["features"]["has_text"] is True
    assert brief["features"]["text_style"] == "karaoke"


def test_brief_features_defaults_to_no_text(content_features_data, style_profile_data):
    brief = build_brief(_features(content_features_data), _style(style_profile_data), TOOL_MANIFEST)
    assert brief["features"]["has_text"] is False
    assert brief["features"]["text_style"] == "none"


def test_brief_features_shape(content_features_data, style_profile_data):
    brief = build_brief(_features(content_features_data), _style(style_profile_data), TOOL_MANIFEST)
    expected_keys = {"aspect", "is_vertical", "has_speech", "has_face", "motion", "music_bpm", "has_text", "text_style", "shots"}
    assert set(brief["features"].keys()) == expected_keys
    assert len(brief["features"]["shots"]) == len(content_features_data["shots"])


def test_brief_shot_shape(content_features_data, style_profile_data):
    brief = build_brief(_features(content_features_data), _style(style_profile_data), TOOL_MANIFEST)
    shot = brief["features"]["shots"][0]
    assert set(shot.keys()) == {"id", "dur", "scale", "faces", "motion"}


def test_brief_style_shape(content_features_data, style_profile_data):
    brief = build_brief(_features(content_features_data), _style(style_profile_data), TOOL_MANIFEST)
    expected_keys = {
        "aspect", "shot_len_median", "shot_len_spread", "cut_on_beat",
        "caption_style_freq", "caption_density", "text_amount", "effect_freq", "sample_count",
    }
    assert set(brief["style"].keys()) == expected_keys
