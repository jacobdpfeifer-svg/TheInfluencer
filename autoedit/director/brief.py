"""build_brief — compress ContentFeatures + StyleProfile into one compact JSON brief.

Per `.cursor/rules/director.mdc`: "Build the brief as compact JSON (boolean
flags + quantized buckets...). Aim for one or two calls per video — never
per-frame or per-shot." This is a pure function: no LLM call happens here,
just a lossy, JSON-serializable summary of what the director needs to see.
"""

from __future__ import annotations

from typing import Any

from autoedit.models.content_features import ContentFeatures
from autoedit.models.shot import Shot
from autoedit.models.style_profile import StyleProfile

# Motion-magnitude bucket edges for per-shot briefing (mirrors the coarser
# whole-video bucketing `content.extract` already does — see AGENTS.md).
_LOW_MOTION_MAX = 0.3
_HIGH_MOTION_MIN = 0.6


def build_brief(features: ContentFeatures, style: StyleProfile, manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the compact JSON brief the director hands to the LLM (or heuristic)."""
    return {
        "features": _brief_features(features),
        "style": _brief_style(style),
        "tools": sorted(manifest.keys()),
    }


def _brief_features(features: ContentFeatures) -> dict[str, Any]:
    return {
        "aspect": features.aspect,
        "is_vertical": features.is_vertical,
        "has_speech": features.has_speech,
        "has_face": features.has_face,
        "motion": features.motion,
        "music_bpm": features.music_bpm,
        "shots": [_brief_shot(shot) for shot in features.shots],
    }


def _brief_shot(shot: Shot) -> dict[str, Any]:
    return {
        "id": shot.id,
        "dur": round(shot.dur, 2),
        "scale": shot.scale,
        "faces": shot.faces,
        "motion": _motion_bucket(shot.motion),
    }


def _motion_bucket(motion: float) -> str:
    if motion <= _LOW_MOTION_MAX:
        return "low"
    if motion >= _HIGH_MOTION_MIN:
        return "high"
    return "med"


def _brief_style(style: StyleProfile) -> dict[str, Any]:
    return {
        "aspect": style.aspect,
        "shot_len_median": style.shot_len_median,
        "shot_len_spread": style.shot_len_spread,
        "cut_on_beat": style.cut_on_beat,
        "caption_style_freq": {
            "karaoke": style.caption_style_freq.karaoke,
            "static": style.caption_style_freq.static,
        },
        "caption_density": style.caption_density,
        "text_amount": style.text_amount,
        "effect_freq": style.effect_freq,
        "sample_count": style.sample_count,
    }
