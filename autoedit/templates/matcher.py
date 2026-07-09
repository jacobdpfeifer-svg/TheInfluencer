"""matcher — match_template: pick the best-fitting Template for this footage + style.

Deterministic, no LLM (see AGENTS.md's architecture note on `templates/`): a
pure scoring function over already-computed `ContentFeatures`/`StyleProfile`,
never touching media or a Timeline.
"""

from __future__ import annotations

from autoedit.models.content_features import ContentFeatures
from autoedit.models.style_profile import StyleProfile
from autoedit.models.template import Template
from autoedit.templates import TEMPLATE_REGISTRY

# Every score starts here; the factors below add/subtract from it. Additive,
# not a normalized probability — match_template only cares about relative
# ranking, and `director.heuristic` compares the winner against a fixed
# minimum threshold to decide whether to trust it at all.
_BASE_SCORE = 1.0

_ASPECT_MISMATCH_PENALTY = 0.5
_SLOT_COUNT_MISMATCH_PENALTY_PER_SHOT = 0.1
_BEAT_SYNC_BONUS = 0.4
_TALKING_HEAD_BONUS = 0.4


def match_template(
    features: ContentFeatures, style: StyleProfile, registry: dict[str, Template] = TEMPLATE_REGISTRY
) -> Template:
    """Return the highest-scoring template in `registry` for this footage + style."""
    return max(registry.values(), key=lambda template: score_template(template, features, style))


def score_template(template: Template, features: ContentFeatures, style: StyleProfile) -> float:
    """How well `template` fits `features` + `style`. Higher is better; not bounded to [0, 1]."""
    del style  # no style-conditioned scoring signal in the current spec; kept for a uniform (template, features, style) API.

    score = _BASE_SCORE
    score -= _ASPECT_MISMATCH_PENALTY * _aspect_mismatch(template, features)
    score -= _SLOT_COUNT_MISMATCH_PENALTY_PER_SHOT * _slot_count_mismatch(template, features)
    score += _music_fit(template, features)
    score += _talking_head_fit(template, features)
    return score


def _aspect_mismatch(template: Template, features: ContentFeatures) -> float:
    return abs(_parse_aspect_ratio(template.aspect_ratio) - features.aspect)


def _parse_aspect_ratio(aspect_ratio: str) -> float:
    width_str, _, height_str = aspect_ratio.partition(":")
    return float(width_str) / float(height_str)


def _slot_count_mismatch(template: Template, features: ContentFeatures) -> float:
    return abs(len(template.slots) - len(features.shots))


def _music_fit(template: Template, features: ContentFeatures) -> float:
    has_music = features.music_bpm is not None
    if has_music and template.music.cut_on == "beat":
        return _BEAT_SYNC_BONUS
    if not has_music and template.music.required:
        # A music-required, beat-synced template is a poor fit for silent footage.
        return -_BEAT_SYNC_BONUS
    return 0.0


def _talking_head_fit(template: Template, features: ContentFeatures) -> float:
    is_talking_head_template = any(slot.role == "talking_head" for slot in template.slots)
    is_talking_head_footage = features.has_speech and features.has_face
    if is_talking_head_template and is_talking_head_footage:
        return _TALKING_HEAD_BONUS
    if is_talking_head_template and not is_talking_head_footage:
        # A talking-head template needs a face and speech to make sense at all.
        return -_TALKING_HEAD_BONUS
    return 0.0
