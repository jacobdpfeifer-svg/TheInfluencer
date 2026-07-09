"""heuristic_plan — the deterministic fallback director.

Per AGENTS.md's inviolable law #4: "Every AI decision has a deterministic
fallback... The heuristic path must produce a valid result with the LLM
fully stubbed." This is that path: rule-based, no LLM, and ALWAYS returns a
valid `EditPlan` for any valid `(StyleProfile, ContentFeatures)` pair.

It is intentionally simple — "AI raises the ceiling; rules are the floor."
It keeps every shot (retiming/beat-syncing per the learned style), and only
adds text/effect/emoji ops when the StyleProfile's own signals justify them,
skipping anything it has no real basis to fabricate (e.g. it never invents
caption copy — a static placeholder stands in for real text generation,
which is an LLM-only concern per `.cursor/rules/director.mdc`).
"""

from __future__ import annotations

from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditOp, EditPlan
from autoedit.models.style_profile import StyleProfile

# A rule-based plan isn't a probabilistic estimate, so this is a fixed,
# documented "baseline" confidence rather than a measured quantity.
HEURISTIC_CONFIDENCE = 0.5

_DEFAULT_CAPTION_TEXT = "New content \U0001f4a5"
_DEFAULT_CAPTION_DURATION_SEC = 2.0
_DEFAULT_EMOJI_GLYPH = "\u2728"


def heuristic_plan(style: StyleProfile, features: ContentFeatures) -> EditPlan:
    """Deterministically build a valid `EditPlan` from learned style + this footage's features."""
    ops = [_cutter_op(style, features)]

    text_op = _text_op(style, features)
    if text_op is not None:
        ops.append(text_op)

    effect_op = _effect_op(style, features)
    if effect_op is not None:
        ops.append(effect_op)

    emoji_op = _emoji_op(style, features)
    if emoji_op is not None:
        ops.append(emoji_op)

    return EditPlan(ops=ops, confidence=HEURISTIC_CONFIDENCE)


def _cutter_op(style: StyleProfile, features: ContentFeatures) -> EditOp:
    keep = [shot.id for shot in features.shots]
    sync = "beat" if style.cut_on_beat and features.music_bpm else "none"
    return EditOp(tool="cutter", params={"keep": keep, "sync": sync})


def _text_op(style: StyleProfile, features: ContentFeatures) -> EditOp | None:
    if style.text_amount <= 0:
        return None
    caption_style = "karaoke" if style.caption_style_freq.karaoke >= style.caption_style_freq.static else "static"
    first_shot_dur = features.shots[0].dur
    duration = min(_DEFAULT_CAPTION_DURATION_SEC, first_shot_dur)
    return EditOp(
        tool="text",
        params={"content": _DEFAULT_CAPTION_TEXT, "style": caption_style, "anchor": "bottom", "start": 0.0, "duration": duration},
    )


def _effect_op(style: StyleProfile, features: ContentFeatures) -> EditOp | None:
    if style.effect_freq <= 0:
        return None
    close_shot = next((shot for shot in features.shots if shot.scale == "close"), None)
    if close_shot is None:
        return None
    return EditOp(tool="effect", params={"kind": "zoom_in", "shot": close_shot.id})


def _emoji_op(style: StyleProfile, features: ContentFeatures) -> EditOp | None:
    del style  # no emoji-specific signal in StyleProfile yet; gated on features only.
    if not features.has_face:
        return None
    midpoint = sum(shot.dur for shot in features.shots) / 2
    return EditOp(tool="emoji", params={"glyph": _DEFAULT_EMOJI_GLYPH, "at": round(midpoint, 2)})
