"""heuristic_plan — the deterministic fallback director.

Per AGENTS.md's inviolable law #4: "Every AI decision has a deterministic
fallback... The heuristic path must produce a valid result with the LLM
fully stubbed." This is that path: rule-based, no LLM, and ALWAYS returns a
valid `EditPlan` for any valid `(StyleProfile, ContentFeatures)` pair.

Since the template system (`autoedit.templates`) landed, `heuristic_plan`
has two layers, both deterministic:

1. `match_template` picks the best-fitting builtin `Template` for this
   footage + style, and `fill_template` pours shots into its slots. This is
   the new, richer floor — see AGENTS.md's architecture note on
   `templates/`.
2. If no template scores above `_TEMPLATE_MATCH_MIN_SCORE`, fall back to
   `_simple_heuristic_plan` below — the original rule-based planner, kept
   verbatim as a floor under the floor so the system never fails to
   produce a valid plan, even for footage no template fits at all.

`_simple_heuristic_plan` is intentionally simple — "AI raises the ceiling;
rules are the floor." It keeps every shot (retiming/beat-syncing per the
learned style), and only adds text/effect/emoji ops when the StyleProfile's
own signals justify them, skipping anything it has no real basis to
fabricate (e.g. it never invents caption copy — a static placeholder stands
in for real text generation, which is an LLM-only concern per
`.cursor/rules/director.mdc`).
"""

from __future__ import annotations

from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditOp, EditPlan
from autoedit.models.style_profile import StyleProfile
from autoedit.templates.filler import fill_template
from autoedit.templates.matcher import match_template, score_template

# Below this, the best-scoring template still isn't a good enough fit to
# trust (see `templates.matcher.score_template`'s docstring for the scale)
# -- fall back to `_simple_heuristic_plan` rather than forcing a bad structure.
_TEMPLATE_MATCH_MIN_SCORE = 0.3

# A rule-based plan isn't a probabilistic estimate, so this is a fixed,
# documented "baseline" confidence rather than a measured quantity.
HEURISTIC_CONFIDENCE = 0.5

_DEFAULT_CAPTION_TEXT = "New content \U0001f4a5"
_DEFAULT_CAPTION_DURATION_SEC = 2.0
_DEFAULT_EMOJI_GLYPH = "\u2728"
# Above this, the style profile leans heavily enough on effects that a fade
# transition between the opening two shots is worth the (small) risk of
# overdoing it -- below it, cutting hard is the safer, style-neutral default.
_HIGH_EFFECT_FREQ_THRESHOLD = 0.5


def heuristic_plan(style: StyleProfile, features: ContentFeatures) -> EditPlan:
    """Deterministically build a valid `EditPlan`: template-fill if a template fits well, else the simple floor."""
    best_template = match_template(features, style)
    if score_template(best_template, features, style) >= _TEMPLATE_MATCH_MIN_SCORE:
        return fill_template(best_template, features, style)
    return _simple_heuristic_plan(style, features)


def _simple_heuristic_plan(style: StyleProfile, features: ContentFeatures) -> EditPlan:
    """The original, template-free rule-based planner -- the floor under the floor."""
    ops = [_cutter_op(style, features)]

    text_op = _text_op(style, features)
    if text_op is not None:
        ops.append(text_op)

    effect_op = _effect_op(style, features)
    if effect_op is not None:
        ops.append(effect_op)

    transition_op = _transition_op(style, features)
    if transition_op is not None:
        ops.append(transition_op)

    emoji_op = _emoji_op(style, features)
    if emoji_op is not None:
        ops.append(emoji_op)

    return EditPlan(ops=ops, confidence=HEURISTIC_CONFIDENCE)


def _cutter_op(style: StyleProfile, features: ContentFeatures) -> EditOp:
    keep = [shot.id for shot in features.shots]
    sync = "beat" if style.cut_on_beat and features.music_bpm else "none"
    params: dict[str, object] = {"keep": keep, "sync": sync}
    if sync == "beat":
        # Wire the actual detected beats through so `cutter`'s beat-snap
        # logic (gated on `sync == "beat" and beat_times`) isn't dead code.
        params["beat_times"] = features.beat_times
    return EditOp(tool="cutter", params=params)


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


def _transition_op(style: StyleProfile, features: ContentFeatures) -> EditOp | None:
    if style.effect_freq <= _HIGH_EFFECT_FREQ_THRESHOLD:
        return None
    if len(features.shots) < 2:
        return None
    first_shot, second_shot = features.shots[0], features.shots[1]
    return EditOp(tool="transition", params={"kind": "fade", "between": [first_shot.id, second_shot.id]})


def _emoji_op(style: StyleProfile, features: ContentFeatures) -> EditOp | None:
    del style  # no emoji-specific signal in StyleProfile yet; gated on features only.
    if not features.has_face:
        return None
    midpoint = sum(shot.dur for shot in features.shots) / 2
    return EditOp(tool="emoji", params={"glyph": _DEFAULT_EMOJI_GLYPH, "at": round(midpoint, 2)})
