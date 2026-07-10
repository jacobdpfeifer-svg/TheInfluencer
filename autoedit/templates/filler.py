"""filler — fill_template: the slot-assignment engine (deterministic, no LLM).

Scores every (slot, shot) pair by how well the shot fits the slot's role,
solves the resulting assignment with the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment`) for a globally optimal — not
greedy — match, and turns the result into an `EditPlan`: one `cutter` op
(shots kept in template-slot order, synced per the template's
`music.cut_on`), one `effect` op per filled slot that names an `effect`,
one `transition` op per adjacent filled-slot boundary that names a
`transition_in`/`transition_out`, one `reframe` op per filled slot whose
`transform` isn't `"none"` (scaled to the template's own `target_aspect`,
so mixed-aspect pooled footage — see `content.extract_pool` — always lands
on one consistent output canvas), and one `text` op per template text slot, with its content resolved via
`templates.captions.generate_caption_copy` (a deterministic, footage-derived
floor under a purpose tag like `"TITLE"`/`"CTA"` — the director/LLM is free
to write better, situational copy on top of this; see
`.cursor/rules/director.mdc`).

Duration budgeting: a slot's `duration`/`max_duration` becomes a `cutter`
`trim` cap (see `subsystems/cutter.py`) whenever the assigned shot is
LONGER than that budget -- e.g. a 3s-max "hook" slot never lets an 8s shot
blow past it. A slot's `min_duration` can't be enforced the same way
(`cutter` only ever shortens footage, it can't fabricate more), so it's
instead a soft penalty in `_slot_shot_score`: a too-short shot is still
assignable (nothing here can *require* enough footage exists), just a
worse fit than a shot that actually meets the slot's floor.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditOp, EditPlan
from autoedit.models.shot import Shot
from autoedit.models.style_profile import StyleProfile
from autoedit.models.template import Template, TemplateSlot
from autoedit.templates.captions import generate_caption_copy

# A template-filled plan is more structured than the bare rule-based floor
# but still not AI-directed -- confidence sits between
# `heuristic.HEURISTIC_CONFIDENCE` (0.5, unstructured) and a real
# high-confidence LLM plan.
TEMPLATE_FILL_CONFIDENCE = 0.7

# Every (slot, shot) score below starts here and only ever adds nonnegative
# bonuses, so a real shot always strictly beats an unfilled ("dummy") slot
# in `_assign_shots_to_slots` — see that function's docstring for why that
# matters. Never referenced directly outside this module.
_BASE_SCORE = 1.0
_MOTION_WEIGHT = 2.0
_FACE_BONUS = 1.5
_CLOSE_SCALE_BONUS = 1.0
_WIDE_SCALE_PENALTY_FOR_TALKING_HEAD = 0.75
# A too-short shot is discouraged, never disqualified -- the penalty is
# clamped so it can never bring a real pairing's score down to (or below)
# `_BASE_SCORE`, preserving `_assign_shots_to_slots`'s "real always beats
# dummy" invariant (see its docstring).
_SHORT_SHOT_PENALTY = 0.5

# hook/high_energy: "prefers high motion + faces" (per the spec). `payoff`
# (the climactic closing beat of a montage) leans the same way.
_MOTION_AND_FACE_ROLES = {"hook", "high_energy", "payoff"}
# `reaction` shots are fundamentally about the face, not the motion.
_FACE_FORWARD_ROLES = {"reaction"}
# `talking_head`/`b_roll`/`any` are handled by their own branches below.

# Music-grid values a Template can name; `cutter`'s own `sync` param only
# understands "beat"/"none" (see `autoedit/subsystems/cutter.py`), so a
# `bar`-level cut still snaps to the beat grid for now.
_CUTTER_SYNC_FOR_CUT_ON = {"beat": "beat", "bar": "beat", "none": "none"}


def fill_template(template: Template, features: ContentFeatures, style: StyleProfile) -> EditPlan:
    """Assign shots to `template`'s slots via the Hungarian algorithm and emit the resulting EditPlan."""
    del style  # no style-conditioned scoring yet; kept so (template, features, style) matches match_template's call shape.

    assignment = _assign_shots_to_slots(template.slots, features.shots)

    ops = [_cutter_op(template, features, assignment)]
    ops.extend(_effect_ops(assignment))
    ops.extend(_transition_ops(assignment))
    ops.extend(_reframe_ops(template, assignment))
    ops.extend(_text_ops(template, features))

    return EditPlan(ops=ops, confidence=TEMPLATE_FILL_CONFIDENCE)


def _assign_shots_to_slots(slots: list[TemplateSlot], shots: list[Shot]) -> list[tuple[TemplateSlot, Shot]]:
    """Globally-optimal slot<->shot pairing via the Hungarian algorithm.

    `linear_sum_assignment` needs a square cost matrix, so whichever side is
    shorter gets padded with zero-cost dummy rows/columns. Every REAL
    (slot, shot) score is >= `_BASE_SCORE` (> 0), so its cost
    (`-score`) is always strictly negative — strictly better than a dummy's
    cost of exactly 0. That means the algorithm's globally-optimal,
    total-cost-minimizing assignment always prefers a real pairing over a
    dummy one when a real option exists, and only reaches for a dummy when
    there simply aren't enough slots or shots to go around:

    - slots > shots: the extra slots are the ones left paired with dummy
      shots, i.e. the LOWEST-PRIORITY slots get dropped (every filled slot
      necessarily out-competed them for the scarcer real shots).
    - shots > slots: the extra shots are the ones left paired with dummy
      slots, i.e. the LOWEST-SCORING shots get dropped, for the same reason.

    Returned pairs are in template order (ascending slot index): scipy
    guarantees `row_ind` is sorted ascending for a square cost matrix, and
    slot rows come before any padding rows.
    """
    if not slots or not shots:
        return []

    n = max(len(slots), len(shots))
    cost = np.zeros((n, n))
    for row, slot in enumerate(slots):
        for col, shot in enumerate(shots):
            cost[row, col] = -_slot_shot_score(slot, shot)

    row_ind, col_ind = linear_sum_assignment(cost)

    return [
        (slots[row], shots[col])
        for row, col in zip(row_ind, col_ind)
        if row < len(slots) and col < len(shots)
    ]


def _slot_shot_score(slot: TemplateSlot, shot: Shot) -> float:
    score = _BASE_SCORE
    has_face = shot.faces > 0

    if slot.role in _MOTION_AND_FACE_ROLES:
        score += _MOTION_WEIGHT * shot.motion
        if has_face:
            score += _FACE_BONUS
    elif slot.role == "talking_head":
        if has_face:
            score += _FACE_BONUS
        score += _CLOSE_SCALE_BONUS if shot.scale == "close" else -_WIDE_SCALE_PENALTY_FOR_TALKING_HEAD
    elif slot.role in _FACE_FORWARD_ROLES:
        if has_face:
            score += _FACE_BONUS
    # else: "b_roll"/"any" -- prefers anything, so the base score alone stands.

    if shot.dur < slot.min_duration:
        score = max(score - _SHORT_SHOT_PENALTY, _BASE_SCORE)

    return score


def _cutter_op(template: Template, features: ContentFeatures, assignment: list[tuple[TemplateSlot, Shot]]) -> EditOp:
    keep = [shot.id for _slot, shot in assignment]
    sync = _CUTTER_SYNC_FOR_CUT_ON.get(template.music.cut_on, "none")
    params: dict[str, object] = {"keep": keep, "sync": sync}
    if sync == "beat":
        params["beat_times"] = features.beat_times
    trim = {shot.id: cap for slot, shot in assignment if (cap := _slot_duration_cap(slot, shot)) is not None}
    if trim:
        params["trim"] = trim
    return EditOp(tool="cutter", params=params)


def _slot_duration_cap(slot: TemplateSlot, shot: Shot) -> float | None:
    """A `cutter` `trim` cap (seconds) for `shot` in `slot`, or `None` if it doesn't need one.

    `slot.duration`, when set, is an exact target (both a floor and a
    ceiling as far as `cutter` is concerned, since it can only shorten); an
    unset `duration` falls back to `slot.max_duration` alone, a ceiling
    only. Either way, no cap is emitted unless the shot actually exceeds it
    -- a shot that's already within budget is left untouched.
    """
    budget = slot.duration if slot.duration is not None else slot.max_duration
    return budget if shot.dur > budget else None


def _effect_ops(assignment: list[tuple[TemplateSlot, Shot]]) -> list[EditOp]:
    return [
        EditOp(tool="effect", params={"kind": slot.effect, "shot": shot.id})
        for slot, shot in assignment
        if slot.effect is not None
    ]


def _transition_ops(assignment: list[tuple[TemplateSlot, Shot]]) -> list[EditOp]:
    """One `transition` op per adjacent pair of FILLED slots that names a transition on either side.

    Walks `assignment` (the post-drop, template-ordered result), not
    `template.slots` directly, since a transition needs two real, adjacent
    shots in the final cut -- a slot the Hungarian assignment dropped can't
    anchor one.
    """
    ops = []
    for (outgoing_slot, outgoing_shot), (incoming_slot, incoming_shot) in zip(assignment, assignment[1:]):
        kind = outgoing_slot.transition_out or incoming_slot.transition_in
        if kind is None:
            continue
        ops.append(EditOp(tool="transition", params={"kind": kind, "between": [outgoing_shot.id, incoming_shot.id]}))
    return ops


def _reframe_ops(template: Template, assignment: list[tuple[TemplateSlot, Shot]]) -> list[EditOp]:
    """One `reframe` op per filled slot with a non-`"none"` `transform`.

    A slot with `transform="none"` emits nothing — the renderer already
    normalizes every shot to *some* canvas by default (see
    `subsystems/reframe.py`'s docstring), so "none" just means "don't
    override that default," not "skip normalization."
    """
    return [
        EditOp(
            tool="reframe",
            params={"kind": slot.transform, "shot": shot.id, "target_aspect": template.target_aspect},
        )
        for slot, shot in assignment
        if slot.transform != "none"
    ]


def _text_ops(template: Template, features: ContentFeatures) -> list[EditOp]:
    return [
        EditOp(
            tool="text",
            params={
                "content": generate_caption_copy(features, text_slot.placeholder),
                "style": text_slot.style,
                "anchor": text_slot.anchor,
                "start": 0.0,
            },
        )
        for text_slot in template.text_slots
    ]
