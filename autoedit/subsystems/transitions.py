"""transitions — append a shot-to-shot transition instruction to the Timeline's transition track.

A transition, unlike an `effect`, is scoped to a PAIR of shots (the outgoing
and incoming shot around one cut point) rather than a single shot. `between`
holds those two shot ids, in `[outgoing, incoming]` order, matching
`payload["shot"]` on the video track's items exactly like `effects.py`'s
`shot` lookup does. The transition's span is centered on the cut point
between them (the outgoing shot's current end, which should equal the
incoming shot's current start on a contiguous, already-cut Timeline):
`duration/2` seconds on each side.

Only ever appends a `TimelineItem` to a transition track; never renders or
touches media (per `.cursor/rules/subsystems.mdc`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autoedit.models.timeline import Timeline, TimelineItem
from autoedit.subsystems._timeline_ops import append_item, count_items_of_kind, find_video_item_by_shot

_TRACK_KIND = "transition"
_DEFAULT_TRACK_NAME = "transitions"
_DEFAULT_DURATION_SEC = 0.3


class TransitionParams(BaseModel):
    """Params for the `transition` tool."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["whip_pan", "fade"] = Field(description="Transition kind.")
    between: list[str] = Field(description="Exactly two shot ids: [outgoing, incoming].")
    duration: float = Field(default=_DEFAULT_DURATION_SEC, gt=0, description="Total transition length, seconds.")

    @field_validator("between")
    @classmethod
    def _check_exactly_two_shots(cls, value: list[str]) -> list[str]:
        if len(value) != 2:
            raise ValueError(f"transition params: 'between' must name exactly 2 shots, got {len(value)}")
        return value


def apply_transition(timeline: Timeline, params: dict) -> Timeline:
    """Return a new Timeline with a transition item appended to the transition track."""
    transition_params = TransitionParams.model_validate(params)
    outgoing_id, incoming_id = transition_params.between

    outgoing_item = find_video_item_by_shot(timeline, outgoing_id)
    if outgoing_item is None:
        raise ValueError(f"transition: outgoing shot id {outgoing_id!r} not found on the video track")
    incoming_item = find_video_item_by_shot(timeline, incoming_id)
    if incoming_item is None:
        raise ValueError(f"transition: incoming shot id {incoming_id!r} not found on the video track")

    half = transition_params.duration / 2
    pivot = outgoing_item.end
    start = max(0.0, pivot - half)
    end = pivot + half

    item = TimelineItem(
        id=f"trans-{count_items_of_kind(timeline, _TRACK_KIND) + 1}",
        start=start,
        end=end,
        payload={"kind": transition_params.kind, "between": [outgoing_id, incoming_id]},
    )
    return append_item(timeline, kind=_TRACK_KIND, track_name=_DEFAULT_TRACK_NAME, item=item)
