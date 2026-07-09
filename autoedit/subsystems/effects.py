"""effects — append a visual-effect instruction to the Timeline's effect track.

Per AGENTS.md's `EditPlan` example, `params` is `{"kind", "shot"}`: the effect
is scoped to a shot already present on the video track (by `payload["shot"]`),
and this subsystem looks up that shot's *current* (post-cut) span so the
effect lands in the right place even after `cutter` has retimed things.
Callers may pass explicit `start`/`end` instead of `shot` to scope the effect
directly. `kind` is an open vocabulary (e.g. "zoom_in") — validating it
against the tool menu is the director/manifest's job, not this subsystem's.

Only ever appends a `TimelineItem` to an effect track; never renders or
touches media.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoedit.models.timeline import Timeline, TimelineItem
from autoedit.subsystems._timeline_ops import append_item, count_items_of_kind, find_video_item_by_shot

_TRACK_KIND = "effect"
_DEFAULT_TRACK_NAME = "effects"


class EffectParams(BaseModel):
    """Params for the `effect` tool."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1, description="Effect kind, e.g. 'zoom_in' (open vocabulary).")
    shot: str | None = Field(default=None, description="Shot id to scope the effect to.")
    start: float | None = Field(default=None, ge=0, description="Explicit start time, overriding `shot` lookup.")
    end: float | None = Field(default=None, gt=0, description="Explicit end time, overriding `shot` lookup.")

    @model_validator(mode="after")
    def _check_has_a_span_source(self) -> "EffectParams":
        has_explicit_span = self.start is not None and self.end is not None
        if not has_explicit_span and self.shot is None:
            raise ValueError("effect params must include either 'shot' or both 'start' and 'end'")
        return self


def apply_effect(timeline: Timeline, params: dict) -> Timeline:
    """Return a new Timeline with an effect item appended to the effect track."""
    effect_params = EffectParams.model_validate(params)

    if effect_params.start is not None and effect_params.end is not None:
        start, end = effect_params.start, effect_params.end
    else:
        video_item = find_video_item_by_shot(timeline, effect_params.shot)
        if video_item is None:
            raise ValueError(f"effect: shot id {effect_params.shot!r} not found on the video track")
        start, end = video_item.start, video_item.end

    payload: dict[str, object] = {"kind": effect_params.kind}
    if effect_params.shot is not None:
        payload["shot"] = effect_params.shot

    item = TimelineItem(id=f"fx-{count_items_of_kind(timeline, _TRACK_KIND) + 1}", start=start, end=end, payload=payload)
    return append_item(timeline, kind=_TRACK_KIND, track_name=_DEFAULT_TRACK_NAME, item=item)
