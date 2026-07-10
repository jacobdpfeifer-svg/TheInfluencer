"""reframe — append a per-shot canvas-normalization instruction to the Timeline's reframe track.

Per AGENTS.md's `ContentFeatures`/`Template` schemas, footage can arrive at
any aspect ratio, and a `Template` names ONE target `aspect_ratio` for the
whole output — but nothing upstream of the renderer ever resized a frame.
`reframe` is the missing instruction: like `effects.py`, it's scoped to a
shot already on the video track (by `payload["shot"]`) or an explicit
`start`/`end` span, and it names how that shot's frame should be
resized+cropped (or letterboxed) onto the output's `target_aspect` canvas.

`mode` mirrors `TemplateSlot.transform`'s vocabulary plus `"fit"`:
  - "center_crop": scale to COVER the canvas, crop centered.
  - "rule_of_thirds": scale to COVER the canvas, crop biased toward the
    upper third (a coarse, deterministic proxy for "keep faces/heads in
    frame" without per-frame subject tracking).
  - "fit": scale to fit ENTIRELY inside the canvas, letterboxed (no crop).

The renderer normalizes EVERY video segment to one canvas regardless of
whether an explicit `reframe` op exists for its shot (a valid output can
only have one frame size) — an explicit op just lets the director/template
override the mode or target aspect for a specific shot; the default for any
shot with no matching op is `"center_crop"` at the Timeline-wide default
aspect (see `renderer._DEFAULT_TARGET_ASPECT`).

Only ever appends a `TimelineItem` to a reframe track; never renders or
touches media (per `.cursor/rules/subsystems.mdc`).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoedit.models.timeline import Timeline, TimelineItem
from autoedit.subsystems._timeline_ops import append_item, count_items_of_kind, find_video_item_by_shot

_TRACK_KIND = "reframe"
_DEFAULT_TRACK_NAME = "reframe"

ReframeMode = Literal["center_crop", "rule_of_thirds", "fit"]

# 9:16 -- the same default AGENTS.md's builtin `Template`s use.
_DEFAULT_TARGET_ASPECT = 9 / 16


class ReframeParams(BaseModel):
    """Params for the `reframe` tool."""

    model_config = ConfigDict(extra="forbid")

    kind: ReframeMode = Field(default="center_crop", description="How to fit the shot's frame onto the target canvas.")
    target_aspect: float = Field(default=_DEFAULT_TARGET_ASPECT, gt=0, description="Output canvas aspect (width / height).")
    shot: str | None = Field(default=None, description="Shot id to scope the reframe to.")
    start: float | None = Field(default=None, ge=0, description="Explicit start time, overriding `shot` lookup.")
    end: float | None = Field(default=None, gt=0, description="Explicit end time, overriding `shot` lookup.")

    @model_validator(mode="after")
    def _check_has_a_span_source(self) -> "ReframeParams":
        has_explicit_span = self.start is not None and self.end is not None
        if not has_explicit_span and self.shot is None:
            raise ValueError("reframe params must include either 'shot' or both 'start' and 'end'")
        return self


def apply_reframe(timeline: Timeline, params: dict) -> Timeline:
    """Return a new Timeline with a reframe item appended to the reframe track."""
    reframe_params = ReframeParams.model_validate(params)

    if reframe_params.start is not None and reframe_params.end is not None:
        start, end = reframe_params.start, reframe_params.end
    else:
        video_item = find_video_item_by_shot(timeline, reframe_params.shot)
        if video_item is None:
            raise ValueError(f"reframe: shot id {reframe_params.shot!r} not found on the video track")
        start, end = video_item.start, video_item.end

    payload: dict[str, object] = {"kind": reframe_params.kind, "target_aspect": reframe_params.target_aspect}
    if reframe_params.shot is not None:
        payload["shot"] = reframe_params.shot

    item = TimelineItem(id=f"reframe-{count_items_of_kind(timeline, _TRACK_KIND) + 1}", start=start, end=end, payload=payload)
    return append_item(timeline, kind=_TRACK_KIND, track_name=_DEFAULT_TRACK_NAME, item=item)
