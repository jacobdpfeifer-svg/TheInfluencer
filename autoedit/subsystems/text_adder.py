"""text_adder — append a caption/title instruction to the Timeline's text track.

Per AGENTS.md's `EditPlan` example, `params` is
`{"content", "style", "anchor", "start"}`. Only ever appends a `TimelineItem`
to a text track; never renders or touches media.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from autoedit.models.timeline import Timeline, TimelineItem
from autoedit.subsystems._timeline_ops import append_item, count_items_of_kind

_TRACK_KIND = "text"
_DEFAULT_TRACK_NAME = "captions"
_DEFAULT_DURATION_SEC = 2.0


class TextParams(BaseModel):
    """Params for the `text` tool."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="On-screen text content.")
    style: Literal["karaoke", "static"] = Field(description="Caption style class.")
    anchor: Literal["top", "middle", "bottom"] = Field(default="bottom", description="Vertical position bucket.")
    start: float = Field(ge=0, description="Start time in seconds, on the output timeline.")
    duration: float = Field(default=_DEFAULT_DURATION_SEC, gt=0, description="How long the caption stays on screen.")


def add_text(timeline: Timeline, params: dict) -> Timeline:
    """Return a new Timeline with a caption item appended to the text track."""
    text_params = TextParams.model_validate(params)
    item = TimelineItem(
        id=f"cap-{count_items_of_kind(timeline, _TRACK_KIND) + 1}",
        start=text_params.start,
        end=text_params.start + text_params.duration,
        payload={"content": text_params.content, "style": text_params.style, "anchor": text_params.anchor},
    )
    return append_item(timeline, kind=_TRACK_KIND, track_name=_DEFAULT_TRACK_NAME, item=item)
