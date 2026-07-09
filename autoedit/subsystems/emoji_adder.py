"""emoji_adder — append an emoji overlay instruction to the Timeline's emoji track.

Per AGENTS.md's `EditPlan` example, `params` is `{"glyph", "at"}`. Only ever
appends a `TimelineItem` to an emoji track; never renders or touches media.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from autoedit.models.timeline import Timeline, TimelineItem
from autoedit.subsystems._timeline_ops import append_item, count_items_of_kind

_TRACK_KIND = "emoji"
_DEFAULT_TRACK_NAME = "emoji"
_DEFAULT_DURATION_SEC = 1.0


class EmojiParams(BaseModel):
    """Params for the `emoji` tool."""

    model_config = ConfigDict(extra="forbid")

    glyph: str = Field(min_length=1, description="Emoji glyph to overlay, e.g. '🔥'.")
    at: float = Field(ge=0, description="Time in seconds, on the output timeline, to show the emoji.")
    duration: float = Field(default=_DEFAULT_DURATION_SEC, gt=0, description="How long the emoji stays on screen.")


def add_emoji(timeline: Timeline, params: dict) -> Timeline:
    """Return a new Timeline with an emoji item appended to the emoji track."""
    emoji_params = EmojiParams.model_validate(params)
    item = TimelineItem(
        id=f"emoji-{count_items_of_kind(timeline, _TRACK_KIND) + 1}",
        start=emoji_params.at,
        end=emoji_params.at + emoji_params.duration,
        payload={"glyph": emoji_params.glyph},
    )
    return append_item(timeline, kind=_TRACK_KIND, track_name=_DEFAULT_TRACK_NAME, item=item)
