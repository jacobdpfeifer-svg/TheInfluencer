"""Timeline — the internal, OTIO-backed editing model.

The Timeline holds INSTRUCTIONS, never media. Subsystems (`cutter`,
`text_adder`, `emoji_adder`, `effects`) each take a Timeline and params and
return a new/mutated Timeline. Only the renderer ever turns a Timeline into
pixels, in a single composite pass.

This module defines the plain-JSON Pydantic contract for a Timeline. It
mirrors the structure OTIO would use (tracks containing timed items) without
requiring the `opentimelineio` library itself, so that every stage before the
renderer can be tested on fixture JSON alone (see AGENTS.md testing rule). The
renderer/executor layer is responsible for translating to/from a real
`opentimelineio.schema.Timeline` when it actually composites.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TrackKind = Literal["video", "audio", "text", "emoji", "effect"]


class TimelineItem(BaseModel):
    """A single timed instruction placed on a track."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Unique item identifier within the timeline.")
    start: float = Field(ge=0, description="Start time in seconds, on the output timeline.")
    end: float = Field(gt=0, description="End time in seconds, on the output timeline.")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Instruction-specific data (e.g. shot ref, text, glyph, effect kind)."
    )

    @model_validator(mode="after")
    def _check_span(self) -> "TimelineItem":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self


class Track(BaseModel):
    """A single track (one of video/audio/text/emoji/effect) of timed items."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Track name, e.g. 'v1', 'captions'.")
    kind: TrackKind = Field(description="Track kind, matching a subsystem's domain.")
    items: list[TimelineItem] = Field(default_factory=list, description="Timed instructions on this track.")


class Timeline(BaseModel):
    """The full set of tracks that make up an edit-in-progress. Not media."""

    model_config = ConfigDict(extra="forbid")

    tracks: list[Track] = Field(default_factory=list, description="All tracks in this timeline.")
