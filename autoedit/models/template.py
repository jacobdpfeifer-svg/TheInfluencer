"""Template — a named timeline skeleton with typed slots that footage gets poured into.

This is the "make my footage look like THIS creator's style" primitive:
`autoedit.templates.matcher.match_template` picks the best-fitting `Template`
for a given `ContentFeatures`/`StyleProfile` pair, and
`autoedit.templates.filler.fill_template` assigns real shots to its
`TemplateSlot`s to produce an `EditPlan`. Both of those are deterministic —
no LLM — so a `Template` is pure, JSON-serializable data, exactly like every
other inter-stage contract in this codebase.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SlotRole = Literal["hook", "high_energy", "b_roll", "talking_head", "reaction", "payoff", "any"]


class TemplateSlot(BaseModel):
    """One position in the template's timeline skeleton, waiting for a shot."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="Unique slot id within the template, e.g. 'hook', 'b_roll_1'.")
    role: SlotRole = Field(description="What kind of shot this slot wants (drives filler.py's slot<->shot scoring).")
    duration: float | None = Field(
        default=None, gt=0, description="Fixed duration in seconds, or None to derive it from the beat grid."
    )
    min_duration: float = Field(default=0.3, gt=0, description="Shortest this slot's assigned shot may be trimmed to.")
    max_duration: float = Field(default=10.0, gt=0, description="Longest this slot's assigned shot may be trimmed to.")
    transform: Literal["center_crop", "rule_of_thirds", "none"] = Field(
        default="none", description="Framing transform to apply to this slot's shot."
    )
    effect: str | None = Field(default=None, description="Effect kind to apply, e.g. 'zoom_in' (None = no effect).")
    transition_in: str | None = Field(default=None, description="Transition kind coming INTO this slot from the previous one.")
    transition_out: str | None = Field(default=None, description="Transition kind going OUT of this slot into the next one.")

    @model_validator(mode="after")
    def _check_duration_bounds(self) -> "TemplateSlot":
        if self.min_duration > self.max_duration:
            raise ValueError(f"min_duration ({self.min_duration}) must be <= max_duration ({self.max_duration})")
        return self


class TextSlot(BaseModel):
    """A caption/title position overlaid on the edit, with placeholder copy."""

    model_config = ConfigDict(extra="forbid")

    anchor: Literal["top", "middle", "bottom"] = Field(description="Vertical position bucket.")
    placeholder: str = Field(min_length=1, description="Placeholder content, e.g. 'TITLE'/'CTA' — an LLM replaces this later.")
    style: Literal["karaoke", "static"] = Field(description="Caption style class.")


class TemplateMusic(BaseModel):
    """This template's expectations of the footage's music track."""

    model_config = ConfigDict(extra="forbid")

    required: bool = Field(description="Whether this template expects a music track to be present.")
    cut_on: Literal["beat", "bar", "none"] = Field(description="Grid cuts should snap to, if any.")


class Template(BaseModel):
    """A named, reusable timeline skeleton: ordered video slots + overlay text slots."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Unique template name, e.g. 'punchy_beat_montage'.")
    aspect_ratio: str = Field(default="9:16", description="Target aspect ratio, e.g. '9:16'.")
    fps: int = Field(default=30, gt=0, description="Target output frame rate.")
    music: TemplateMusic = Field(description="This template's music requirements.")
    slots: list[TemplateSlot] = Field(min_length=1, description="Ordered video slots to fill with footage.")
    text_slots: list[TextSlot] = Field(default_factory=list, description="Caption/title overlays for the edit.")

    @property
    def target_aspect(self) -> float:
        """`aspect_ratio` ("W:H") as a float, e.g. "9:16" -> 0.5625.

        The single shared parse of `aspect_ratio` — `templates.matcher`
        (footage-fit scoring) and `templates.filler` (emitting `reframe` ops
        so the renderer normalizes every shot onto this canvas) both need
        the same float, so it lives here once rather than being
        re-implemented per caller.
        """
        width_str, _, height_str = self.aspect_ratio.partition(":")
        return float(width_str) / float(height_str)
