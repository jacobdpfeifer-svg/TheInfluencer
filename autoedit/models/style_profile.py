"""StyleProfile — the learned preference context (Phase A output).

A StyleProfile is a DISTRIBUTION aggregated across many reference videos, never
a single-video snapshot. `sample_count` records how many videos contributed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Slack allowed when checking that caption style frequencies sum to 1.0.
_FREQ_SUM_TOLERANCE = 1e-3


class CaptionStyleFreq(BaseModel):
    """Relative frequency of each caption style, each in [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    karaoke: float = Field(ge=0, le=1, description="Fraction of captions that are karaoke-style.")
    static: float = Field(ge=0, le=1, description="Fraction of captions that are static titles.")

    @model_validator(mode="after")
    def _check_sums_to_one(self) -> "CaptionStyleFreq":
        total = self.karaoke + self.static
        if abs(total - 1.0) > _FREQ_SUM_TOLERANCE:
            raise ValueError(f"caption_style_freq must sum to 1.0, got {total}")
        return self


class StyleProfile(BaseModel):
    """Aggregated editing-style preferences learned across reference videos."""

    model_config = ConfigDict(extra="forbid")

    aspect: float = Field(gt=0, description="Dominant aspect ratio (width / height).")
    shot_len_median: float = Field(gt=0, description="Median shot length in seconds.")
    shot_len_spread: float = Field(ge=0, description="Spread (e.g. IQR/stddev) of shot length, seconds.")
    cut_on_beat: bool = Field(description="Whether cuts tend to land on audio beats.")
    caption_style_freq: CaptionStyleFreq = Field(description="Karaoke vs static caption frequency.")
    caption_density: float = Field(ge=0, description="Captions per second, averaged across samples.")
    text_amount: float = Field(ge=0, description="Relative amount of on-screen text (0+ scale).")
    effect_freq: float = Field(ge=0, description="Effects applied per second, averaged across samples.")
    sample_count: int = Field(ge=1, description="Number of reference videos aggregated into this profile.")
