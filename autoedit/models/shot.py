"""Shot — a single per-shot measurement record.

Emitted by the pacing/framing extractors and shared by both phases: many Shots
aggregate into a StyleProfile (Phase A), and a list of Shots is embedded in
ContentFeatures (Phase B).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ShotScale = Literal["close", "wide"]

# Tolerance for cross-checking `dur` against `out - in` (floating point / frame
# rounding slack), in seconds.
_DUR_TOLERANCE = 1e-2


class Shot(BaseModel):
    """A single detected shot (contiguous span between two cuts)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, description="Unique shot identifier, e.g. 's1'.")
    source: str = Field(min_length=1, description="Path or id of the source MediaAsset.")
    in_: float = Field(alias="in", ge=0, description="Start time in seconds.")
    out_: float = Field(alias="out", gt=0, description="End time in seconds.")
    dur: float = Field(gt=0, description="Shot duration in seconds (== out - in).")
    motion: float = Field(ge=0, description="Raw motion magnitude score from frame diff.")
    brightness: float = Field(ge=0, le=1, description="Normalized mean brightness (0-1).")
    sharpness: float = Field(ge=0, description="Sharpness score (e.g. variance of Laplacian).")
    faces: int = Field(ge=0, description="Number of faces detected in the shot.")
    scale: ShotScale = Field(description="Shot scale: 'close' or 'wide'.")

    @model_validator(mode="after")
    def _check_span(self) -> "Shot":
        if self.out_ <= self.in_:
            raise ValueError(f"out ({self.out_}) must be greater than in ({self.in_})")
        expected_dur = self.out_ - self.in_
        if abs(self.dur - expected_dur) > _DUR_TOLERANCE:
            raise ValueError(
                f"dur ({self.dur}) does not match out - in ({expected_dur}) "
                f"within tolerance {_DUR_TOLERANCE}"
            )
        return self
