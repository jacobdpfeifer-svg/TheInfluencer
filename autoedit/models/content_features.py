"""ContentFeatures — the Phase B output.

Machine-readable description of a single piece of raw footage, produced by
running the same four extractors used in Phase A. Consumed by the director
alongside a StyleProfile to produce an EditPlan.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from autoedit.models.shot import Shot

MotionBucket = Literal["low", "med", "high"]
TextStyleBucket = Literal["karaoke", "static", "none"]


class ContentFeatures(BaseModel):
    """Quantized, LLM-brief-ready features for a single raw footage asset."""

    model_config = ConfigDict(extra="forbid")

    aspect: float = Field(gt=0, description="Aspect ratio (width / height) of the source footage.")
    has_speech: bool = Field(description="Whether speech was detected in the audio track.")
    music_bpm: Optional[float] = Field(
        default=None, gt=0, description="Detected music BPM, or null if no music/beat detected."
    )
    shots: list[Shot] = Field(min_length=1, description="Per-shot measurements for this footage.")
    motion: MotionBucket = Field(description="Quantized overall motion bucket.")
    is_vertical: bool = Field(description="Whether the footage is vertical (portrait) aspect.")
    has_face: bool = Field(description="Whether any face was detected across the footage.")
    beat_times: list[float] = Field(
        default_factory=list, description="Detected audio beat timestamps (seconds), from the audio extractor."
    )
    has_text: bool = Field(default=False, description="Whether any on-screen text was detected in the footage.")
    text_style: TextStyleBucket = Field(
        default="none", description="Dominant on-screen text style class across shots with text, or 'none'."
    )
