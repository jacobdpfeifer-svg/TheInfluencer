"""MediaAsset — the output of `ingest.probe`.

A pure description of a media file on disk, as reported by ffprobe. Carries no
opinions about editing; it is the raw input to the extractors.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MediaType = Literal["video", "audio", "image"]


class MediaAsset(BaseModel):
    """Probed metadata for a single media file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Filesystem path to the media file.")
    type: MediaType = Field(description="Coarse media kind, as determined by ffprobe.")
    duration: float = Field(gt=0, description="Duration in seconds.")
    width: int = Field(gt=0, description="Pixel width.")
    height: int = Field(gt=0, description="Pixel height.")
    fps: float = Field(gt=0, description="Frames per second.")
    codec: str = Field(min_length=1, description="Video (or audio) codec name.")
    audio_channels: int = Field(ge=0, description="Number of audio channels (0 if none).")
