"""music — append a music-bed instruction to the Timeline's audio track.

Per AGENTS.md's build-order note (a music bed is "the missing subsystem" for
Workstream B): raw footage's own embedded audio is often silence, ambient
noise, or dialogue -- not the music the director/template beat-synced cuts
against. `music` names an EXTERNAL audio file to mix in under the video at a
given volume; the CLI is the one place that wires a real music FILE's
detected `beat_times`/`bpm` (via `extractors.audio.extract_audio`) in as the
*authoritative* beat grid before the director/template ever runs (see
`cli.make`'s `--music` flag), so `cutter`'s beat-sync snaps to the actual
music, not to whatever the raw footage happened to contain.

Only ever appends a `TimelineItem` to an audio track; never renders or
touches media (per `.cursor/rules/subsystems.mdc`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from autoedit.models.timeline import Timeline, TimelineItem
from autoedit.subsystems._timeline_ops import append_item, count_items_of_kind

_TRACK_KIND = "audio"
_DEFAULT_TRACK_NAME = "music"

# Mixed under the video's own (dialogue/ambient) audio by default, per the
# usual "music bed sits behind the talent" convention for short-form edits.
_DEFAULT_MUSIC_VOLUME = 0.6


class MusicParams(BaseModel):
    """Params for the `music` tool."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, description="Path to the music file to mix in.")
    offset: float = Field(default=0.0, ge=0, description="Where in the music FILE to start reading from.")
    volume: float = Field(default=_DEFAULT_MUSIC_VOLUME, ge=0, le=2, description="Gain multiplier applied to the music bed.")
    duration: float | None = Field(
        default=None, gt=0, description="How long the music bed plays; defaults to the whole video track's span."
    )


def add_music(timeline: Timeline, params: dict) -> Timeline:
    """Return a new Timeline with a music-bed item appended to the audio track."""
    music_params = MusicParams.model_validate(params)

    span = music_params.duration if music_params.duration is not None else _video_track_duration(timeline)
    if span <= 0:
        raise ValueError("music: cannot size a music bed with no explicit 'duration' and an empty video track")

    payload = {
        "source": music_params.source,
        "in": music_params.offset,
        "out": music_params.offset + span,
        "volume": music_params.volume,
    }
    item = TimelineItem(id=f"music-{count_items_of_kind(timeline, _TRACK_KIND) + 1}", start=0.0, end=span, payload=payload)
    return append_item(timeline, kind=_TRACK_KIND, track_name=_DEFAULT_TRACK_NAME, item=item)


def _video_track_duration(timeline: Timeline) -> float:
    ends = [item.end for track in timeline.tracks if track.kind == "video" for item in track.items]
    return max(ends, default=0.0)
