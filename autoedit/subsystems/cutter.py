"""cutter — keep/reorder/retime a subset of shots on the Timeline's video track.

Per AGENTS.md's `EditPlan` example, `params` is `{"keep": [...], "sync": ...}`
where `keep` is a list of *shot ids* (matching `payload["shot"]` on existing
video-track items — see `fixtures/timeline.json`). `cutter` never has its own
notion of source media or shot duration: it reads the candidate shots already
present on the input Timeline's video track (seeded upstream from
`ContentFeatures.shots`), drops everything not in `keep`, reorders the rest to
match `keep`, and packs them back-to-back starting at t=0, preserving each
item's original duration and any extra payload fields it already carried
(e.g. `source`/`in`/`out`, if the caller put them there).

An optional `trim: {shot_id: max_seconds}` caps specific shots down to
`max_seconds` (taken from the shot's existing `in` point, i.e. the START of
the shot survives, not the end) before packing — this is how
`templates.filler.fill_template` enforces a `TemplateSlot.duration`/
`max_duration` budget without needing its own re-timing logic (`cutter`
already owns all timeline retiming). A shot NOT in `trim`, or shorter than
its cap already, is left at its full original duration — `cutter` can only
ever shorten a shot, never fabricate footage to lengthen one.

When `sync='beat'`, `beat_times` (explicit params) takes priority; if empty,
`cutter` falls back to `timeline.beat_times` (see `models/timeline.py` and
`executor.build_initial_timeline`, which seeds it from `ContentFeatures`) so
a beat grid set once on the Timeline itself still works without every caller
having to re-thread it through `EditOp.params` by hand.

It never opens, decodes, or writes a video file — only Timeline instructions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from autoedit.models.timeline import Timeline, TimelineItem
from autoedit.subsystems._timeline_ops import replace_track_items

_VIDEO_TRACK_KIND = "video"
_DEFAULT_VIDEO_TRACK_NAME = "v1"
# How close (seconds) a natural cut point must be to a beat to snap onto it.
_BEAT_SNAP_TOLERANCE_SEC = 0.5


class CutterParams(BaseModel):
    """Params for the `cutter` tool."""

    model_config = ConfigDict(extra="forbid")

    keep: list[str] = Field(min_length=1, description="Shot ids to keep, in the desired output order.")
    sync: Literal["beat", "none"] = Field(default="none", description="Whether to snap cut points to beats.")
    beat_times: list[float] = Field(
        default_factory=list, description="Beat timestamps (seconds) to snap cuts to when sync='beat'."
    )
    trim: dict[str, float] = Field(
        default_factory=dict,
        description="Optional shot_id -> max duration (seconds) cap; longer shots are trimmed from their start.",
    )

    @field_validator("trim")
    @classmethod
    def _check_trim_values_are_positive(cls, trim: dict[str, float]) -> dict[str, float]:
        if any(cap <= 0 for cap in trim.values()):
            raise ValueError("cutter: every 'trim' cap must be > 0")
        return trim


def cut(timeline: Timeline, params: dict) -> Timeline:
    """Return a new Timeline whose video track is `params["keep"]`, reordered and retimed."""
    cutter_params = CutterParams.model_validate(params)

    shots_by_id = _index_video_items_by_shot(timeline)
    missing = [shot_id for shot_id in cutter_params.keep if shot_id not in shots_by_id]
    if missing:
        raise ValueError(f"cutter: shot id(s) not found on the video track: {missing}")

    kept_items = [shots_by_id[shot_id] for shot_id in cutter_params.keep]
    natural_durations = [item.end - item.start for item in kept_items]
    was_trimmed = [
        shot_id in cutter_params.trim and natural > cutter_params.trim[shot_id]
        for shot_id, natural in zip(cutter_params.keep, natural_durations)
    ]
    durations = [
        min(natural, cutter_params.trim[shot_id]) if shot_id in cutter_params.trim else natural
        for shot_id, natural in zip(cutter_params.keep, natural_durations)
    ]
    boundaries = _contiguous_boundaries(durations)

    if cutter_params.sync == "beat":
        beat_times = cutter_params.beat_times or timeline.beat_times
        if beat_times:
            boundaries = _snap_to_beats(boundaries, beat_times)

    new_items = [
        TimelineItem(
            id=f"clip-{shot_id}", start=start, end=end,
            payload=_trimmed_payload(item, end - start) if trimmed else dict(item.payload),
        )
        for shot_id, item, (start, end), trimmed in zip(cutter_params.keep, kept_items, boundaries, was_trimmed)
    ]

    track_name = _existing_video_track_name(timeline) or _DEFAULT_VIDEO_TRACK_NAME
    return replace_track_items(
        timeline, kind=_VIDEO_TRACK_KIND, track_name=track_name, transform=lambda _existing: new_items
    )


def _trimmed_payload(item: TimelineItem, kept_duration: float) -> dict:
    """Copy `item.payload`, shrinking its `out` source point to match an applied trim."""
    payload = dict(item.payload)
    payload["out"] = payload.get("in", 0.0) + kept_duration
    return payload


def _index_video_items_by_shot(timeline: Timeline) -> dict[str, TimelineItem]:
    return {
        item.payload["shot"]: item
        for track in timeline.tracks
        if track.kind == _VIDEO_TRACK_KIND
        for item in track.items
        if "shot" in item.payload
    }


def _existing_video_track_name(timeline: Timeline) -> str | None:
    for track in timeline.tracks:
        if track.kind == _VIDEO_TRACK_KIND:
            return track.name
    return None


def _contiguous_boundaries(durations: list[float]) -> list[tuple[float, float]]:
    boundaries = []
    t = 0.0
    for duration in durations:
        boundaries.append((t, t + duration))
        t += duration
    return boundaries


def _snap_to_beats(
    boundaries: list[tuple[float, float]], beat_times: list[float]
) -> list[tuple[float, float]]:
    """Snap each *internal* cut point to its nearest beat within tolerance, staying contiguous.

    The very first start (0.0) and very last end are left alone; only the cut
    points *between* kept shots move, so total coverage never gains a gap.
    """
    if len(boundaries) < 2:
        return boundaries
    cut_points = [boundaries[0][0]]
    for start, end in boundaries[:-1]:
        nearest_beat = min(beat_times, key=lambda beat: abs(beat - end))
        if abs(nearest_beat - end) <= _BEAT_SNAP_TOLERANCE_SEC:
            cut_points.append(nearest_beat)
        else:
            cut_points.append(end)
    cut_points.append(boundaries[-1][1])
    return list(zip(cut_points[:-1], cut_points[1:]))
