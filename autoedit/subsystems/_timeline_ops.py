"""Shared helpers for mutating a `Timeline` immutably.

Every subsystem returns a NEW `Timeline` rather than mutating its input in
place (Pydantic models are treated as values throughout the pipeline, per
AGENTS.md's "pure function" rule). These helpers centralize the
find-or-create-track-then-append/replace pattern so each subsystem module can
stay focused on its own params and payload shape.
"""

from __future__ import annotations

from typing import Callable

from autoedit.models.timeline import Timeline, TimelineItem, Track, TrackKind


def append_item(timeline: Timeline, *, kind: TrackKind, track_name: str, item: TimelineItem) -> Timeline:
    """Return a copy of `timeline` with `item` appended to the `(kind, track_name)` track.

    Creates the track if no track with that kind+name exists yet. Every other
    track is passed through untouched.
    """
    return replace_track_items(
        timeline,
        kind=kind,
        track_name=track_name,
        transform=lambda items: [*items, item],
    )


def replace_track_items(
    timeline: Timeline,
    *,
    kind: TrackKind,
    track_name: str,
    transform: Callable[[list[TimelineItem]], list[TimelineItem]],
) -> Timeline:
    """Return a copy of `timeline` with the named track's items replaced by `transform(items)`.

    Creates the track (passing an empty list to `transform`) if it does not
    exist yet. Every other track is passed through untouched.
    """
    found = False
    new_tracks: list[Track] = []
    for track in timeline.tracks:
        if track.kind == kind and track.name == track_name:
            new_tracks.append(Track(name=track.name, kind=track.kind, items=transform(track.items)))
            found = True
        else:
            new_tracks.append(track)
    if not found:
        new_tracks.append(Track(name=track_name, kind=kind, items=transform([])))
    return Timeline(tracks=new_tracks)


def count_items_of_kind(timeline: Timeline, kind: TrackKind) -> int:
    """Total number of items already on any track of the given `kind` (used for deterministic ids)."""
    return sum(len(track.items) for track in timeline.tracks if track.kind == kind)


def find_video_item_by_shot(timeline: Timeline, shot_id: str) -> TimelineItem | None:
    """Find the video-track item whose `payload["shot"]` matches `shot_id`, if any."""
    for track in timeline.tracks:
        if track.kind == "video":
            for item in track.items:
                if item.payload.get("shot") == shot_id:
                    return item
    return None
