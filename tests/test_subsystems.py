"""Tests for autoedit/subsystems/*.

Per AGENTS.md's testing rule, none of this touches real media or renders —
every subsystem is exercised purely as `(Timeline, params) -> Timeline` on
in-memory fixtures.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from autoedit.models.timeline import Timeline, TimelineItem, Track
from autoedit.models.plan import EditPlan
from autoedit.subsystems import TOOL_MANIFEST
from autoedit.subsystems.cutter import cut
from autoedit.subsystems.effects import apply_effect
from autoedit.subsystems.emoji_adder import add_emoji
from autoedit.subsystems.text_adder import add_text


def _video_timeline(*shots: tuple[str, float, float]) -> Timeline:
    """Build a Timeline with one video track of `(shot_id, start, end)` items."""
    items = [
        TimelineItem(id=f"clip-{shot_id}", start=start, end=end, payload={"shot": shot_id, "source": "raw.mp4"})
        for shot_id, start, end in shots
    ]
    return Timeline(tracks=[Track(name="v1", kind="video", items=items)])


# --- cutter --------------------------------------------------------------


class TestCutter:
    def test_keeps_and_reorders_and_retimes_contiguously(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 3.5), ("s3", 3.5, 4.0))

        result = cut(timeline, {"keep": ["s3", "s1"]})

        video_track = result.tracks[0]
        assert [item.id for item in video_track.items] == ["clip-s3", "clip-s1"]
        # s3 was 0.5s long, s1 was 2.0s long; packed back-to-back from t=0.
        assert (video_track.items[0].start, video_track.items[0].end) == (0.0, 0.5)
        assert (video_track.items[1].start, video_track.items[1].end) == (0.5, 2.5)

    def test_preserves_extra_payload_fields(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = cut(timeline, {"keep": ["s1"]})
        assert result.tracks[0].items[0].payload == {"shot": "s1", "source": "raw.mp4"}

    def test_passes_through_other_tracks_untouched(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        timeline = Timeline(
            tracks=[
                *timeline.tracks,
                Track(name="captions", kind="text", items=[TimelineItem(id="cap-1", start=0.0, end=1.0, payload={})]),
            ]
        )

        result = cut(timeline, {"keep": ["s2"]})

        text_tracks = [t for t in result.tracks if t.kind == "text"]
        assert len(text_tracks) == 1
        assert text_tracks[0].items[0].id == "cap-1"

    def test_missing_shot_id_raises(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        with pytest.raises(ValueError, match="nope"):
            cut(timeline, {"keep": ["nope"]})

    def test_beat_sync_snaps_cut_point_within_tolerance(self) -> None:
        # Natural cut point (after s1) lands at t=2.0; nearest beat 2.1 is within tolerance.
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))

        result = cut(timeline, {"keep": ["s1", "s2"], "sync": "beat", "beat_times": [0.5, 2.1, 4.0]})

        items = result.tracks[0].items
        assert items[0].end == pytest.approx(2.1)
        assert items[1].start == pytest.approx(2.1)
        assert items[1].end == pytest.approx(4.0)  # last end untouched

    def test_beat_sync_leaves_cut_point_when_no_beat_is_close(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))

        result = cut(timeline, {"keep": ["s1", "s2"], "sync": "beat", "beat_times": [10.0]})

        items = result.tracks[0].items
        assert items[0].end == pytest.approx(2.0)

    def test_no_beat_snap_when_sync_is_none_even_with_beat_times(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))

        result = cut(timeline, {"keep": ["s1", "s2"], "beat_times": [2.1]})

        assert result.tracks[0].items[0].end == pytest.approx(2.0)

    def test_rejects_unknown_params(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        with pytest.raises(ValidationError):
            cut(timeline, {"keep": ["s1"], "bogus": True})


# --- text_adder ------------------------------------------------------------


class TestTextAdder:
    def test_appends_item_to_new_captions_track(self) -> None:
        timeline = Timeline(tracks=[])

        result = add_text(timeline, {"content": "3 months in", "style": "karaoke", "anchor": "top", "start": 0.0})

        text_tracks = [t for t in result.tracks if t.kind == "text"]
        assert len(text_tracks) == 1
        item = text_tracks[0].items[0]
        assert item.id == "cap-1"
        assert (item.start, item.end) == (0.0, 2.0)  # default 2s duration
        assert item.payload == {"content": "3 months in", "style": "karaoke", "anchor": "top"}

    def test_appends_subsequent_items_with_incrementing_ids(self) -> None:
        timeline = add_text(Timeline(tracks=[]), {"content": "a", "style": "static", "start": 0.0})
        result = add_text(timeline, {"content": "b", "style": "static", "start": 3.0, "duration": 1.5})

        items = result.tracks[0].items
        assert [item.id for item in items] == ["cap-1", "cap-2"]
        assert (items[1].start, items[1].end) == (3.0, 4.5)

    def test_anchor_defaults_to_bottom(self) -> None:
        result = add_text(Timeline(tracks=[]), {"content": "x", "style": "static", "start": 0.0})
        assert result.tracks[0].items[0].payload["anchor"] == "bottom"

    def test_invalid_style_rejected(self) -> None:
        with pytest.raises(ValidationError):
            add_text(Timeline(tracks=[]), {"content": "x", "style": "bogus", "start": 0.0})

    def test_does_not_disturb_other_tracks(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = add_text(timeline, {"content": "x", "style": "static", "start": 0.0})
        assert any(t.kind == "video" for t in result.tracks)
        assert any(t.kind == "text" for t in result.tracks)


# --- emoji_adder -------------------------------------------------------------


class TestEmojiAdder:
    def test_appends_item_to_new_emoji_track(self) -> None:
        timeline = Timeline(tracks=[])

        result = add_emoji(timeline, {"glyph": "\U0001f525", "at": 14.2})

        emoji_tracks = [t for t in result.tracks if t.kind == "emoji"]
        assert len(emoji_tracks) == 1
        item = emoji_tracks[0].items[0]
        assert item.id == "emoji-1"
        assert (item.start, item.end) == (14.2, 15.2)  # default 1s duration
        assert item.payload == {"glyph": "\U0001f525"}

    def test_custom_duration(self) -> None:
        result = add_emoji(Timeline(tracks=[]), {"glyph": "x", "at": 1.0, "duration": 0.5})
        assert result.tracks[0].items[0].end == pytest.approx(1.5)

    def test_incrementing_ids_across_calls(self) -> None:
        timeline = add_emoji(Timeline(tracks=[]), {"glyph": "a", "at": 0.0})
        result = add_emoji(timeline, {"glyph": "b", "at": 1.0})
        assert [item.id for item in result.tracks[0].items] == ["emoji-1", "emoji-2"]


# --- effects -------------------------------------------------------------


class TestEffects:
    def test_resolves_span_from_shot_on_video_track(self) -> None:
        timeline = _video_timeline(("s1", 2.0, 5.0))

        result = apply_effect(timeline, {"kind": "zoom_in", "shot": "s1"})

        effect_tracks = [t for t in result.tracks if t.kind == "effect"]
        item = effect_tracks[0].items[0]
        assert (item.start, item.end) == (2.0, 5.0)
        assert item.payload == {"kind": "zoom_in", "shot": "s1"}

    def test_explicit_span_overrides_shot_lookup(self) -> None:
        timeline = Timeline(tracks=[])
        result = apply_effect(timeline, {"kind": "pan", "start": 1.0, "end": 2.0})
        item = result.tracks[0].items[0]
        assert (item.start, item.end) == (1.0, 2.0)

    def test_shot_not_found_raises(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        with pytest.raises(ValueError, match="s2"):
            apply_effect(timeline, {"kind": "zoom_in", "shot": "s2"})

    def test_requires_shot_or_explicit_span(self) -> None:
        with pytest.raises(ValidationError):
            apply_effect(Timeline(tracks=[]), {"kind": "zoom_in"})

    def test_incrementing_ids_across_calls(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        timeline = apply_effect(timeline, {"kind": "zoom_in", "shot": "s1"})
        result = apply_effect(timeline, {"kind": "pan", "shot": "s2"})
        effect_track = next(t for t in result.tracks if t.kind == "effect")
        assert [item.id for item in effect_track.items] == ["fx-1", "fx-2"]


# --- tool manifest / integration -----------------------------------------


class TestToolManifest:
    def test_manifest_maps_every_editplan_tool_name(self, edit_plan_data: dict) -> None:
        plan = EditPlan.model_validate(edit_plan_data)
        for op in plan.ops:
            assert op.tool in TOOL_MANIFEST

    def test_dispatching_full_plan_produces_expected_timeline(self, edit_plan_data: dict) -> None:
        """Replays fixtures/edit_plan.json's ops through TOOL_MANIFEST end to end.

        No render happens here — this only checks that dispatch, in order,
        builds the Timeline we'd expect, including that `effect`'s shot
        lookup resolves against the *post-cut* video track.
        """
        plan = EditPlan.model_validate(edit_plan_data)
        timeline = _video_timeline(("s1", 0.0, 2.5), ("s2", 2.5, 5.0))

        for op in plan.ops:
            timeline = TOOL_MANIFEST[op.tool](timeline, op.params)

        video_track = next(t for t in timeline.tracks if t.kind == "video")
        text_track = next(t for t in timeline.tracks if t.kind == "text")
        emoji_track = next(t for t in timeline.tracks if t.kind == "emoji")
        effect_track = next(t for t in timeline.tracks if t.kind == "effect")

        assert [item.id for item in video_track.items] == ["clip-s1", "clip-s2"]
        assert text_track.items[0].payload["content"] == "3 months in"
        assert emoji_track.items[0].payload["glyph"] == "\U0001f525"
        # zoom_in on s1 should resolve to s1's current (unchanged, already-first) span.
        assert (effect_track.items[0].start, effect_track.items[0].end) == (0.0, 2.5)
