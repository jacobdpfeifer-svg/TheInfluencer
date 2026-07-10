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
from autoedit.subsystems import TOOL_MANIFEST, TOOL_PARAMS_MANIFEST
from autoedit.subsystems.cutter import cut
from autoedit.subsystems.effects import EffectParams, apply_effect
from autoedit.subsystems.emoji_adder import add_emoji
from autoedit.subsystems.music import MusicParams, add_music
from autoedit.subsystems.reframe import ReframeParams, apply_reframe
from autoedit.subsystems.text_adder import add_text
from autoedit.subsystems.transitions import apply_transition


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

    def test_trim_caps_a_shot_longer_than_its_cap(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 8.0), ("s2", 8.0, 10.0))

        result = cut(timeline, {"keep": ["s1", "s2"], "trim": {"s1": 3.0}})

        video_track = result.tracks[0]
        assert (video_track.items[0].start, video_track.items[0].end) == (0.0, 3.0)
        # s2 was untouched (not in `trim`) and starts right after s1's trimmed span.
        assert (video_track.items[1].start, video_track.items[1].end) == (3.0, 5.0)

    def test_trim_shrinks_the_payloads_out_point_from_its_in_point(self) -> None:
        timeline = Timeline(
            tracks=[
                Track(
                    name="v1", kind="video",
                    items=[TimelineItem(id="s1", start=0.0, end=8.0, payload={"shot": "s1", "in": 2.0, "out": 10.0})],
                )
            ]
        )

        result = cut(timeline, {"keep": ["s1"], "trim": {"s1": 3.0}})

        assert result.tracks[0].items[0].payload["out"] == pytest.approx(5.0)  # in(2.0) + capped duration(3.0)

    def test_trim_is_a_noop_for_a_shot_already_shorter_than_its_cap(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))

        result = cut(timeline, {"keep": ["s1"], "trim": {"s1": 10.0}})

        assert (result.tracks[0].items[0].start, result.tracks[0].items[0].end) == (0.0, 2.0)
        assert "out" not in result.tracks[0].items[0].payload

    def test_trim_ignores_shot_ids_not_in_keep(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = cut(timeline, {"keep": ["s1"], "trim": {"unrelated-shot": 1.0}})
        assert (result.tracks[0].items[0].start, result.tracks[0].items[0].end) == (0.0, 2.0)

    def test_trim_value_must_be_positive(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        with pytest.raises(ValueError, match="> 0"):
            cut(timeline, {"keep": ["s1"], "trim": {"s1": 0.0}})

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

    def test_beat_sync_falls_back_to_the_timelines_own_beat_times_when_params_omit_them(self) -> None:
        timeline = Timeline(
            tracks=[t for t in _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0)).tracks],
            beat_times=[2.1],
        )

        result = cut(timeline, {"keep": ["s1", "s2"], "sync": "beat"})

        assert result.tracks[0].items[0].end == pytest.approx(2.1)

    def test_explicit_params_beat_times_take_priority_over_the_timelines_own(self) -> None:
        timeline = Timeline(
            tracks=[t for t in _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0)).tracks],
            beat_times=[10.0],  # would leave the cut point untouched if used
        )

        result = cut(timeline, {"keep": ["s1", "s2"], "sync": "beat", "beat_times": [2.1]})

        assert result.tracks[0].items[0].end == pytest.approx(2.1)

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


# --- effects: the 5 new kinds (speed_ramp, flash, shake, ken_burns, blur_intro) ---


class TestNewEffectKinds:
    @pytest.mark.parametrize("kind", ["speed_ramp", "flash", "shake", "ken_burns", "blur_intro"])
    def test_adds_a_correctly_shaped_item_to_the_effect_track(self, kind: str) -> None:
        timeline = _video_timeline(("s1", 2.0, 5.0))

        result = apply_effect(timeline, {"kind": kind, "shot": "s1"})

        effect_tracks = [t for t in result.tracks if t.kind == "effect"]
        assert len(effect_tracks) == 1
        item = effect_tracks[0].items[0]
        assert item.id == "fx-1"
        assert (item.start, item.end) == (2.0, 5.0)  # resolved from s1's current span, same as zoom_in/zoom_out
        assert item.payload["kind"] == kind
        assert item.payload["shot"] == "s1"

    @staticmethod
    def _effect_item(timeline: Timeline):
        return next(t for t in timeline.tracks if t.kind == "effect").items[0]

    def test_speed_ramp_defaults_factor_to_1_5_on_the_timeline(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = apply_effect(timeline, {"kind": "speed_ramp", "shot": "s1"})
        assert self._effect_item(result).payload["factor"] == pytest.approx(1.5)

    def test_speed_ramp_accepts_a_custom_factor(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = apply_effect(timeline, {"kind": "speed_ramp", "shot": "s1", "factor": 2.0})
        assert self._effect_item(result).payload["factor"] == pytest.approx(2.0)

    def test_non_speed_ramp_kinds_have_no_factor_key(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        for kind in ("zoom_in", "flash", "shake", "ken_burns", "blur_intro"):
            result = apply_effect(timeline, {"kind": kind, "shot": "s1"})
            assert "factor" not in self._effect_item(result).payload


class TestEffectParamsFactorValidation:
    def test_factor_is_accepted_for_speed_ramp(self) -> None:
        params = EffectParams.model_validate({"kind": "speed_ramp", "shot": "s1", "factor": 2.0})
        assert params.factor == 2.0

    def test_factor_defaults_to_none_when_omitted(self) -> None:
        params = EffectParams.model_validate({"kind": "speed_ramp", "shot": "s1"})
        assert params.factor is None

    @pytest.mark.parametrize("kind", ["zoom_in", "zoom_out", "flash", "shake", "ken_burns", "blur_intro"])
    def test_factor_is_rejected_for_every_non_speed_ramp_kind(self, kind: str) -> None:
        with pytest.raises(ValidationError, match="factor"):
            EffectParams.model_validate({"kind": kind, "shot": "s1", "factor": 2.0})

    def test_factor_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            EffectParams.model_validate({"kind": "speed_ramp", "shot": "s1", "factor": 0.0})


# --- transitions -----------------------------------------------------------


class TestTransitions:
    def test_fade_adds_an_item_to_a_new_transition_track(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))

        result = apply_transition(timeline, {"kind": "fade", "between": ["s1", "s2"]})

        transition_tracks = [t for t in result.tracks if t.kind == "transition"]
        assert len(transition_tracks) == 1
        item = transition_tracks[0].items[0]
        assert item.id == "trans-1"
        assert item.payload == {"kind": "fade", "between": ["s1", "s2"]}

    def test_whip_pan_adds_an_item_to_the_transition_track(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        result = apply_transition(timeline, {"kind": "whip_pan", "between": ["s1", "s2"]})
        transition_track = next(t for t in result.tracks if t.kind == "transition")
        assert transition_track.items[0].payload["kind"] == "whip_pan"

    @staticmethod
    def _transition_item(timeline: Timeline):
        return next(t for t in timeline.tracks if t.kind == "transition").items[0]

    def test_span_is_centered_on_the_cut_point_between_the_two_shots(self) -> None:
        # Cut point (s1's end / s2's start) is at t=2.0; default duration=0.3 -> +/-0.15.
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        result = apply_transition(timeline, {"kind": "fade", "between": ["s1", "s2"]})
        item = self._transition_item(result)
        assert (item.start, item.end) == pytest.approx((1.85, 2.15))

    def test_custom_duration_widens_the_span(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        result = apply_transition(timeline, {"kind": "fade", "between": ["s1", "s2"], "duration": 1.0})
        item = self._transition_item(result)
        assert (item.start, item.end) == pytest.approx((1.5, 2.5))

    def test_span_is_clamped_at_zero_near_the_start_of_the_timeline(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 0.1), ("s2", 0.1, 2.0))
        result = apply_transition(timeline, {"kind": "fade", "between": ["s1", "s2"], "duration": 1.0})
        item = self._transition_item(result)
        assert item.start == 0.0

    def test_missing_outgoing_shot_raises(self) -> None:
        timeline = _video_timeline(("s2", 2.0, 4.0))
        with pytest.raises(ValueError, match="s1"):
            apply_transition(timeline, {"kind": "fade", "between": ["s1", "s2"]})

    def test_missing_incoming_shot_raises(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        with pytest.raises(ValueError, match="s2"):
            apply_transition(timeline, {"kind": "fade", "between": ["s1", "s2"]})

    def test_rejects_a_between_list_that_is_not_exactly_two_shots(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        with pytest.raises(ValidationError):
            apply_transition(timeline, {"kind": "fade", "between": ["s1"]})

    def test_rejects_unknown_kind(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        with pytest.raises(ValidationError):
            apply_transition(timeline, {"kind": "swipe", "between": ["s1", "s2"]})

    def test_incrementing_ids_across_calls(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0), ("s3", 4.0, 6.0))
        timeline = apply_transition(timeline, {"kind": "fade", "between": ["s1", "s2"]})
        result = apply_transition(timeline, {"kind": "whip_pan", "between": ["s2", "s3"]})
        transition_track = next(t for t in result.tracks if t.kind == "transition")
        assert [item.id for item in transition_track.items] == ["trans-1", "trans-2"]

    def test_does_not_disturb_other_tracks(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        result = apply_transition(timeline, {"kind": "fade", "between": ["s1", "s2"]})
        assert any(t.kind == "video" for t in result.tracks)
        assert any(t.kind == "transition" for t in result.tracks)


# --- reframe ---------------------------------------------------------------


class TestReframe:
    @staticmethod
    def _reframe_item(timeline: Timeline):
        return next(t for t in timeline.tracks if t.kind == "reframe").items[0]

    def test_default_kind_is_center_crop_at_9x16(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = apply_reframe(timeline, {"shot": "s1"})
        item = self._reframe_item(result)
        assert item.payload["kind"] == "center_crop"
        assert item.payload["target_aspect"] == pytest.approx(9 / 16)

    def test_span_is_taken_from_the_shots_video_item(self) -> None:
        timeline = _video_timeline(("s1", 1.0, 3.5))
        result = apply_reframe(timeline, {"shot": "s1", "kind": "rule_of_thirds"})
        item = self._reframe_item(result)
        assert (item.start, item.end) == (1.0, 3.5)

    def test_explicit_span_overrides_shot_lookup(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = apply_reframe(timeline, {"start": 0.5, "end": 1.5, "kind": "fit"})
        item = self._reframe_item(result)
        assert (item.start, item.end) == (0.5, 1.5)

    def test_custom_target_aspect_is_carried_onto_the_payload(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = apply_reframe(timeline, {"shot": "s1", "target_aspect": 16 / 9})
        assert self._reframe_item(result).payload["target_aspect"] == pytest.approx(16 / 9)

    def test_missing_shot_raises(self) -> None:
        timeline = _video_timeline(("s2", 2.0, 4.0))
        with pytest.raises(ValueError, match="s1"):
            apply_reframe(timeline, {"shot": "s1"})

    def test_neither_shot_nor_explicit_span_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReframeParams.model_validate({"kind": "fit"})

    def test_rejects_unknown_kind(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        with pytest.raises(ValidationError):
            apply_reframe(timeline, {"shot": "s1", "kind": "dutch_angle"})

    def test_incrementing_ids_across_calls(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        timeline = apply_reframe(timeline, {"shot": "s1"})
        result = apply_reframe(timeline, {"shot": "s2"})
        reframe_track = next(t for t in result.tracks if t.kind == "reframe")
        assert [item.id for item in reframe_track.items] == ["reframe-1", "reframe-2"]

    def test_does_not_disturb_other_tracks(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = apply_reframe(timeline, {"shot": "s1"})
        assert any(t.kind == "video" for t in result.tracks)
        assert any(t.kind == "reframe" for t in result.tracks)


# --- music -------------------------------------------------------------


class TestMusic:
    @staticmethod
    def _music_item(timeline: Timeline):
        return next(t for t in timeline.tracks if t.kind == "audio").items[0]

    def test_defaults_span_to_the_video_tracks_total_duration(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 5.0))
        result = add_music(timeline, {"source": "song.mp3"})
        item = self._music_item(result)
        assert (item.start, item.end) == (0.0, 5.0)
        assert item.payload["source"] == "song.mp3"

    def test_explicit_duration_overrides_the_video_track_span(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = add_music(timeline, {"source": "song.mp3", "duration": 10.0})
        assert self._music_item(result).end == 10.0

    def test_offset_and_volume_are_carried_onto_the_payload(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = add_music(timeline, {"source": "song.mp3", "offset": 8.0, "volume": 0.3})
        item = self._music_item(result)
        assert item.payload["in"] == pytest.approx(8.0)
        assert item.payload["out"] == pytest.approx(10.0)
        assert item.payload["volume"] == pytest.approx(0.3)

    def test_default_volume_is_below_unity(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = add_music(timeline, {"source": "song.mp3"})
        assert 0 < self._music_item(result).payload["volume"] < 1.0

    def test_empty_video_track_and_no_explicit_duration_raises(self) -> None:
        timeline = Timeline(tracks=[])
        with pytest.raises(ValueError, match="duration"):
            add_music(timeline, {"source": "song.mp3"})

    def test_volume_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            MusicParams.model_validate({"source": "song.mp3", "volume": 5.0})

    def test_missing_source_raises(self) -> None:
        with pytest.raises(ValidationError):
            MusicParams.model_validate({})

    def test_incrementing_ids_across_calls(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        timeline = add_music(timeline, {"source": "a.mp3"})
        result = add_music(timeline, {"source": "b.mp3"})
        audio_track = next(t for t in result.tracks if t.kind == "audio")
        assert [item.id for item in audio_track.items] == ["music-1", "music-2"]

    def test_does_not_disturb_other_tracks(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0))
        result = add_music(timeline, {"source": "song.mp3"})
        assert any(t.kind == "video" for t in result.tracks)
        assert any(t.kind == "audio" for t in result.tracks)


# --- tool manifest / integration -----------------------------------------


class TestToolManifest:
    def test_manifest_maps_every_editplan_tool_name(self, edit_plan_data: dict) -> None:
        plan = EditPlan.model_validate(edit_plan_data)
        for op in plan.ops:
            assert op.tool in TOOL_MANIFEST

    def test_manifest_includes_the_transition_tool(self) -> None:
        assert "transition" in TOOL_MANIFEST
        assert "transition" in TOOL_PARAMS_MANIFEST

    def test_manifest_includes_the_reframe_tool(self) -> None:
        assert "reframe" in TOOL_MANIFEST
        assert "reframe" in TOOL_PARAMS_MANIFEST

    def test_manifest_includes_the_music_tool(self) -> None:
        assert "music" in TOOL_MANIFEST
        assert "music" in TOOL_PARAMS_MANIFEST

    def test_dispatching_a_transition_op_through_the_manifest(self) -> None:
        timeline = _video_timeline(("s1", 0.0, 2.0), ("s2", 2.0, 4.0))
        timeline = TOOL_MANIFEST["cutter"](timeline, {"keep": ["s1", "s2"]})
        result = TOOL_MANIFEST["transition"](timeline, {"kind": "fade", "between": ["s1", "s2"]})
        transition_track = next(t for t in result.tracks if t.kind == "transition")
        assert transition_track.items[0].payload["kind"] == "fade"

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
