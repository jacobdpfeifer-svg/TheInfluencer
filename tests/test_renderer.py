"""Tests for autoedit/renderer.py's pure planning logic.

Per AGENTS.md's testing rule and `.cursor/rules/subsystems.mdc`, `render()`
itself (the actual MoviePy composite pass) is never invoked here — only
`build_render_plan`, which is a pure Timeline -> RenderPlan translation with
no file IO. See `renderer.py`'s module docstring for the rationale.
"""

from __future__ import annotations

import pytest

from autoedit import renderer
from autoedit.models.timeline import Timeline, TimelineItem, Track
from autoedit.renderer import EffectInstruction, VideoSegment, build_render_plan


def _timeline(*tracks: Track) -> Timeline:
    return Timeline(tracks=list(tracks))


class TestBuildRenderPlan:
    def test_empty_timeline_yields_empty_plan(self) -> None:
        plan = build_render_plan(Timeline(tracks=[]))
        assert plan.video_segments == []
        assert plan.text_overlays == []
        assert plan.emoji_overlays == []
        assert plan.effects == []
        assert plan.duration == 0.0

    def test_video_segments_resolved_from_payload(self) -> None:
        track = Track(
            name="v1",
            kind="video",
            items=[
                TimelineItem(
                    id="clip-s1", start=0.0, end=2.0, payload={"shot": "s1", "source": "raw.mp4", "in": 1.0, "out": 3.0}
                ),
                TimelineItem(id="clip-s2", start=2.0, end=3.5, payload={"shot": "s2", "source": "raw2.mp4"}),
            ],
        )
        plan = build_render_plan(_timeline(track))

        assert len(plan.video_segments) == 2
        seg1, seg2 = plan.video_segments
        assert (seg1.source, seg1.in_, seg1.out_) == ("raw.mp4", 1.0, 3.0)
        assert (seg1.output_start, seg1.output_end, seg1.shot) == (0.0, 2.0, "s1")
        # seg2 has no explicit in/out -> defaults to (0, duration).
        assert (seg2.source, seg2.in_, seg2.out_) == ("raw2.mp4", 0.0, 1.5)

    def test_video_segments_sorted_by_output_start(self) -> None:
        track = Track(
            name="v1",
            kind="video",
            items=[
                TimelineItem(id="clip-s2", start=2.0, end=4.0, payload={"source": "b.mp4"}),
                TimelineItem(id="clip-s1", start=0.0, end=2.0, payload={"source": "a.mp4"}),
            ],
        )
        plan = build_render_plan(_timeline(track))
        assert [segment.source for segment in plan.video_segments] == ["a.mp4", "b.mp4"]

    def test_missing_source_raises(self) -> None:
        track = Track(name="v1", kind="video", items=[TimelineItem(id="clip-s1", start=0.0, end=1.0, payload={})])
        with pytest.raises(ValueError, match="source"):
            build_render_plan(_timeline(track))

    def test_text_overlays_resolved_with_defaults(self) -> None:
        track = Track(
            name="captions",
            kind="text",
            items=[TimelineItem(id="cap-1", start=0.0, end=2.0, payload={"content": "hi", "style": "karaoke"})],
        )
        plan = build_render_plan(_timeline(track))
        overlay = plan.text_overlays[0]
        assert (overlay.content, overlay.style, overlay.anchor) == ("hi", "karaoke", "bottom")

    def test_emoji_overlays_resolved(self) -> None:
        track = Track(
            name="emoji", kind="emoji", items=[TimelineItem(id="emoji-1", start=1.0, end=2.0, payload={"glyph": "\U0001f525"})]
        )
        plan = build_render_plan(_timeline(track))
        assert plan.emoji_overlays[0].glyph == "\U0001f525"

    def test_effects_resolved(self) -> None:
        track = Track(
            name="effects",
            kind="effect",
            items=[TimelineItem(id="fx-1", start=0.0, end=2.5, payload={"kind": "zoom_in", "shot": "s1"})],
        )
        plan = build_render_plan(_timeline(track))
        effect = plan.effects[0]
        assert (effect.kind, effect.shot, effect.start, effect.end) == ("zoom_in", "s1", 0.0, 2.5)

    def test_duration_is_max_end_across_all_tracks(self) -> None:
        video = Track(name="v1", kind="video", items=[TimelineItem(id="c1", start=0.0, end=5.0, payload={"source": "a.mp4"})])
        emoji = Track(name="emoji", kind="emoji", items=[TimelineItem(id="e1", start=4.5, end=6.0, payload={"glyph": "x"})])
        plan = build_render_plan(_timeline(video, emoji))
        assert plan.duration == pytest.approx(6.0)

    def test_full_fixture_timeline_builds_a_plan(self, timeline_data: dict) -> None:
        """Sanity check against the repo's own fixtures/timeline.json."""
        timeline = Timeline.model_validate(timeline_data)
        # This fixture's video items have no 'source' payload -- expected to
        # raise, since the renderer can't resolve pixels without one.
        with pytest.raises(ValueError, match="source"):
            build_render_plan(timeline)

    def test_effect_factor_is_resolved_for_speed_ramp(self) -> None:
        track = Track(
            name="effects",
            kind="effect",
            items=[TimelineItem(id="fx-1", start=0.0, end=2.0, payload={"kind": "speed_ramp", "shot": "s1", "factor": 2.0})],
        )
        plan = build_render_plan(_timeline(track))
        assert plan.effects[0].factor == pytest.approx(2.0)

    def test_effect_factor_defaults_to_none_for_other_kinds(self) -> None:
        track = Track(
            name="effects", kind="effect", items=[TimelineItem(id="fx-1", start=0.0, end=2.0, payload={"kind": "shake", "shot": "s1"})]
        )
        plan = build_render_plan(_timeline(track))
        assert plan.effects[0].factor is None

    def test_transitions_resolved_from_the_transition_track(self) -> None:
        track = Track(
            name="transitions",
            kind="transition",
            items=[
                TimelineItem(id="trans-1", start=1.85, end=2.15, payload={"kind": "fade", "between": ["s1", "s2"]})
            ],
        )
        plan = build_render_plan(_timeline(track))
        assert len(plan.transitions) == 1
        transition = plan.transitions[0]
        assert (transition.kind, transition.between, transition.start, transition.end) == ("fade", ["s1", "s2"], 1.85, 2.15)

    def test_transitions_default_to_empty(self) -> None:
        plan = build_render_plan(Timeline(tracks=[]))
        assert plan.transitions == []

    def test_transitions_sorted_by_start(self) -> None:
        track = Track(
            name="transitions",
            kind="transition",
            items=[
                TimelineItem(id="trans-2", start=5.0, end=5.3, payload={"kind": "fade", "between": ["s2", "s3"]}),
                TimelineItem(id="trans-1", start=1.85, end=2.15, payload={"kind": "whip_pan", "between": ["s1", "s2"]}),
            ],
        )
        plan = build_render_plan(_timeline(track))
        assert [t.between for t in plan.transitions] == [["s1", "s2"], ["s2", "s3"]]


class TestApplyEffectsToClipNoOp:
    """`_apply_effects_to_clip` is the ONE place effect kinds get dispatched to a real clip
    transform; per `.cursor/rules/subsystems.mdc`'s "AI raises the ceiling; rules are the
    floor", an unrecognized kind must be a silent no-op rather than a crash. These use a
    bare sentinel object as the "clip" -- no MoviePy/real media needed, since a no-op never
    calls into the clip at all.
    """

    def test_unknown_effect_kind_returns_the_same_clip_object_unchanged(self) -> None:
        segment = VideoSegment(source="a.mp4", in_=0.0, out_=1.0, output_start=0.0, output_end=1.0, shot="s1")
        effect = EffectInstruction(kind="teleport", shot="s1", start=0.0, end=1.0)
        sentinel_clip = object()

        result = renderer._apply_effects_to_clip(sentinel_clip, segment, [effect])

        assert result is sentinel_clip

    def test_flash_is_also_a_noop_here_since_it_is_handled_as_an_overlay_not_a_clip_transform(self) -> None:
        segment = VideoSegment(source="a.mp4", in_=0.0, out_=1.0, output_start=0.0, output_end=1.0, shot="s1")
        effect = EffectInstruction(kind="flash", shot="s1", start=0.0, end=1.0)
        sentinel_clip = object()

        result = renderer._apply_effects_to_clip(sentinel_clip, segment, [effect])

        assert result is sentinel_clip

    def test_effect_scoped_to_a_different_shot_is_a_noop(self) -> None:
        segment = VideoSegment(source="a.mp4", in_=0.0, out_=1.0, output_start=0.0, output_end=1.0, shot="s1")
        effect = EffectInstruction(kind="zoom_in", shot="s2", start=0.0, end=1.0)
        sentinel_clip = object()

        result = renderer._apply_effects_to_clip(sentinel_clip, segment, [effect])

        assert result is sentinel_clip

    def test_known_effect_kind_does_call_into_the_clip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Contrast case: a *registered* kind (zoom_in) must actually be dispatched, not no-op'd."""
        calls = []
        monkeypatch.setitem(renderer._EFFECT_FUNCS, "zoom_in", lambda clip, duration, effect: calls.append(1) or "transformed")

        segment = VideoSegment(source="a.mp4", in_=0.0, out_=1.0, output_start=0.0, output_end=1.0, shot="s1")
        effect = EffectInstruction(kind="zoom_in", shot="s1", start=0.0, end=1.0)

        result = renderer._apply_effects_to_clip(object(), segment, [effect])

        assert calls == [1]
        assert result == "transformed"
