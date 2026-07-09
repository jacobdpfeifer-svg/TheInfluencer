"""Tests for autoedit/executor.py.

`build_initial_timeline` and `execute` are pure (Timeline in, Timeline out)
and tested directly on fixture JSON, per AGENTS.md's testing rule. `run`'s
own logic (seed -> execute -> render, in order, with the right arguments) is
tested with `renderer.render` monkeypatched out — no real render here; see
`tests/test_end_to_end.py` for the one deliberate real-render exception.
"""

from __future__ import annotations

import pytest

from autoedit.executor import ExecutorError, build_initial_timeline, execute, run
from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditOp, EditPlan
from autoedit.models.timeline import Timeline, TimelineItem, Track


@pytest.fixture
def features(content_features_data: dict) -> ContentFeatures:
    return ContentFeatures.model_validate(content_features_data)


class TestBuildInitialTimeline:
    def test_seeds_one_video_item_per_shot_at_its_natural_position(self, features) -> None:
        timeline = build_initial_timeline(features)

        video_track = next(t for t in timeline.tracks if t.kind == "video")
        assert [item.id for item in video_track.items] == ["clip-s1", "clip-s2"]
        assert (video_track.items[0].start, video_track.items[0].end) == (0.0, 2.5)
        assert (video_track.items[1].start, video_track.items[1].end) == (2.5, 5.0)

    def test_payload_carries_shot_source_in_out(self, features) -> None:
        timeline = build_initial_timeline(features)
        item = timeline.tracks[0].items[0]
        assert item.payload == {"shot": "s1", "source": "raw/clip_001.mp4", "in": 0.0, "out": 2.5}

    def test_carries_beat_times_from_features_onto_the_timeline(self, content_features_data: dict) -> None:
        content_features_data["beat_times"] = [0.5, 1.5, 2.5]
        features = ContentFeatures.model_validate(content_features_data)
        timeline = build_initial_timeline(features)
        assert timeline.beat_times == [0.5, 1.5, 2.5]

    def test_defaults_to_empty_beat_times_when_features_has_none(self, features) -> None:
        timeline = build_initial_timeline(features)
        assert timeline.beat_times == []


class TestExecute:
    def test_dispatches_ops_in_order_through_the_manifest(self, features) -> None:
        timeline = build_initial_timeline(features)
        plan = EditPlan(ops=[EditOp(tool="cutter", params={"keep": ["s2", "s1"]})], confidence=1.0)

        result = execute(plan, timeline)

        assert [item.id for item in result.tracks[0].items] == ["clip-s2", "clip-s1"]

    def test_unknown_tool_raises_executor_error(self, features) -> None:
        timeline = build_initial_timeline(features)
        plan = EditPlan(ops=[EditOp(tool="not_a_real_tool", params={})], confidence=1.0)

        with pytest.raises(ExecutorError, match="not_a_real_tool"):
            execute(plan, timeline)

    def test_custom_manifest_is_honored(self, features) -> None:
        timeline = build_initial_timeline(features)
        plan = EditPlan(ops=[EditOp(tool="noop", params={})], confidence=1.0)
        custom_manifest = {"noop": lambda tl, params: tl}

        result = execute(plan, timeline, manifest=custom_manifest)

        assert result == timeline

    def test_empty_video_timeline_plus_text_op_still_works(self) -> None:
        timeline = Timeline(tracks=[])
        plan = EditPlan(ops=[EditOp(tool="text", params={"content": "hi", "style": "static", "start": 0.0})], confidence=1.0)

        result = execute(plan, timeline)

        text_track = next(t for t in result.tracks if t.kind == "text")
        assert text_track.items[0].payload["content"] == "hi"


class TestBeatSyncFullChain:
    """The full chain the audit was about: content.extract's beat_times ->
    heuristic_plan's cutter op params -> the cutter subsystem's actual snap.
    Confirms the wiring isn't dead code end to end, not just at each seam.
    """

    def test_heuristic_plan_dispatched_through_execute_snaps_a_cut_to_a_beat(self, content_features_data: dict) -> None:
        from autoedit.director.heuristic import _simple_heuristic_plan
        from autoedit.models.style_profile import StyleProfile

        # s1 ends at 2.5s (natural cut point); a beat at 2.6s is within the
        # cutter's 0.5s snap tolerance, so the cut should move to land on it.
        content_features_data["music_bpm"] = 128.0
        content_features_data["beat_times"] = [0.5, 2.6, 5.0]
        features = ContentFeatures.model_validate(content_features_data)
        style = StyleProfile.model_validate(
            {
                "aspect": 0.5625,
                "shot_len_median": 2.5,
                "shot_len_spread": 0.5,
                "cut_on_beat": True,
                "caption_style_freq": {"karaoke": 0.5, "static": 0.5},
                "caption_density": 0.0,
                "text_amount": 0.0,
                "effect_freq": 0.0,
                "sample_count": 1,
            }
        )

        plan = _simple_heuristic_plan(style, features)
        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert cutter_op.params["sync"] == "beat"
        assert cutter_op.params["beat_times"] == [0.5, 2.6, 5.0]

        timeline = build_initial_timeline(features)
        result = execute(plan, timeline)

        video_items = result.tracks[0].items
        assert video_items[0].end == pytest.approx(2.6)  # snapped from its natural 2.5
        assert video_items[1].start == pytest.approx(2.6)


class TestRun:
    def test_seeds_executes_and_renders_exactly_once_in_order(self, features, monkeypatch) -> None:
        calls = []

        def fake_render(timeline, output_path):
            calls.append((timeline, output_path))
            return output_path

        monkeypatch.setattr("autoedit.executor.render", fake_render)

        plan = EditPlan(ops=[EditOp(tool="cutter", params={"keep": ["s2"]})], confidence=1.0)
        result = run(features, plan, "out.mp4")

        assert len(calls) == 1  # rendered exactly once
        rendered_timeline, output_path = calls[0]
        assert output_path == "out.mp4"
        assert [item.id for item in rendered_timeline.tracks[0].items] == ["clip-s2"]
        assert result == "out.mp4"

    def test_run_propagates_executor_errors_without_rendering(self, features, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr("autoedit.executor.render", lambda timeline, output_path: calls.append(1))

        plan = EditPlan(ops=[EditOp(tool="bogus", params={})], confidence=1.0)
        with pytest.raises(ExecutorError):
            run(features, plan, "out.mp4")
        assert calls == []  # never reached the renderer
