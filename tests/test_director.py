"""Tests for autoedit/director/*.

Pure fixture-JSON tests (per AGENTS.md's testing rule) — the director never
touches media or a Timeline, only ContentFeatures/StyleProfile/EditPlan
JSON. No real LLM is ever called; `direct()` is exercised with the default
stub (always garbage -> fallback) and with hand-crafted `llm` callables that
return good, bad, low-confidence, and exception-raising responses.
"""

from __future__ import annotations

import pytest

from autoedit.director.brief import build_brief
from autoedit.director.director import direct
from autoedit.director.heuristic import HEURISTIC_CONFIDENCE, _simple_heuristic_plan, heuristic_plan
from autoedit.director.llm import stub_llm
from autoedit.director.validate import PlanValidationError, validate_plan
from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditPlan
from autoedit.models.style_profile import StyleProfile
from autoedit.subsystems import TOOL_MANIFEST
from autoedit.templates.filler import TEMPLATE_FILL_CONFIDENCE


@pytest.fixture
def features(content_features_data: dict) -> ContentFeatures:
    return ContentFeatures.model_validate(content_features_data)


@pytest.fixture
def style(style_profile_data: dict) -> StyleProfile:
    return StyleProfile.model_validate(style_profile_data)


def _bare_style(**overrides) -> StyleProfile:
    base = dict(
        aspect=0.5625,
        shot_len_median=2.0,
        shot_len_spread=0.5,
        cut_on_beat=False,
        caption_style_freq={"karaoke": 0.5, "static": 0.5},
        caption_density=0.0,
        text_amount=0.0,
        effect_freq=0.0,
        sample_count=1,
    )
    base.update(overrides)
    return StyleProfile.model_validate(base)


def _bare_features(**overrides) -> ContentFeatures:
    base = dict(
        aspect=0.5625,
        has_speech=False,
        music_bpm=None,
        shots=[
            {
                "id": "s1",
                "source": "raw.mp4",
                "in": 0.0,
                "out": 2.0,
                "dur": 2.0,
                "motion": 0.1,
                "brightness": 0.5,
                "sharpness": 50.0,
                "faces": 0,
                "scale": "wide",
            }
        ],
        motion="low",
        is_vertical=True,
        has_face=False,
    )
    base.update(overrides)
    return ContentFeatures.model_validate(base)


# --- build_brief -----------------------------------------------------------


class TestBuildBrief:
    def test_brief_is_compact_and_json_serializable(self, features, style) -> None:
        import json

        brief = build_brief(features, style, TOOL_MANIFEST)
        json.dumps(brief)  # must not raise

    def test_brief_includes_quantized_shot_motion_buckets(self, features, style) -> None:
        brief = build_brief(features, style, TOOL_MANIFEST)
        shots = brief["features"]["shots"]
        assert shots[0]["id"] == "s1"
        assert shots[0]["motion"] in ("low", "med", "high")
        assert shots[1]["motion"] in ("low", "med", "high")

    def test_brief_includes_style_and_sorted_tool_names(self, features, style) -> None:
        brief = build_brief(features, style, TOOL_MANIFEST)
        assert brief["style"]["cut_on_beat"] is True
        assert brief["style"]["sample_count"] == 12
        assert brief["tools"] == sorted(TOOL_MANIFEST.keys())

    def test_brief_omits_raw_shot_fields_not_in_the_compact_schema(self, features, style) -> None:
        brief = build_brief(features, style, TOOL_MANIFEST)
        shot_brief = brief["features"]["shots"][0]
        assert "brightness" not in shot_brief
        assert "sharpness" not in shot_brief
        assert "source" not in shot_brief


# --- validate_plan -----------------------------------------------------------


class TestValidatePlan:
    def test_valid_plan_passes(self, edit_plan_data: dict) -> None:
        plan = validate_plan(edit_plan_data)
        assert isinstance(plan, EditPlan)
        assert len(plan.ops) == 4

    @pytest.mark.parametrize("garbage", [None, "not json", 42, [], {"totally": "wrong"}])
    def test_structurally_malformed_raises(self, garbage) -> None:
        with pytest.raises(PlanValidationError):
            validate_plan(garbage)

    def test_empty_ops_raises(self) -> None:
        with pytest.raises(PlanValidationError):
            validate_plan({"ops": [], "confidence": 0.9})

    def test_unknown_tool_raises(self) -> None:
        with pytest.raises(PlanValidationError, match="unknown tool"):
            validate_plan({"ops": [{"tool": "teleporter", "params": {}}], "confidence": 0.9})

    def test_invalid_params_for_known_tool_raises(self) -> None:
        with pytest.raises(PlanValidationError, match="invalid params"):
            validate_plan({"ops": [{"tool": "cutter", "params": {"sync": "beat"}}], "confidence": 0.9})  # missing 'keep'

    def test_valid_params_for_every_tool_kind(self) -> None:
        raw = {
            "ops": [
                {"tool": "cutter", "params": {"keep": ["s1"]}},
                {"tool": "text", "params": {"content": "hi", "style": "static", "start": 0.0}},
                {"tool": "emoji", "params": {"glyph": "x", "at": 1.0}},
                {"tool": "effect", "params": {"kind": "zoom_in", "shot": "s1"}},
            ],
            "confidence": 0.7,
        }
        plan = validate_plan(raw)
        assert len(plan.ops) == 4


# --- _simple_heuristic_plan (the template-free floor) ------------------------


class TestSimpleHeuristicPlan:
    """Tests for `_simple_heuristic_plan`, the original rule-based floor.

    Called directly (bypassing `match_template`/`fill_template`) so these
    stay pinned to that specific, unstructured rule set regardless of
    whether a builtin template would also fit the fixture features/style
    used here. See `TestHeuristicPlanTemplateIntegration` below for the
    public `heuristic_plan`'s template-first routing.
    """

    def test_always_returns_a_valid_editplan(self, features, style) -> None:
        plan = _simple_heuristic_plan(style, features)
        assert isinstance(plan, EditPlan)
        # Round-trips through the same validation a real model's output would face.
        validate_plan(plan.model_dump())

    def test_confidence_is_the_fixed_heuristic_baseline(self, features, style) -> None:
        plan = _simple_heuristic_plan(style, features)
        assert plan.confidence == HEURISTIC_CONFIDENCE

    def test_cutter_keeps_every_shot_and_syncs_to_beat_when_style_prefers_it(self, features, style) -> None:
        plan = _simple_heuristic_plan(style, features)
        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert cutter_op.params["keep"] == ["s1", "s2"]
        assert cutter_op.params["sync"] == "beat"  # style.cut_on_beat=True and features.music_bpm is set

    def test_no_beat_sync_without_music(self, style) -> None:
        features = _bare_features(music_bpm=None)
        plan = _simple_heuristic_plan(style, features)
        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert cutter_op.params["sync"] == "none"

    def test_cutter_op_includes_real_beat_times_when_beat_synced(self, style) -> None:
        features = _bare_features(music_bpm=128.0, beat_times=[0.5, 1.0, 1.5, 2.0])
        plan = _simple_heuristic_plan(style, features)  # style fixture has cut_on_beat=True
        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert cutter_op.params["sync"] == "beat"
        assert cutter_op.params["beat_times"] == [0.5, 1.0, 1.5, 2.0]

    def test_no_beat_times_key_when_not_beat_synced(self, features) -> None:
        style = _bare_style(cut_on_beat=False)
        plan = _simple_heuristic_plan(style, features)
        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert "beat_times" not in cutter_op.params

    def test_no_beat_sync_when_style_does_not_prefer_it(self, features) -> None:
        style = _bare_style(cut_on_beat=False)
        plan = _simple_heuristic_plan(style, features)
        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert cutter_op.params["sync"] == "none"

    def test_text_op_prefers_majority_caption_style(self, features) -> None:
        style = _bare_style(text_amount=1.0, caption_style_freq={"karaoke": 0.7, "static": 0.3})
        plan = _simple_heuristic_plan(style, features)
        text_op = next(op for op in plan.ops if op.tool == "text")
        assert text_op.params["style"] == "karaoke"

    def test_no_text_op_when_style_has_no_text(self, features) -> None:
        style = _bare_style(text_amount=0.0)
        plan = _simple_heuristic_plan(style, features)
        assert not any(op.tool == "text" for op in plan.ops)

    def test_effect_op_targets_a_close_shot_when_style_uses_effects(self, features) -> None:
        style = _bare_style(effect_freq=0.2)
        plan = _simple_heuristic_plan(style, features)
        effect_op = next(op for op in plan.ops if op.tool == "effect")
        assert effect_op.params == {"kind": "zoom_in", "shot": "s1"}  # s1 is the 'close' shot in the fixture

    def test_no_effect_op_when_style_never_uses_effects(self, features) -> None:
        style = _bare_style(effect_freq=0.0)
        plan = _simple_heuristic_plan(style, features)
        assert not any(op.tool == "effect" for op in plan.ops)

    def test_no_effect_op_when_no_close_shot_exists(self, style) -> None:
        features = _bare_features(shots=[
            {"id": "s1", "source": "raw.mp4", "in": 0.0, "out": 2.0, "dur": 2.0, "motion": 0.1, "brightness": 0.5, "sharpness": 50.0, "faces": 0, "scale": "wide"}
        ])
        plan = _simple_heuristic_plan(_bare_style(effect_freq=0.2), features)
        assert not any(op.tool == "effect" for op in plan.ops)

    def _two_shot_features(self, **overrides) -> ContentFeatures:
        shot = {
            "source": "raw.mp4", "in": 0.0, "out": 2.0, "dur": 2.0, "motion": 0.1,
            "brightness": 0.5, "sharpness": 50.0, "faces": 0, "scale": "wide",
        }
        base = dict(shots=[{**shot, "id": "s1"}, {**shot, "id": "s2"}])
        base.update(overrides)
        return _bare_features(**base)

    def test_fade_transition_added_between_first_two_shots_when_effect_freq_is_high(self) -> None:
        style = _bare_style(effect_freq=0.6)
        features = self._two_shot_features()
        plan = _simple_heuristic_plan(style, features)
        transition_op = next(op for op in plan.ops if op.tool == "transition")
        assert transition_op.params == {"kind": "fade", "between": ["s1", "s2"]}

    def test_no_transition_op_when_effect_freq_is_at_or_below_the_threshold(self) -> None:
        features = self._two_shot_features()
        for effect_freq in (0.5, 0.2, 0.0):
            plan = _simple_heuristic_plan(_bare_style(effect_freq=effect_freq), features)
            assert not any(op.tool == "transition" for op in plan.ops)

    def test_no_transition_op_with_only_a_single_shot_even_with_high_effect_freq(self, style) -> None:
        plan = _simple_heuristic_plan(_bare_style(effect_freq=0.9), _bare_features())  # fixture has 1 shot
        assert not any(op.tool == "transition" for op in plan.ops)

    def test_transition_op_round_trips_through_validate_plan(self) -> None:
        style = _bare_style(effect_freq=0.9)
        features = self._two_shot_features()
        plan = _simple_heuristic_plan(style, features)
        validate_plan(plan.model_dump())

    def test_emoji_op_only_when_a_face_was_detected(self, features) -> None:
        plan = _simple_heuristic_plan(_bare_style(), features)  # fixture features has_face=True
        assert any(op.tool == "emoji" for op in plan.ops)

    def test_no_emoji_op_without_a_face(self, style) -> None:
        features = _bare_features(has_face=False)
        plan = _simple_heuristic_plan(style, features)
        assert not any(op.tool == "emoji" for op in plan.ops)

    def test_minimal_style_and_features_still_yields_a_valid_plan_with_only_a_cutter_op(self) -> None:
        plan = _simple_heuristic_plan(_bare_style(), _bare_features())
        assert [op.tool for op in plan.ops] == ["cutter"]
        validate_plan(plan.model_dump())


# --- heuristic_plan (template-first, with the simple plan as its fallback) ---


class TestHeuristicPlanTemplateIntegration:
    """Tests for the public `heuristic_plan`: template-fill first, `_simple_heuristic_plan` only as a last resort."""

    def test_always_returns_a_valid_editplan(self, features, style) -> None:
        plan = heuristic_plan(style, features)
        assert isinstance(plan, EditPlan)
        validate_plan(plan.model_dump())

    def test_uses_a_template_fill_when_a_template_scores_above_threshold(self, features, style) -> None:
        plan = heuristic_plan(style, features)
        # `features`/`style` (speech + face + music, 2 shots) fit `talking_head_with_broll`
        # well enough to clear the threshold, so this is a template-filled plan, not the bare floor.
        assert plan.confidence == TEMPLATE_FILL_CONFIDENCE

    def test_template_fill_produces_a_richer_plan_than_the_simple_heuristic(self) -> None:
        # A style with zero text/effect signal so `_simple_heuristic_plan` emits
        # nothing beyond its cutter op, paired with high-motion, beat-synced,
        # 5-shot footage that fits `punchy_beat_montage` (hook/payoff effects +
        # a whip-pan transition + a title card) far richer than that bare floor.
        style = _bare_style()
        shot = {
            "source": "raw.mp4", "in": 0.0, "out": 1.0, "dur": 1.0, "motion": 0.8,
            "brightness": 0.5, "sharpness": 50.0, "faces": 0, "scale": "wide",
        }
        features = _bare_features(
            music_bpm=128.0,
            motion="high",
            shots=[{**shot, "id": f"s{i}"} for i in range(1, 6)],
        )

        templated = heuristic_plan(style, features)
        simple = _simple_heuristic_plan(style, features)

        assert [op.tool for op in simple.ops] == ["cutter"]
        assert len(templated.ops) > len(simple.ops)
        assert {op.tool for op in templated.ops} >= {"cutter", "effect", "transition", "text"}

    def test_falls_back_to_the_simple_heuristic_when_no_template_scores_above_threshold(self) -> None:
        # Extreme aspect mismatch (16:9 footage vs every 9:16 template) + no
        # music + no speech/face + a single shot: every builtin template
        # scores well below the 0.3 minimum, so this must hit the fallback.
        style = _bare_style()
        features = _bare_features(aspect=16 / 9, is_vertical=False)

        plan = heuristic_plan(style, features)

        assert plan == _simple_heuristic_plan(style, features)
        assert plan.confidence == HEURISTIC_CONFIDENCE


# --- direct (full orchestration) --------------------------------------------


class TestDirect:
    def test_default_stub_llm_always_falls_back_to_heuristic(self, features, style) -> None:
        plan = direct(features, style)
        assert plan == heuristic_plan(style, features)

    def test_explicit_stub_llm_is_the_default(self, features, style) -> None:
        assert direct(features, style, llm=stub_llm) == heuristic_plan(style, features)

    def test_valid_high_confidence_llm_output_is_used_as_is(self, features, style) -> None:
        good_plan = {"ops": [{"tool": "cutter", "params": {"keep": ["s1"]}}], "confidence": 0.95}
        plan = direct(features, style, llm=lambda brief: good_plan)
        assert plan.confidence == 0.95
        assert plan.ops[0].tool == "cutter"

    def test_low_confidence_llm_output_falls_back_to_heuristic(self, features, style) -> None:
        low_confidence_plan = {"ops": [{"tool": "cutter", "params": {"keep": ["s1"]}}], "confidence": 0.05}
        plan = direct(features, style, llm=lambda brief: low_confidence_plan)
        assert plan == heuristic_plan(style, features)

    @pytest.mark.parametrize(
        "garbage",
        [
            None,
            "not json at all",
            {"ops": []},
            {"ops": [{"tool": "nonexistent_tool", "params": {}}], "confidence": 0.9},
            {"ops": [{"tool": "cutter", "params": {"bogus": True}}], "confidence": 0.9},
        ],
    )
    def test_garbage_llm_output_falls_back_to_heuristic(self, features, style, garbage) -> None:
        plan = direct(features, style, llm=lambda brief: garbage)
        assert plan == heuristic_plan(style, features)

    def test_llm_raising_falls_back_to_heuristic_instead_of_crashing(self, features, style) -> None:
        def _raises(brief: dict) -> None:
            raise RuntimeError("simulated API failure")

        plan = direct(features, style, llm=_raises)
        assert plan == heuristic_plan(style, features)

    def test_confidence_threshold_is_configurable(self, features, style) -> None:
        plan_payload = {"ops": [{"tool": "cutter", "params": {"keep": ["s1"]}}], "confidence": 0.5}
        # Below the default threshold (0.6) -> falls back...
        assert direct(features, style, llm=lambda brief: plan_payload) == heuristic_plan(style, features)
        # ...but accepted once the caller lowers the bar.
        plan = direct(features, style, llm=lambda brief: plan_payload, confidence_threshold=0.4)
        assert plan.confidence == 0.5

    def test_llm_receives_the_brief_built_from_features_and_style(self, features, style) -> None:
        seen = {}

        def _capture(brief: dict) -> dict:
            seen["brief"] = brief
            return {"ops": [], "confidence": 0.0}

        direct(features, style, llm=_capture)
        assert seen["brief"] == build_brief(features, style, TOOL_MANIFEST)
