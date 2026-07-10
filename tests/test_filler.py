"""Tests for autoedit/templates/filler.py.

Pure fixture-JSON tests (per AGENTS.md's testing rule): `fill_template`
never touches media or a Timeline, only ContentFeatures/StyleProfile, and
never renders. Its output is validated with the same `validate_plan` a real
model's output would face.
"""

from __future__ import annotations

import pytest

from autoedit.director.validate import validate_plan
from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditPlan
from autoedit.models.shot import Shot
from autoedit.models.style_profile import StyleProfile
from autoedit.models.template import Template
from autoedit.templates import TEMPLATE_REGISTRY
from autoedit.templates.filler import TEMPLATE_FILL_CONFIDENCE, fill_template

_NEUTRAL_STYLE = StyleProfile.model_validate(
    {
        "aspect": 0.5625,
        "shot_len_median": 2.0,
        "shot_len_spread": 0.5,
        "cut_on_beat": False,
        "caption_style_freq": {"karaoke": 0.5, "static": 0.5},
        "caption_density": 0.0,
        "text_amount": 0.0,
        "effect_freq": 0.0,
        "sample_count": 1,
    }
)


def _shot(
    shot_id: str, *, motion: float = 0.1, faces: int = 0, scale: str = "wide", dur: float = 2.0, source_override: str = "raw.mp4"
) -> dict:
    return {
        "id": shot_id,
        "source": source_override,
        "in": 0.0,
        "out": dur,
        "dur": dur,
        "motion": motion,
        "brightness": 0.5,
        "sharpness": 50.0,
        "faces": faces,
        "scale": scale,
    }


def _features(**overrides) -> ContentFeatures:
    base = dict(
        aspect=0.5625,
        has_speech=False,
        music_bpm=None,
        shots=[_shot("s1")],
        motion="low",
        is_vertical=True,
        has_face=False,
    )
    base.update(overrides)
    return ContentFeatures.model_validate(base)


class TestFillTemplateBasics:
    def test_returns_a_valid_editplan(self) -> None:
        template = TEMPLATE_REGISTRY["punchy_beat_montage"]
        features = _features(
            music_bpm=128.0,
            shots=[_shot(f"s{i}", motion=0.7, dur=1.0) for i in range(1, 6)],
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        assert isinstance(plan, EditPlan)
        validate_plan(plan.model_dump())

    def test_confidence_is_the_fixed_template_fill_baseline(self) -> None:
        template = TEMPLATE_REGISTRY["quick_montage"]
        features = _features(music_bpm=120.0, shots=[_shot(f"s{i}", dur=1.0) for i in range(1, 9)])

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        assert plan.confidence == TEMPLATE_FILL_CONFIDENCE

    def test_cutter_op_keep_count_matches_the_smaller_of_slots_and_shots(self) -> None:
        template = TEMPLATE_REGISTRY["talking_head_with_broll"]  # 3 slots
        features = _features(
            has_speech=True, has_face=True, shots=[_shot("s1", faces=1, scale="close", dur=8.0)]
        )  # 1 shot

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert len(cutter_op.params["keep"]) == 1  # min(3 slots, 1 shot)

    def test_cutter_op_keep_count_matches_slot_count_when_more_shots_than_slots(self) -> None:
        template = TEMPLATE_REGISTRY["talking_head_with_broll"]  # 3 slots
        features = _features(shots=[_shot(f"s{i}", dur=1.0) for i in range(1, 6)])  # 5 shots

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert len(cutter_op.params["keep"]) == 3  # min(3 slots, 5 shots)

    def test_cutter_sync_follows_the_templates_music_cut_on(self) -> None:
        beat_synced = TEMPLATE_REGISTRY["quick_montage"]
        features = _features(music_bpm=120.0, beat_times=[0.5, 1.0], shots=[_shot("s1")])

        plan = fill_template(beat_synced, features, _NEUTRAL_STYLE)

        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert cutter_op.params["sync"] == "beat"
        assert cutter_op.params["beat_times"] == [0.5, 1.0]

    def test_no_trim_when_every_assigned_shot_is_within_its_slots_budget(self) -> None:
        template = TEMPLATE_REGISTRY["talking_head_with_broll"]  # min/max_duration 5.0/15.0
        features = _features(has_speech=True, has_face=True, shots=[_shot("s1", faces=1, scale="close", dur=8.0)])

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        assert cutter_op.params.get("trim", {}) == {}

    def test_effect_ops_match_the_templates_slot_effects(self) -> None:
        template = TEMPLATE_REGISTRY["punchy_beat_montage"]  # hook=zoom_in, payoff=zoom_out
        features = _features(
            music_bpm=128.0,
            shots=[_shot(f"s{i}", motion=0.7, dur=1.0) for i in range(1, 6)],
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        effect_kinds = sorted(op.params["kind"] for op in plan.ops if op.tool == "effect")
        assert effect_kinds == ["zoom_in", "zoom_out"]

    def test_no_effect_ops_when_the_template_has_none(self) -> None:
        template = TEMPLATE_REGISTRY["talking_head_with_broll"]
        features = _features(has_speech=True, has_face=True, shots=[_shot("s1", faces=1, scale="close", dur=8.0)])

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        assert not any(op.tool == "effect" for op in plan.ops)

    def test_transition_op_emitted_between_hook_and_first_b_roll(self) -> None:
        template = TEMPLATE_REGISTRY["punchy_beat_montage"]
        features = _features(
            music_bpm=128.0,
            shots=[_shot(f"s{i}", motion=0.7, dur=1.0) for i in range(1, 6)],
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        transition_ops = [op for op in plan.ops if op.tool == "transition"]
        assert len(transition_ops) == 1
        assert transition_ops[0].params["kind"] == "whip_pan"
        assert len(transition_ops[0].params["between"]) == 2

    def test_text_ops_resolve_placeholder_tags_into_footage_derived_copy(self) -> None:
        from autoedit.templates.captions import generate_caption_copy

        template = TEMPLATE_REGISTRY["talking_head_with_broll"]  # TITLE (top, static), CTA (bottom, static)
        features = _features(
            has_speech=True, has_face=True,
            shots=[_shot("s1", faces=1, scale="close", dur=8.0), _shot("s2", faces=1, scale="close", dur=6.0)],
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        text_ops = [op for op in plan.ops if op.tool == "text"]
        expected = {generate_caption_copy(features, "TITLE"), generate_caption_copy(features, "CTA")}
        assert {op.params["content"] for op in text_ops} == expected
        assert expected != {"TITLE", "CTA"}  # sanity: the tags really did get resolved, not passed through
        assert all(op.params["style"] == "static" for op in text_ops)

    def test_no_text_ops_when_the_template_has_no_text_slots(self) -> None:
        template = TEMPLATE_REGISTRY["quick_montage"]
        features = _features(music_bpm=120.0, shots=[_shot(f"s{i}", dur=1.0) for i in range(1, 9)])

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        assert not any(op.tool == "text" for op in plan.ops)

    def test_reframe_ops_emitted_for_slots_with_a_transform(self) -> None:
        template = TEMPLATE_REGISTRY["talking_head_with_broll"]  # talking_head slots -> center_crop
        features = _features(
            has_speech=True, has_face=True,
            shots=[_shot("s1", faces=1, scale="close", dur=8.0), _shot("s2", faces=1, scale="close", dur=6.0)],
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        reframe_ops = [op for op in plan.ops if op.tool == "reframe"]
        assert len(reframe_ops) == 2  # both talking_head slots have transform="center_crop"
        assert all(op.params["kind"] == "center_crop" for op in reframe_ops)
        assert all(op.params["target_aspect"] == pytest.approx(template.target_aspect) for op in reframe_ops)

    def test_no_reframe_op_for_slots_with_transform_none(self) -> None:
        template = TEMPLATE_REGISTRY["punchy_beat_montage"]  # every slot has transform="none"
        features = _features(
            music_bpm=128.0,
            shots=[_shot(f"s{i}", motion=0.7, dur=1.0) for i in range(1, 6)],
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        assert not any(op.tool == "reframe" for op in plan.ops)

    def test_reframe_op_params_validate_against_reframeparams(self) -> None:
        from autoedit.subsystems.reframe import ReframeParams

        template = TEMPLATE_REGISTRY["talking_head_with_broll"]
        features = _features(
            has_speech=True, has_face=True, shots=[_shot("s1", faces=1, scale="close", dur=8.0)]
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        reframe_op = next(op for op in plan.ops if op.tool == "reframe")
        ReframeParams.model_validate(reframe_op.params)  # must not raise


class TestDurationBudgeting:
    """`TemplateSlot.duration`/`max_duration` become `cutter` `trim` caps;
    `min_duration` is a soft `_slot_shot_score` penalty (see filler.py's
    module docstring for why the two need different treatment).
    """

    def test_slot_duration_cap_is_none_when_shot_is_within_max_duration(self) -> None:
        from autoedit.templates.filler import _slot_duration_cap

        slot = next(s for s in TEMPLATE_REGISTRY["punchy_beat_montage"].slots if s.id == "hook")  # max_duration 3.0
        shot = Shot.model_validate(_shot("s1", dur=2.0))
        assert _slot_duration_cap(slot, shot) is None

    def test_slot_duration_cap_is_max_duration_when_shot_exceeds_it(self) -> None:
        from autoedit.templates.filler import _slot_duration_cap

        slot = next(s for s in TEMPLATE_REGISTRY["punchy_beat_montage"].slots if s.id == "hook")  # max_duration 3.0
        shot = Shot.model_validate(_shot("s1", dur=8.0))
        assert _slot_duration_cap(slot, shot) == pytest.approx(3.0)

    def test_slot_duration_cap_prefers_an_exact_duration_target_over_max_duration(self) -> None:
        from autoedit.templates.filler import _slot_duration_cap

        slot = TEMPLATE_REGISTRY["punchy_beat_montage"].slots[0].model_copy(update={"duration": 1.5, "max_duration": 3.0})
        shot = Shot.model_validate(_shot("s1", dur=8.0))
        assert _slot_duration_cap(slot, shot) == pytest.approx(1.5)

    def test_cutter_trim_is_emitted_end_to_end_for_a_shot_that_blows_the_hook_budget(self) -> None:
        template = TEMPLATE_REGISTRY["punchy_beat_montage"]  # hook: max_duration 3.0
        features = _features(
            music_bpm=128.0,
            shots=[_shot("s1", motion=0.9, faces=1, dur=8.0)] + [_shot(f"s{i}", motion=0.2, dur=1.0) for i in range(2, 6)],
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        # s1 (highest motion + a face) wins the "hook" slot and gets capped to its 3.0s budget.
        assert cutter_op.params["keep"][0] == "s1"
        assert cutter_op.params["trim"] == {"s1": pytest.approx(3.0)}

    def test_short_shot_score_is_penalized_but_never_drops_to_the_dummy_floor(self) -> None:
        from autoedit.templates.filler import _BASE_SCORE, _slot_shot_score

        slot = next(s for s in TEMPLATE_REGISTRY["talking_head_with_broll"].slots if s.id == "talking_head_1")
        short_shot = Shot.model_validate(_shot("s1", faces=1, scale="close", dur=1.0))  # min_duration is 5.0
        long_shot = Shot.model_validate(_shot("s2", faces=1, scale="close", dur=8.0))

        assert _slot_shot_score(slot, short_shot) < _slot_shot_score(slot, long_shot)
        assert _slot_shot_score(slot, short_shot) >= _BASE_SCORE  # never at/below the dummy-slot floor


class TestHungarianAssignmentNeverDuplicatesAShot:
    @pytest.mark.parametrize("shot_count", [1, 2, 3, 5, 8, 10])
    def test_every_kept_shot_id_is_unique_regardless_of_slot_vs_shot_count(self, shot_count: int) -> None:
        for template in TEMPLATE_REGISTRY.values():
            features = _features(
                music_bpm=128.0 if template.music.required else None,
                shots=[_shot(f"s{i}", motion=0.5, faces=i % 2, scale="close" if i % 2 else "wide", dur=1.0) for i in range(1, shot_count + 1)],
            )

            plan = fill_template(template, features, _NEUTRAL_STYLE)

            cutter_op = next(op for op in plan.ops if op.tool == "cutter")
            keep = cutter_op.params["keep"]
            assert len(keep) == len(set(keep)), f"{template.name} with {shot_count} shots assigned a shot twice"

    def test_assignment_helper_never_pairs_two_slots_with_the_same_shot(self) -> None:
        from autoedit.templates.filler import _assign_shots_to_slots

        template: Template = TEMPLATE_REGISTRY["quick_montage"]  # 8 slots
        shots = [_shot(f"s{i}", dur=1.0) for i in range(1, 4)]  # only 3 real shots
        features = _features(shots=shots)

        pairs = _assign_shots_to_slots(template.slots, features.shots)

        assigned_shot_ids = [shot.id for _slot, shot in pairs]
        assert len(assigned_shot_ids) == len(set(assigned_shot_ids))
        assert len(pairs) == 3  # min(8 slots, 3 shots) -- the rest are dropped, not duplicated


class TestFillTemplateAcrossMultipleSources:
    """Filler is source-agnostic: pooled footage (shots from several clips, per
    `content.extract_pool`) is assigned to slots exactly like single-source
    shots, and each kept shot's own `source` rides through untouched.
    """

    def _multi_source_features(self) -> ContentFeatures:
        # 5 shots drawn from 3 different source clips (a montage pool).
        shots = [
            _shot("s1", source_override="clip_a.mp4", motion=0.8, dur=1.0),
            _shot("s2", source_override="clip_a.mp4", motion=0.3, dur=1.0),
            _shot("s3", source_override="clip_b.mp4", motion=0.6, dur=1.0),
            _shot("s4", source_override="clip_c.mp4", motion=0.7, dur=1.0),
            _shot("s5", source_override="clip_c.mp4", motion=0.9, dur=1.0),
        ]
        return _features(music_bpm=128.0, motion="high", shots=shots)

    def test_keeps_shots_from_multiple_sources_without_collapsing_them(self) -> None:
        template = TEMPLATE_REGISTRY["punchy_beat_montage"]  # 5 slots
        features = self._multi_source_features()

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        keep = cutter_op.params["keep"]
        assert len(keep) == 5
        assert len(set(keep)) == 5  # no shot assigned twice, even across sources

    def test_kept_ids_are_a_subset_of_the_pooled_shot_ids(self) -> None:
        template = TEMPLATE_REGISTRY["talking_head_with_broll"]  # 3 slots, 5 shots -> drop 2
        features = self._multi_source_features()

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        cutter_op = next(op for op in plan.ops if op.tool == "cutter")
        pooled_ids = {shot.id for shot in features.shots}
        assert set(cutter_op.params["keep"]) <= pooled_ids
        assert len(cutter_op.params["keep"]) == 3
