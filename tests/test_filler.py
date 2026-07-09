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


def _shot(shot_id: str, *, motion: float = 0.1, faces: int = 0, scale: str = "wide", dur: float = 2.0) -> dict:
    return {
        "id": shot_id,
        "source": "raw.mp4",
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

    def test_text_ops_use_template_text_slot_placeholders(self) -> None:
        template = TEMPLATE_REGISTRY["talking_head_with_broll"]  # TITLE (top, static), CTA (bottom, static)
        features = _features(
            has_speech=True, has_face=True,
            shots=[_shot("s1", faces=1, scale="close", dur=8.0), _shot("s2", faces=1, scale="close", dur=6.0)],
        )

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        text_ops = [op for op in plan.ops if op.tool == "text"]
        assert {op.params["content"] for op in text_ops} == {"TITLE", "CTA"}
        assert all(op.params["style"] == "static" for op in text_ops)

    def test_no_text_ops_when_the_template_has_no_text_slots(self) -> None:
        template = TEMPLATE_REGISTRY["quick_montage"]
        features = _features(music_bpm=120.0, shots=[_shot(f"s{i}", dur=1.0) for i in range(1, 9)])

        plan = fill_template(template, features, _NEUTRAL_STYLE)

        assert not any(op.tool == "text" for op in plan.ops)


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
