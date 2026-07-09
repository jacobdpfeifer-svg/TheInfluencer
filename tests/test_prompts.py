"""Tests for `director.prompts`: SYSTEM_PROMPT content, few-shot example validity, build_messages shape."""

from __future__ import annotations

import json

from autoedit.director.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_messages
from autoedit.models.plan import EditPlan

_SOME_BRIEF = {
    "features": {"aspect": 0.5625, "is_vertical": True, "shots": []},
    "style": {"aspect": 0.5625, "cut_on_beat": False},
    "tools": ["cutter", "effect", "emoji", "text"],
}


class TestSystemPrompt:
    def test_mentions_hook_beat_and_retention(self) -> None:
        lowered = SYSTEM_PROMPT.lower()
        assert "hook" in lowered
        assert "beat" in lowered
        assert "retention" in lowered

    def test_mentions_the_remaining_editing_principles(self) -> None:
        lowered = SYSTEM_PROMPT.lower()
        assert "pattern interrupt" in lowered
        assert "energy" in lowered
        assert "caption" in lowered
        assert "vertical" in lowered or "9:16" in lowered
        assert "15" in lowered and "60" in lowered
        assert "less is more" in lowered

    def test_is_a_nonempty_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100


class TestFewShotExamples:
    def test_has_two_or_three_examples(self) -> None:
        assert 2 <= len(FEW_SHOT_EXAMPLES) <= 3

    def test_each_example_has_brief_ideal_plan_and_reasoning(self) -> None:
        for example in FEW_SHOT_EXAMPLES:
            assert "brief" in example
            assert "ideal_plan" in example
            assert "reasoning" in example
            assert isinstance(example["reasoning"], str)
            assert example["reasoning"].strip() != ""

    def test_each_brief_is_json_serializable_and_shaped_like_build_brief_output(self) -> None:
        for example in FEW_SHOT_EXAMPLES:
            brief = example["brief"]
            json.dumps(brief)  # must not raise
            assert set(brief.keys()) == {"features", "style", "tools"}
            assert isinstance(brief["tools"], list)

    def test_each_ideal_plan_validates_as_an_editplan(self) -> None:
        for example in FEW_SHOT_EXAMPLES:
            plan = EditPlan.model_validate(example["ideal_plan"])
            assert len(plan.ops) >= 1
            for op in plan.ops:
                assert op.tool in example["brief"]["tools"]

    def test_examples_are_diverse_in_beat_sync_usage(self) -> None:
        syncs = {
            op["params"].get("sync")
            for example in FEW_SHOT_EXAMPLES
            for op in example["ideal_plan"]["ops"]
            if op["tool"] == "cutter"
        }
        assert "beat" in syncs
        assert "none" in syncs


class TestBuildMessages:
    def test_starts_with_the_system_prompt(self) -> None:
        messages = build_messages(_SOME_BRIEF)
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}

    def test_ends_with_the_real_brief_as_the_final_user_message(self) -> None:
        messages = build_messages(_SOME_BRIEF)
        assert messages[-1]["role"] == "user"
        assert json.loads(messages[-1]["content"]) == _SOME_BRIEF

    def test_includes_one_user_assistant_pair_per_few_shot_example(self) -> None:
        messages = build_messages(_SOME_BRIEF)
        assert len(messages) == 1 + 2 * len(FEW_SHOT_EXAMPLES) + 1

    def test_each_few_shot_pair_round_trips_its_example_data_without_the_reasoning_field(self) -> None:
        messages = build_messages(_SOME_BRIEF)
        for i, example in enumerate(FEW_SHOT_EXAMPLES):
            user_msg = messages[1 + 2 * i]
            assistant_msg = messages[2 + 2 * i]

            assert user_msg["role"] == "user"
            assert json.loads(user_msg["content"]) == example["brief"]

            assert assistant_msg["role"] == "assistant"
            assert json.loads(assistant_msg["content"]) == example["ideal_plan"]
            assert "reasoning" not in json.loads(assistant_msg["content"])

    def test_every_message_content_is_a_plain_string(self) -> None:
        for message in build_messages(_SOME_BRIEF):
            assert isinstance(message["content"], str)
            assert message["role"] in {"system", "user", "assistant"}

    def test_different_briefs_produce_different_final_messages(self) -> None:
        other_brief = {**_SOME_BRIEF, "tools": ["cutter"]}
        messages_a = build_messages(_SOME_BRIEF)
        messages_b = build_messages(other_brief)
        assert messages_a[-1] != messages_b[-1]
        assert messages_a[:-1] == messages_b[:-1]
