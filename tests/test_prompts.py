"""Tests for `director.prompts`: SYSTEM_PROMPT content, few-shot example validity, build_messages shape."""

from __future__ import annotations

import json

from autoedit.director.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_messages
from autoedit.director.validate import validate_plan
from autoedit.models.plan import EditPlan

_SOME_BRIEF = {
    "features": {"aspect": 0.5625, "is_vertical": True, "shots": []},
    "style": {"aspect": 0.5625, "cut_on_beat": False},
    "tools": ["cutter", "effect", "emoji", "music", "reframe", "text", "transition"],
}

_SOME_BRIEF_WITH_TEMPLATE = {
    **_SOME_BRIEF,
    "template": "punchy_beat_montage",
}

_ALL_SEVEN_TOOLS = ["cutter", "effect", "emoji", "music", "reframe", "text", "transition"]


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

    def test_mentions_all_seven_tool_names(self) -> None:
        for tool in _ALL_SEVEN_TOOLS:
            assert tool in SYSTEM_PROMPT, f"SYSTEM_PROMPT does not mention tool {tool!r}"

    def test_says_not_to_emit_music_ops(self) -> None:
        assert "music" in SYSTEM_PROMPT.lower()
        assert "do not emit" in SYSTEM_PROMPT.lower() or "not emit" in SYSTEM_PROMPT.lower()

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
            assert set(brief.keys()) == {"features", "style", "tools", "template"}
            assert isinstance(brief["tools"], list)
            assert isinstance(brief["template"], str)

    def test_each_brief_tools_list_contains_all_seven_tools(self) -> None:
        for i, example in enumerate(FEW_SHOT_EXAMPLES):
            assert sorted(example["brief"]["tools"]) == sorted(_ALL_SEVEN_TOOLS), (
                f"example {i}: brief['tools'] is missing tools or has extra ones"
            )

    def test_each_ideal_plan_validates_as_an_editplan(self) -> None:
        for example in FEW_SHOT_EXAMPLES:
            plan = EditPlan.model_validate(example["ideal_plan"])
            assert len(plan.ops) >= 1
            for op in plan.ops:
                assert op.tool in example["brief"]["tools"]

    def test_each_ideal_plan_passes_validate_plan_against_the_manifest(self) -> None:
        """validate_plan also checks tool names against TOOL_PARAMS_MANIFEST and params schemas."""
        for i, example in enumerate(FEW_SHOT_EXAMPLES):
            plan = validate_plan(example["ideal_plan"])
            assert plan.confidence > 0.5, f"example {i}: low confidence {plan.confidence}"

    def test_examples_are_diverse_in_beat_sync_usage(self) -> None:
        syncs = {
            op["params"].get("sync")
            for example in FEW_SHOT_EXAMPLES
            for op in example["ideal_plan"]["ops"]
            if op["tool"] == "cutter"
        }
        assert "beat" in syncs
        assert "none" in syncs

    def test_at_least_one_example_uses_transition(self) -> None:
        transition_tools = [
            op
            for example in FEW_SHOT_EXAMPLES
            for op in example["ideal_plan"]["ops"]
            if op["tool"] == "transition"
        ]
        assert transition_tools, "expected at least one transition op across the few-shot examples"

    def test_at_least_one_example_uses_reframe(self) -> None:
        reframe_tools = [
            op
            for example in FEW_SHOT_EXAMPLES
            for op in example["ideal_plan"]["ops"]
            if op["tool"] == "reframe"
        ]
        assert reframe_tools, "expected at least one reframe op across the few-shot examples"

    def test_no_example_emits_a_music_op(self) -> None:
        for i, example in enumerate(FEW_SHOT_EXAMPLES):
            music_ops = [op for op in example["ideal_plan"]["ops"] if op["tool"] == "music"]
            assert not music_ops, f"example {i}: ideal_plan should not emit music ops (CLI handles music)"


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

    def test_brief_with_template_name_is_included_verbatim_in_the_final_user_message(self) -> None:
        messages = build_messages(_SOME_BRIEF_WITH_TEMPLATE)
        final = json.loads(messages[-1]["content"])
        assert final.get("template") == "punchy_beat_montage"

    def test_brief_without_template_name_has_no_template_key_in_the_final_user_message(self) -> None:
        messages = build_messages(_SOME_BRIEF)
        final = json.loads(messages[-1]["content"])
        assert "template" not in final
