"""Tests for autoedit/director/eval.py.

Pure fixture-JSON tests (per AGENTS.md's testing rule): `run_eval` calls a
plain `LLMClient` callable directly, never a real model/network/media.
"""

from __future__ import annotations

import json

import pytest

from autoedit.director.eval import EvalCase, load_eval_cases, run_eval
from autoedit.director.llm import stub_llm
from autoedit.models.style_profile import CaptionStyleFreq, StyleProfile


def _style(**overrides) -> StyleProfile:
    base = dict(
        aspect=0.5625, shot_len_median=1.0, shot_len_spread=0.2, cut_on_beat=True,
        caption_style_freq=CaptionStyleFreq(karaoke=0.5, static=0.5),
        caption_density=0.0, text_amount=0.0, effect_freq=0.0, sample_count=1,
    )
    base.update(overrides)
    return StyleProfile.model_validate(base)


def _case(name: str, content_features_data: dict, **style_overrides) -> EvalCase:
    return EvalCase.model_validate({"name": name, "features": content_features_data, "style": _style(**style_overrides).model_dump()})


class TestRunEvalWithTheStub:
    def test_the_stub_llm_is_always_invalid_never_falls_back(self, content_features_data):
        """`run_eval` must NOT silently swap in `heuristic_plan` the way `direct()` does --
        that's the entire point of a harness that measures the raw model."""
        cases = [_case("c1", content_features_data)]
        report = run_eval(cases, stub_llm)

        assert report.total_count == 1
        assert report.valid_count == 0
        assert report.cases[0].valid is False
        assert report.cases[0].error is not None
        assert report.cases[0].confidence is None


class TestRunEvalWithAWorkingLlm:
    def _good_llm(self, brief):
        shots = [s["id"] for s in brief["features"]["shots"]]
        return {"ops": [{"tool": "cutter", "params": {"keep": shots}}], "confidence": 0.9}

    def test_valid_response_is_scored_and_reviewed(self, content_features_data):
        # fixture's 2 shots are 2.5s each; match the style to that pacing so only the
        # thing under test (valid/confidence/review-pass wiring) is exercised.
        cases = [_case("c1", content_features_data, shot_len_median=2.5, shot_len_spread=1.0, caption_density=0.0, text_amount=0.0)]
        report = run_eval(cases, self._good_llm)

        result = report.cases[0]
        assert result.valid is True
        assert result.confidence == pytest.approx(0.9)
        assert result.review_passed is True
        assert result.error is None

    def test_valid_rate_and_review_passed_count_aggregate_correctly(self, content_features_data):
        cases = [_case("a", content_features_data), _case("b", content_features_data)]
        report = run_eval(cases, self._good_llm)

        assert report.total_count == 2
        assert report.valid_count == 2
        assert report.valid_rate == pytest.approx(1.0)

    def test_empty_case_list_yields_a_zero_rate_without_a_crash(self):
        report = run_eval([], self._good_llm)
        assert report.total_count == 0
        assert report.valid_rate == 0.0


class TestRunEvalHandlesFailures:
    def test_llm_call_raising_is_recorded_not_propagated(self, content_features_data):
        def broken_llm(brief):
            raise RuntimeError("boom")

        cases = [_case("c1", content_features_data)]
        report = run_eval(cases, broken_llm)

        assert report.cases[0].valid is False
        assert "boom" in report.cases[0].error

    def test_an_unknown_tool_name_is_recorded_as_invalid(self, content_features_data):
        def bad_tool_llm(brief):
            return {"ops": [{"tool": "not_a_real_tool", "params": {}}], "confidence": 0.9}

        cases = [_case("c1", content_features_data)]
        report = run_eval(cases, bad_tool_llm)

        assert report.cases[0].valid is False
        assert "not_a_real_tool" in report.cases[0].error


class TestLoadEvalCases:
    def test_loads_every_json_file_in_the_directory(self):
        cases = load_eval_cases("fixtures/eval_cases")
        assert len(cases) == 3
        assert {case.name for case in cases} == {
            "hook_montage_high_energy", "talking_head_low_energy", "quick_montage_no_faces",
        }

    def test_raises_on_an_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="no \\*.json"):
            load_eval_cases(tmp_path)

    def test_each_loaded_case_directs_cleanly_through_run_eval(self):
        cases = load_eval_cases("fixtures/eval_cases")
        report = run_eval(cases, stub_llm)
        assert report.total_count == 3
        assert all(not case.valid for case in report.cases)  # the stub is always invalid
