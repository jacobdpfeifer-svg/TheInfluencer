"""Tests for autoedit/review.py.

Text-only, per AGENTS.md's testing rule: `check_plan` and the gallery
builder only ever touch EditPlan/StyleProfile/ContentFeatures JSON and
plain strings — no media, no Timeline, no render.
"""

from __future__ import annotations

import json

import pytest

from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditOp, EditPlan
from autoedit.models.style_profile import CaptionStyleFreq, StyleProfile
from autoedit.review import GalleryEntry, ReviewResult, append_gallery_entry, build_gallery, check_plan


def _style(**overrides) -> StyleProfile:
    base = dict(
        aspect=0.5625,
        shot_len_median=2.0,
        shot_len_spread=0.5,
        cut_on_beat=False,
        caption_style_freq={"karaoke": 0.5, "static": 0.5},
        caption_density=0.3,
        text_amount=0.3,
        effect_freq=0.0,
        sample_count=5,
    )
    base.update(overrides)
    return StyleProfile.model_validate(base)


def _plan(*ops: EditOp, confidence: float = 0.8) -> EditPlan:
    return EditPlan(ops=list(ops), confidence=confidence)


def _cutter(keep: list[str]) -> EditOp:
    return EditOp(tool="cutter", params={"keep": keep})


def _text(start: float, duration: float = 2.0, content: str = "hi") -> EditOp:
    return EditOp(tool="text", params={"content": content, "style": "static", "start": start, "duration": duration})


@pytest.fixture
def features(content_features_data: dict) -> ContentFeatures:
    return ContentFeatures.model_validate(content_features_data)


# --- check_plan: the over-captioned case (the main ask) --------------------


class TestCheckPlanFlagsOverCaptioning:
    def test_flags_a_plan_with_far_more_captions_than_the_learned_density(self) -> None:
        style = _style(caption_density=0.2, text_amount=0.2)
        # 6s of footage (3 shots * fallback 2s estimate), but 8 captions covering 16s of text.
        plan = _plan(_cutter(["s1", "s2", "s3"]), *[_text(start=i * 2.0) for i in range(8)])

        result = check_plan(plan, style)

        assert result.passed is False
        assert any("caption density" in warning for warning in result.warnings)
        assert any("text amount" in warning for warning in result.warnings)

    def test_flags_captions_at_all_when_the_learned_style_never_captions(self) -> None:
        style = _style(caption_density=0.0, text_amount=0.0)
        plan = _plan(_cutter(["s1", "s2"]), _text(start=0.0))

        result = check_plan(plan, style)

        assert result.passed is False
        assert any("caption density" in warning for warning in result.warnings)

    def test_over_captioned_plan_flagged_using_real_shot_durations(self, features) -> None:
        """Same check, but with `features` supplied for accurate (non-estimated) durations."""
        style = _style(caption_density=0.2, text_amount=0.2)
        # features fixture: 2 shots totaling 5.0s (s1: 0-2.5, s2: 2.5-5.0).
        plan = _plan(_cutter(["s1", "s2"]), *[_text(start=i * 0.5, duration=0.5) for i in range(6)])

        result = check_plan(plan, style, features=features)

        assert result.passed is False
        assert result.metrics["duration_is_estimated"] is False
        assert result.metrics["duration"] == pytest.approx(5.0)


class TestCheckPlanPasses:
    def test_passes_a_plan_whose_captioning_matches_the_learned_style(self) -> None:
        style = _style(caption_density=0.25, text_amount=0.5)
        # 3 shots * 2s fallback estimate = 6s; 1 caption of 2s -> density ~0.17/s, amount ~0.33.
        plan = _plan(_cutter(["s1", "s2", "s3"]), _text(start=0.0, duration=2.0))

        result = check_plan(plan, style)

        assert result.passed is True
        assert result.warnings == []

    def test_passes_a_cutter_only_plan_with_no_captions_when_style_expects_none(self) -> None:
        style = _style(caption_density=0.0, text_amount=0.0)
        plan = _plan(_cutter(["s1", "s2"]))

        result = check_plan(plan, style)

        assert result.passed is True


class TestCheckPlanCutPacing:
    def test_flags_implied_shot_length_far_outside_the_learned_pacing(self, features) -> None:
        # features: 2 shots of 2.5s each -> keeping both implies a 2.5s average shot.
        style = _style(shot_len_median=0.3, shot_len_spread=0.05)  # a "fast cuts" style
        plan = _plan(_cutter(["s1", "s2"]))

        result = check_plan(plan, style, features=features)

        assert result.passed is False
        assert any("implied average shot length" in warning for warning in result.warnings)

    def test_does_not_flag_pacing_within_the_learned_spread(self, features) -> None:
        style = _style(shot_len_median=2.5, shot_len_spread=1.0)
        plan = _plan(_cutter(["s1", "s2"]))

        result = check_plan(plan, style, features=features)

        assert not any("implied average shot length" in warning for warning in result.warnings)


class TestCheckPlanDurationBudget:
    def test_flags_a_near_empty_plan_as_too_short(self) -> None:
        style = _style()
        plan = _plan(_cutter(["s1"]))  # 1 shot * 2s fallback estimate = 2s, below the 3s floor

        result = check_plan(plan, style)

        assert result.passed is False
        assert any("total duration" in warning for warning in result.warnings)

    def test_flags_a_runaway_plan_as_too_long(self) -> None:
        style = _style()
        plan = _plan(_cutter([f"s{i}" for i in range(1, 60)]))  # 59 shots * 2s = 118s, over the 90s ceiling

        result = check_plan(plan, style)

        assert result.passed is False
        assert any("total duration" in warning for warning in result.warnings)

    def test_does_not_flag_a_plan_within_the_short_form_budget(self) -> None:
        style = _style()
        plan = _plan(_cutter(["s1", "s2", "s3"]))  # 3 * 2s = 6s

        result = check_plan(plan, style)

        assert not any("total duration" in warning for warning in result.warnings)

    def test_flags_and_notes_estimation_when_features_are_not_supplied(self) -> None:
        style = _style()
        plan = _plan(_cutter(["s1"]))

        result = check_plan(plan, style)

        assert any("estimated durations" in warning for warning in result.warnings)

    def test_uses_real_durations_when_features_are_supplied(self, features) -> None:
        # features fixture totals 5.0s for s1+s2 -- comfortably within budget, no estimation note.
        style = _style()
        plan = _plan(_cutter(["s1", "s2"]))

        result = check_plan(plan, style, features=features)

        assert not any("total duration" in warning for warning in result.warnings)
        assert result.metrics["duration_is_estimated"] is False


class TestCheckPlanMetricsAndEdgeCases:
    def test_metrics_include_cut_count_and_shot_count(self) -> None:
        result = check_plan(_plan(_cutter(["s1", "s2", "s3"])), _style())
        assert result.metrics["shot_count"] == 3
        assert result.metrics["cut_count"] == 2

    def test_no_cutter_op_yields_zero_shot_count_and_no_crash(self) -> None:
        result = check_plan(_plan(_text(start=0.0)), _style())
        assert result.metrics["shot_count"] == 0
        assert isinstance(result, ReviewResult)

    def test_result_is_json_serializable(self) -> None:
        result = check_plan(_plan(_cutter(["s1"]), _text(start=0.0)), _style())
        json.dumps(result.model_dump())


# --- gallery -----------------------------------------------------------------


class TestBuildGallery:
    def test_empty_gallery_renders_a_placeholder(self) -> None:
        html_output = build_gallery([])
        assert "No videos yet" in html_output

    def test_renders_one_card_per_entry_with_pass_and_warn_badges(self) -> None:
        entries = [
            GalleryEntry(
                video_path="raw/a.mp4",
                output_path="out/a.mp4",
                confidence=0.9,
                review=ReviewResult(passed=True, warnings=[], metrics={}),
            ),
            GalleryEntry(
                video_path="raw/b.mp4",
                output_path="out/b.mp4",
                confidence=0.5,
                review=ReviewResult(passed=False, warnings=["caption density too high"], metrics={}),
            ),
        ]

        html_output = build_gallery(entries)

        assert "a.mp4" in html_output and "b.mp4" in html_output
        assert "PASS" in html_output
        assert "WARN" in html_output
        assert "caption density too high" in html_output

    def test_escapes_untrusted_content(self) -> None:
        entry = GalleryEntry(
            video_path="<script>alert(1)</script>",
            output_path="out.mp4",
            confidence=0.5,
            review=ReviewResult(passed=False, warnings=["<b>bad</b>"], metrics={}),
        )
        html_output = build_gallery([entry])
        assert "<script>" not in html_output
        assert "&lt;script&gt;" in html_output


class TestAppendGalleryEntry:
    def test_creates_manifest_and_html_and_persists_across_calls(self, tmp_path) -> None:
        gallery_dir = tmp_path / "gallery"
        entry1 = GalleryEntry(
            video_path="a.mp4", output_path="out_a.mp4", confidence=0.9,
            review=ReviewResult(passed=True, warnings=[], metrics={}),
        )
        entry2 = GalleryEntry(
            video_path="b.mp4", output_path="out_b.mp4", confidence=0.4,
            review=ReviewResult(passed=False, warnings=["over-captioned"], metrics={}),
        )

        append_gallery_entry(entry1, gallery_dir)
        html_path = append_gallery_entry(entry2, gallery_dir)

        assert html_path == gallery_dir / "gallery.html"
        html_content = html_path.read_text()
        assert "out_a.mp4" in html_content
        assert "out_b.mp4" in html_content

        manifest = json.loads((gallery_dir / "gallery.json").read_text())
        assert len(manifest) == 2
