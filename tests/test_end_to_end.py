"""End-to-end tests: `learn` and `make`, running fully with zero AI.

Per AGENTS.md's testing rule, no OTHER test in this suite touches a real
render — this file is the one deliberate exception (like `ingest.probe`,
the extractors, and `content.extract` before it, this is the seam between
real files and the rest of the JSON-only pipeline; the renderer's single
composite pass is the last such seam). Every LLM call here goes through
`director.stub_llm` (the default), so this proves build step 7's
requirement directly: the whole pipeline runs start to finish with zero AI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoedit import cli, ingest
from autoedit.executor import build_initial_timeline, execute, run as run_executor
from autoedit.models.style_profile import StyleProfile


class TestLearnEndToEnd:
    def test_learn_aggregates_three_real_reference_videos_into_a_style_profile(
        self, tmp_path, multi_shot_clip_path, face_scene_clip_path, static_caption_clip_path
    ) -> None:
        output = tmp_path / "style_profile.json"

        result_path = cli.learn(
            [multi_shot_clip_path, face_scene_clip_path, static_caption_clip_path], output
        )

        assert result_path == output
        assert output.exists()
        profile = cli.load_style_profile(output)
        assert isinstance(profile, StyleProfile)
        assert profile.sample_count == 3


class TestMakeEndToEnd:
    def test_make_runs_start_to_finish_with_zero_ai_and_produces_a_real_mp4(
        self, tmp_path, multi_shot_clip_path
    ) -> None:
        output = tmp_path / "out.mp4"

        result_path = cli.make(multi_shot_clip_path, output)

        assert result_path == output
        assert output.exists()
        assert output.stat().st_size > 0
        # Sanity-check the rendered file is itself valid, playable media.
        rendered = ingest.probe(output)
        assert rendered.duration > 0

    def test_make_with_a_learned_style_profile_still_renders(self, tmp_path, multi_shot_clip_path, face_scene_clip_path) -> None:
        style_path = tmp_path / "style.json"
        cli.learn([multi_shot_clip_path, face_scene_clip_path], style_path)

        output = tmp_path / "out.mp4"
        result_path = cli.make(multi_shot_clip_path, output, style_path=style_path)

        assert result_path == output
        assert output.exists() and output.stat().st_size > 0

    def test_make_output_matches_the_heuristic_plan_for_a_fully_neutral_style(
        self, tmp_path, multi_shot_clip_path
    ) -> None:
        """With the neutral DEFAULT_STYLE_PROFILE, the stub LLM always falls back to
        `heuristic_plan`. `_simple_heuristic_plan` (per its own rules, with no
        music/faces/text/effect signal) keeps every shot uncut -- and since `keep`
        never drops or trims a shot regardless of which plan layer wins (a
        template fill also keeps every real shot per `filler.py`), the rendered
        duration should match the source's either way."""
        from autoedit.content.extract import extract as extract_content_features
        from autoedit.director.heuristic import _simple_heuristic_plan

        features = extract_content_features(multi_shot_clip_path)
        plan = _simple_heuristic_plan(cli.DEFAULT_STYLE_PROFILE, features)
        assert [op.tool for op in plan.ops] == ["cutter"]  # no text/effect/emoji signal in a neutral profile

        output = tmp_path / "out.mp4"
        cli.make(multi_shot_clip_path, output)

        source = ingest.probe(multi_shot_clip_path)
        rendered = ingest.probe(output)
        assert rendered.duration == pytest.approx(source.duration, abs=0.2)


class TestCliMain:
    def test_learn_command_writes_a_style_profile_and_prints_a_summary(
        self, tmp_path, multi_shot_clip_path, face_scene_clip_path, static_caption_clip_path, capsys
    ) -> None:
        output = tmp_path / "style_profile.json"
        exit_code = cli.main(
            [
                "learn",
                str(multi_shot_clip_path),
                str(face_scene_clip_path),
                str(static_caption_clip_path),
                "-o",
                str(output),
            ]
        )

        assert exit_code == 0
        assert output.exists()
        captured = capsys.readouterr()
        assert "3 video(s)" in captured.out
        assert str(output) in captured.out

    def test_make_command_writes_an_mp4_and_prints_a_summary(self, tmp_path, multi_shot_clip_path, capsys) -> None:
        output = tmp_path / "out.mp4"
        exit_code = cli.main(["make", str(multi_shot_clip_path), "-o", str(output)])

        assert exit_code == 0
        assert output.exists()
        captured = capsys.readouterr()
        assert str(output) in captured.out

    def test_make_command_respects_confidence_threshold_flag(self, tmp_path, multi_shot_clip_path) -> None:
        output = tmp_path / "out.mp4"
        exit_code = cli.main(
            ["make", str(multi_shot_clip_path), "-o", str(output), "--confidence-threshold", "0.9"]
        )
        assert exit_code == 0
        assert output.exists()

    def test_make_command_with_gallery_flag_writes_a_gallery_html(self, tmp_path, multi_shot_clip_path) -> None:
        output = tmp_path / "out.mp4"
        gallery_dir = tmp_path / "gallery"

        exit_code = cli.main(["make", str(multi_shot_clip_path), "-o", str(output), "-g", str(gallery_dir)])

        assert exit_code == 0
        gallery_html = gallery_dir / "gallery.html"
        assert gallery_html.exists()
        assert output.name in gallery_html.read_text()

    def test_make_prints_review_warnings_to_stderr_for_an_over_captioned_plan(
        self, tmp_path, multi_shot_clip_path, monkeypatch, capsys
    ) -> None:
        """`make` surfaces whatever `review.check_plan` finds: force the director to
        return a heavily-captioned plan against a style that expects none."""
        from autoedit.models.plan import EditOp, EditPlan

        over_captioned_plan = EditPlan(
            ops=[
                EditOp(tool="cutter", params={"keep": ["s1", "s2", "s3"]}),
                EditOp(tool="text", params={"content": "hi", "style": "static", "start": 0.0}),
                EditOp(tool="text", params={"content": "there", "style": "static", "start": 1.0}),
            ],
            confidence=1.0,
        )
        monkeypatch.setattr("autoedit.cli.direct", lambda features, style, confidence_threshold: over_captioned_plan)

        style_path = tmp_path / "no_captions_style.json"
        no_caption_style = cli.DEFAULT_STYLE_PROFILE.model_copy(update={"caption_density": 0.0, "text_amount": 0.0})
        style_path.write_text(json.dumps(no_caption_style.model_dump()))

        output = tmp_path / "out.mp4"
        exit_code = cli.main(["make", str(multi_shot_clip_path), "-s", str(style_path), "-o", str(output)])

        assert exit_code == 0
        stderr = capsys.readouterr().err
        assert "[review]" in stderr
        assert "caption density" in stderr
