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

    def test_make_pours_a_pool_of_clips_into_one_rendered_mp4(
        self, tmp_path, multi_shot_clip_path, face_scene_clip_path, static_caption_clip_path
    ) -> None:
        """Multi-clip `make`: several raw clips -> `content.extract_pool` -> one edit.

        Proves the whole pooled path runs start to finish with zero AI and
        produces a real, playable mp4 (the renderer already loads each
        segment from its own `source`, so multi-source timelines just work)."""
        output = tmp_path / "out.mp4"

        result_path = cli.make(
            [multi_shot_clip_path, face_scene_clip_path, static_caption_clip_path], output
        )

        assert result_path == output
        assert output.exists() and output.stat().st_size > 0
        rendered = ingest.probe(output)
        assert rendered.duration > 0

    def test_make_with_a_learned_style_profile_still_renders(self, tmp_path, multi_shot_clip_path, face_scene_clip_path) -> None:
        style_path = tmp_path / "style.json"
        cli.learn([multi_shot_clip_path, face_scene_clip_path], style_path)

        output = tmp_path / "out.mp4"
        result_path = cli.make(multi_shot_clip_path, output, style_path=style_path)

        assert result_path == output
        assert output.exists() and output.stat().st_size > 0

    def test_make_with_a_music_path_overrides_beat_times_and_bpm_before_directing(
        self, tmp_path, multi_shot_clip_path, click_track_path, monkeypatch
    ) -> None:
        """`music_path`'s REAL beat grid (from `click_track_path`, ~117 BPM) must reach the
        director/template instead of whatever the (silent) `multi_shot_clip_path` fixture measured.
        """
        seen_features = {}

        import autoedit.cli as cli_module

        real_direct = cli_module.direct

        def spying_direct(features, style, **kwargs):
            seen_features["features"] = features
            return real_direct(features, style, **kwargs)

        monkeypatch.setattr(cli_module, "direct", spying_direct)

        output = tmp_path / "out.mp4"
        cli.make(multi_shot_clip_path, output, music_path=click_track_path)

        features = seen_features["features"]
        assert features.beat_times  # non-empty, real beats from the music file
        assert features.music_bpm is not None and features.music_bpm > 0
        assert output.exists() and output.stat().st_size > 0

    def test_make_with_a_music_path_mixes_a_real_audio_track_into_the_output(
        self, tmp_path, multi_shot_clip_path, click_track_path
    ) -> None:
        output = tmp_path / "out.mp4"
        cli.make(multi_shot_clip_path, output, music_path=click_track_path)

        probe = ingest.probe(output)
        assert probe.audio_channels is not None and probe.audio_channels > 0

    def test_make_with_music_flag_wires_beat_times_through_to_the_plan(
        self, tmp_path, multi_shot_clip_path, click_track_path
    ) -> None:
        """--music overrides beat_times + music_bpm on features before directing.

        Uses dry_run=True (no render) with the default style. The multi-shot
        fixture matches `punchy_beat_montage` (which prefers music and uses
        beat sync), so when the music file's real beat grid arrives on
        `features.beat_times`, the cutter op in the output plan carries
        `sync='beat'` and a non-empty `beat_times` list.
        """
        output = tmp_path / "plan.json"
        result = cli.make(multi_shot_clip_path, output, music_path=click_track_path, dry_run=True)

        assert result == output
        assert output.exists() and output.stat().st_size > 0

        data = json.loads(output.read_text())
        cutter_ops = [op for op in data["plan"]["ops"] if op["tool"] == "cutter"]
        assert cutter_ops, "expected at least one cutter op in the dry-run plan"
        params = cutter_ops[0]["params"]
        assert params.get("sync") == "beat", (
            "cutter should use sync='beat' when music_path is given and the matched "
            "template prefers beat-synced cuts"
        )
        assert params.get("beat_times"), (
            "cutter params should carry non-empty beat_times from the music file"
        )

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

    def test_make_command_respects_music_flag(self, tmp_path, multi_shot_clip_path, click_track_path) -> None:
        output = tmp_path / "out.mp4"
        exit_code = cli.main(["make", str(multi_shot_clip_path), "-o", str(output), "--music", str(click_track_path)])
        assert exit_code == 0
        assert output.exists()
        assert ingest.probe(output).audio_channels > 0

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

    def test_make_dry_run_writes_plan_and_review_json_without_rendering(
        self, tmp_path, multi_shot_clip_path, capsys
    ) -> None:
        output = tmp_path / "dry_run.json"
        exit_code = cli.main(["make", str(multi_shot_clip_path), "-o", str(output), "--dry-run"])

        assert exit_code == 0
        assert output.exists()
        data = json.loads(output.read_text())
        assert "plan" in data and "ops" in data["plan"] and "confidence" in data["plan"]
        assert "review" in data and "passed" in data["review"]
        captured = capsys.readouterr()
        assert "Directed (dry-run)" in captured.out

    def test_make_dry_run_skips_the_gallery_and_warns_on_stderr(
        self, tmp_path, multi_shot_clip_path, capsys
    ) -> None:
        output = tmp_path / "dry_run.json"
        gallery_dir = tmp_path / "gallery"

        exit_code = cli.main(["make", str(multi_shot_clip_path), "-o", str(output), "--dry-run", "-g", str(gallery_dir)])

        assert exit_code == 0
        assert not gallery_dir.exists()
        assert "skipping the gallery entry" in capsys.readouterr().err

    def test_eval_command_reports_the_zero_ai_stub_as_entirely_invalid(self, capsys) -> None:
        exit_code = cli.main(["eval"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "0/3 valid (0%)" in captured.out
        assert "FAIL" in captured.out

    def test_eval_command_respects_cases_dir_flag(self, tmp_path, content_features_data, capsys) -> None:
        case_path = tmp_path / "one_case.json"
        case_path.write_text(
            json.dumps(
                {
                    "name": "solo_case",
                    "features": content_features_data,
                    "style": cli.DEFAULT_STYLE_PROFILE.model_dump(),
                }
            )
        )

        exit_code = cli.main(["eval", "--cases-dir", str(tmp_path)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "0/1 valid (0%)" in captured.out
        assert "solo_case" in captured.out
