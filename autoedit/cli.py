"""CLI — `learn <videos...>` -> StyleProfile JSON, `make <video>` -> mp4, `eval` -> offline LLM report.

Both `learn`/`make` run start to finish with **zero AI** by default: `make`
calls `director.direct` with its default stub LLM, which always falls
through to `heuristic_plan` (see `.cursor/rules/director.mdc` and
AGENTS.md's inviolable law #4: "Every AI decision has a deterministic
fallback"). Passing `--api-key` (or setting `AUTOEDIT_API_KEY`) swaps in a
real model via `director.make_llm_client` — everything else about the
pipeline, including the heuristic fallback on a bad/low-confidence response,
is unchanged.

`make --dry-run` directs and reviews a plan without ever rendering (writes
`{plan, review}` JSON to `-o` instead of an mp4) — useful for iterating on
the director without paying for a render every time. `eval` is a separate,
harsher offline harness (`director.eval.run_eval`): it calls the raw LLM
directly with NO heuristic fallback, so it actually measures the model
(see `director/eval.py`'s docstring for why `direct()` itself can't be used
for this). Both `make --api-key ...` and `eval --api-key ...` accept
`--cache-dir` to skip repeat network calls for an identical brief across
re-runs (`director.cache.cached_llm_client`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from autoedit.content.extract import extract as extract_content_features
from autoedit.content.extract import extract_pool as extract_content_features_pool
from autoedit.director import direct, make_llm_client
from autoedit.director.cache import cached_llm_client
from autoedit.director.eval import format_report, load_eval_cases, run_eval
from autoedit.director.llm import LLMClient, stub_llm
from autoedit.executor import run as run_executor
from autoedit.extractors.audio import extract_audio
from autoedit.models.style_profile import CaptionStyleFreq, StyleProfile
from autoedit.review import GalleryEntry, append_gallery_entry, check_plan
from autoedit.style import aggregate, extract_video_features

# Anthropic's OpenAI-SDK-compatible endpoint (https://docs.anthropic.com/en/api/openai-sdk):
# swap in `--base-url`/`--model` for any other OpenAI-compatible provider.
DEFAULT_API_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_API_MODEL = "claude-sonnet-4-6"

# A neutral starting point for `make` when no learned `--style` profile is
# given (e.g. before ever running `learn`). Not a substitute for actually
# learning a style from reference videos.
DEFAULT_STYLE_PROFILE = StyleProfile(
    aspect=9 / 16,
    shot_len_median=2.0,
    shot_len_spread=0.5,
    cut_on_beat=False,
    caption_style_freq=CaptionStyleFreq(karaoke=0.5, static=0.5),
    caption_density=0.0,
    text_amount=0.0,
    effect_freq=0.0,
    sample_count=1,
)


def learn(video_paths: list[str | Path], output_path: str | Path) -> Path:
    """Extract features from each reference video and write an aggregated StyleProfile JSON."""
    video_features = [extract_video_features(path) for path in video_paths]
    profile = aggregate(video_features)
    output = Path(output_path)
    output.write_text(json.dumps(profile.model_dump(), indent=2))
    return output


def make(
    video_path: str | Path | list[str | Path],
    output_path: str | Path,
    *,
    style_path: str | Path | None = None,
    confidence_threshold: float = 0.6,
    gallery_dir: str | Path | None = None,
    llm: LLMClient | None = None,
    music_path: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Auto-edit raw footage into an mp4, using a learned (or default) StyleProfile.

    `video_path` may be a single path or a list of paths: a pool of clips is
    reduced to one `ContentFeatures` via `content.extract_pool` (its shots
    span every source), so the template system can pour distinct clips into
    distinct slots. A lone path (or a one-element list) behaves exactly as
    the single-clip path always did.

    `music_path`, if given, names a dedicated music bed: its REAL detected
    `beat_times`/`bpm` (via `extractors.audio.extract_audio`) become the
    `features.beat_times`/`music_bpm` the director/template beat-sync
    against — overriding whatever the raw footage's own embedded audio
    measured, which for talking-head/b-roll footage is rarely the intended
    music (AGENTS.md's "known traps" territory: a director beat-syncing
    against silence is a bug, not a feature). The music file is then mixed
    in under the video at render time (see `subsystems/music.py`).

    Sanity-checks the resulting plan against `style` (`review.check_plan`)
    and prints any warnings; pass `gallery_dir` to also append this run to a
    simple HTML results gallery there (`review.append_gallery_entry`). Pass
    `llm` to use a real model instead of the zero-AI stub default (see
    `main`'s `--api-key`/`--model`/`--base-url` flags for the CLI surface).

    `dry_run=True` runs everything through `direct()`/`check_plan` exactly as
    normal, but skips the executor/renderer entirely: `output_path` gets the
    `{"plan": ..., "review": ...}` JSON instead of an mp4, so you can inspect
    what a real director call would decide (structure, confidence, review
    warnings) for the cost of one JSON write, no render or gallery entry.
    """
    paths = list(video_path) if isinstance(video_path, list) else [video_path]
    style = load_style_profile(style_path) if style_path else DEFAULT_STYLE_PROFILE
    features = extract_content_features(paths[0]) if len(paths) == 1 else extract_content_features_pool(paths)
    if music_path is not None:
        music_features = extract_audio(music_path)
        features = features.model_copy(update={"beat_times": music_features.beat_times, "music_bpm": music_features.bpm})
    direct_kwargs = {"confidence_threshold": confidence_threshold}
    if llm is not None:
        direct_kwargs["llm"] = llm
    plan = direct(features, style, **direct_kwargs)

    review = check_plan(plan, style, features=features)
    for warning in review.warnings:
        print(f"[review] {warning}", file=sys.stderr)

    if dry_run:
        output = Path(output_path)
        output.write_text(json.dumps({"plan": plan.model_dump(), "review": review.model_dump()}, indent=2))
        if gallery_dir is not None:
            print("[dry-run] skipping the gallery entry -- there's no rendered mp4 to show.", file=sys.stderr)
        return output

    output = run_executor(features, plan, output_path, music_path=music_path)

    if gallery_dir is not None:
        entry = GalleryEntry(
            video_path=", ".join(str(p) for p in paths),
            output_path=str(output),
            confidence=plan.confidence,
            review=review,
        )
        append_gallery_entry(entry, gallery_dir)

    return output


def load_style_profile(style_path: str | Path) -> StyleProfile:
    data = json.loads(Path(style_path).read_text())
    return StyleProfile.model_validate(data)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoedit", description="Autonomous short-form video editor.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    learn_parser = subparsers.add_parser("learn", help="Learn a StyleProfile from reference videos.")
    learn_parser.add_argument("videos", nargs="+", help="Paths to reference videos you like the style of.")
    learn_parser.add_argument("-o", "--output", default="style_profile.json", help="Output StyleProfile JSON path.")

    make_parser = subparsers.add_parser("make", help="Auto-edit raw footage into an mp4.")
    make_parser.add_argument(
        "video", nargs="+", help="Path(s) to the raw footage. Pass several to pour a pool of clips into the edit."
    )
    make_parser.add_argument(
        "-s", "--style", default=None, help="Path to a learned StyleProfile JSON (default: neutral built-in)."
    )
    make_parser.add_argument("-o", "--output", default="out.mp4", help="Output mp4 path.")
    make_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum director confidence to accept an LLM plan over the heuristic fallback.",
    )
    make_parser.add_argument(
        "-g", "--gallery", default=None, help="Directory to append this run's result to a simple HTML gallery."
    )
    make_parser.add_argument(
        "--music",
        default=None,
        help="Path to a music file to mix in and beat-sync against (overrides any beat grid from the raw footage).",
    )
    make_parser.add_argument(
        "--api-key",
        default=os.environ.get("AUTOEDIT_API_KEY"),
        help="API key for a real LLM director (env: AUTOEDIT_API_KEY). Omit to keep zero-AI mode (default).",
    )
    make_parser.add_argument(
        "--model",
        default=DEFAULT_API_MODEL,
        help=f"Model name for the real LLM director (default: {DEFAULT_API_MODEL!r}). Ignored without --api-key.",
    )
    make_parser.add_argument(
        "--base-url",
        default=DEFAULT_API_BASE_URL,
        help=f"OpenAI-compatible chat completions base URL (default: {DEFAULT_API_BASE_URL!r}). Ignored without --api-key.",
    )
    make_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache LLM responses under this directory, keyed by brief -- repeat runs skip the network call. Ignored without --api-key.",
    )
    make_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Direct the plan and print review warnings, but skip rendering: -o gets {plan, review} JSON instead of an mp4.",
    )

    eval_parser = subparsers.add_parser(
        "eval", help="Offline harness: direct every EvalCase in a directory with the RAW llm response (no heuristic fallback)."
    )
    eval_parser.add_argument(
        "--cases-dir", default="fixtures/eval_cases", help="Directory of EvalCase JSON files (default: fixtures/eval_cases)."
    )
    eval_parser.add_argument(
        "--api-key",
        default=os.environ.get("AUTOEDIT_API_KEY"),
        help="API key for a real LLM (env: AUTOEDIT_API_KEY). Omit to evaluate the always-invalid zero-AI stub "
        "(a harness smoke test, not a real score).",
    )
    eval_parser.add_argument("--model", default=DEFAULT_API_MODEL, help=f"Model name (default: {DEFAULT_API_MODEL!r}).")
    eval_parser.add_argument(
        "--base-url", default=DEFAULT_API_BASE_URL, help=f"OpenAI-compatible chat completions base URL (default: {DEFAULT_API_BASE_URL!r})."
    )
    eval_parser.add_argument(
        "--cache-dir", default=None, help="Cache LLM responses under this directory, keyed by brief."
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "learn":
        output = learn(args.videos, args.output)
        print(f"Learned a style profile from {len(args.videos)} video(s) -> {output}")
    elif args.command == "make":
        llm = (
            make_llm_client(api_key=args.api_key, model=args.model, base_url=args.base_url)
            if args.api_key
            else None
        )
        if llm is not None and args.cache_dir:
            llm = cached_llm_client(llm, args.cache_dir)
        output = make(
            args.video,
            args.output,
            style_path=args.style,
            confidence_threshold=args.confidence_threshold,
            gallery_dir=args.gallery,
            llm=llm,
            music_path=args.music,
            dry_run=args.dry_run,
        )
        verb = "Directed (dry-run)" if args.dry_run else "Rendered"
        print(f"{verb} {len(args.video)} clip(s) -> {output}")
    elif args.command == "eval":
        llm = make_llm_client(api_key=args.api_key, model=args.model, base_url=args.base_url) if args.api_key else stub_llm
        if args.cache_dir:
            llm = cached_llm_client(llm, args.cache_dir)
        cases = load_eval_cases(args.cases_dir)
        report = run_eval(cases, llm)
        print(format_report(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
