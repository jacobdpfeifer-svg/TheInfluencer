"""CLI — `learn <videos...>` -> StyleProfile JSON, `make <video>` -> mp4.

Both commands run start to finish with **zero AI** by default: `make` calls
`director.direct` with its default stub LLM, which always falls through to
`heuristic_plan` (see `.cursor/rules/director.mdc` and AGENTS.md's
inviolable law #4: "Every AI decision has a deterministic fallback"). Passing
`--api-key` (or setting `AUTOEDIT_API_KEY`) swaps in a real model via
`director.make_llm_client` — everything else about the pipeline, including
the heuristic fallback on a bad/low-confidence response, is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from autoedit.content.extract import extract as extract_content_features
from autoedit.director import direct, make_llm_client
from autoedit.director.llm import LLMClient
from autoedit.executor import run as run_executor
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
    video_path: str | Path,
    output_path: str | Path,
    *,
    style_path: str | Path | None = None,
    confidence_threshold: float = 0.6,
    gallery_dir: str | Path | None = None,
    llm: LLMClient | None = None,
) -> Path:
    """Auto-edit one raw footage video into an mp4, using a learned (or default) StyleProfile.

    Sanity-checks the resulting plan against `style` (`review.check_plan`)
    and prints any warnings; pass `gallery_dir` to also append this run to a
    simple HTML results gallery there (`review.append_gallery_entry`). Pass
    `llm` to use a real model instead of the zero-AI stub default (see
    `main`'s `--api-key`/`--model`/`--base-url` flags for the CLI surface).
    """
    style = load_style_profile(style_path) if style_path else DEFAULT_STYLE_PROFILE
    features = extract_content_features(video_path)
    direct_kwargs = {"confidence_threshold": confidence_threshold}
    if llm is not None:
        direct_kwargs["llm"] = llm
    plan = direct(features, style, **direct_kwargs)

    review = check_plan(plan, style, features=features)
    for warning in review.warnings:
        print(f"[review] {warning}", file=sys.stderr)

    output = run_executor(features, plan, output_path)

    if gallery_dir is not None:
        entry = GalleryEntry(video_path=str(video_path), output_path=str(output), confidence=plan.confidence, review=review)
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

    make_parser = subparsers.add_parser("make", help="Auto-edit one raw footage video into an mp4.")
    make_parser.add_argument("video", help="Path to the raw footage video.")
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
        output = make(
            args.video,
            args.output,
            style_path=args.style,
            confidence_threshold=args.confidence_threshold,
            gallery_dir=args.gallery,
            llm=llm,
        )
        print(f"Rendered {args.video} -> {output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
