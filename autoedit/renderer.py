"""renderer — the single composite pass that turns a Timeline into an mp4.

Per `.cursor/rules/subsystems.mdc`: **one and only one place renders.**
Subsystems only ever rewrite `Timeline` instructions; this module is the one
place that opens/decodes/writes media, and it does so exactly once per call
(one `write_videofile`, never a chain of renders / re-encodes).

This module is split in two for testability (see AGENTS.md's testing rule —
"no test may require a real video or a render"):

- `build_render_plan(timeline)` is a **pure** function: Timeline -> RenderPlan.
  It resolves every track's items into an ordered, fully-inspectable plan
  with no MoviePy/file IO at all, and is exercised directly by the test
  suite with plain Timeline fixtures.
- `render(timeline, output_path)` is the actual composite pass: it calls
  `build_render_plan`, then drives MoviePy 2.x to realize it into an mp4.
  It requires real source media referenced by the Timeline and is
  intentionally NOT exercised by the automated test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from autoedit.models.timeline import Timeline

# `.cursor/rules/subsystems.mdc`: TextClip needs an explicit font-file path —
# never rely on system font resolution (it fails on Linux). This file is
# vendored in-repo (Apache-2.0, see fonts/Roboto-LICENSE.txt).
DEFAULT_FONT_PATH = Path(__file__).resolve().parent.parent / "fonts" / "Roboto-Regular.ttf"

_VIDEO_TRACK_KIND = "video"
_TEXT_TRACK_KIND = "text"
_EMOJI_TRACK_KIND = "emoji"
_EFFECT_TRACK_KIND = "effect"


class VideoSegment(BaseModel):
    """One source-media excerpt placed on the output timeline."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Path to the source media file this segment is cut from.")
    in_: float = Field(ge=0, description="Start time (seconds) within the SOURCE file.")
    out_: float = Field(gt=0, description="End time (seconds) within the SOURCE file.")
    output_start: float = Field(ge=0, description="Start time (seconds) on the OUTPUT timeline.")
    output_end: float = Field(gt=0, description="End time (seconds) on the OUTPUT timeline.")
    shot: str | None = Field(default=None, description="Shot id this segment came from, if any.")


class TextOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    style: str
    anchor: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class EmojiOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    glyph: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class EffectInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    shot: str | None = None
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class RenderPlan(BaseModel):
    """A fully-resolved, MoviePy-agnostic description of one composite pass."""

    model_config = ConfigDict(extra="forbid")

    video_segments: list[VideoSegment] = Field(default_factory=list)
    text_overlays: list[TextOverlay] = Field(default_factory=list)
    emoji_overlays: list[EmojiOverlay] = Field(default_factory=list)
    effects: list[EffectInstruction] = Field(default_factory=list)
    duration: float = Field(ge=0, description="Total output duration (seconds).")


def build_render_plan(timeline: Timeline) -> RenderPlan:
    """Pure Timeline -> RenderPlan translation. No file IO, no MoviePy."""
    video_segments = _build_video_segments(timeline)
    text_overlays = _build_text_overlays(timeline)
    emoji_overlays = _build_emoji_overlays(timeline)
    effects = _build_effects(timeline)

    ends = [segment.output_end for segment in video_segments]
    ends += [overlay.end for overlay in text_overlays]
    ends += [overlay.end for overlay in emoji_overlays]
    duration = max(ends, default=0.0)

    return RenderPlan(
        video_segments=video_segments,
        text_overlays=text_overlays,
        emoji_overlays=emoji_overlays,
        effects=effects,
        duration=duration,
    )


def _build_video_segments(timeline: Timeline) -> list[VideoSegment]:
    segments = []
    for track in timeline.tracks:
        if track.kind != _VIDEO_TRACK_KIND:
            continue
        for item in track.items:
            if "source" not in item.payload:
                raise ValueError(
                    f"renderer: video item {item.id!r} has no 'source' in its payload; "
                    "the Timeline must be seeded with source media before rendering"
                )
            duration = item.end - item.start
            segments.append(
                VideoSegment(
                    source=item.payload["source"],
                    in_=item.payload.get("in", 0.0),
                    out_=item.payload.get("out", duration),
                    output_start=item.start,
                    output_end=item.end,
                    shot=item.payload.get("shot"),
                )
            )
    return sorted(segments, key=lambda segment: segment.output_start)


def _build_text_overlays(timeline: Timeline) -> list[TextOverlay]:
    overlays = []
    for track in timeline.tracks:
        if track.kind != _TEXT_TRACK_KIND:
            continue
        for item in track.items:
            overlays.append(
                TextOverlay(
                    content=item.payload.get("content", ""),
                    style=item.payload.get("style", "static"),
                    anchor=item.payload.get("anchor", "bottom"),
                    start=item.start,
                    end=item.end,
                )
            )
    return sorted(overlays, key=lambda overlay: overlay.start)


def _build_emoji_overlays(timeline: Timeline) -> list[EmojiOverlay]:
    overlays = []
    for track in timeline.tracks:
        if track.kind != _EMOJI_TRACK_KIND:
            continue
        for item in track.items:
            overlays.append(EmojiOverlay(glyph=item.payload.get("glyph", ""), start=item.start, end=item.end))
    return sorted(overlays, key=lambda overlay: overlay.start)


def _build_effects(timeline: Timeline) -> list[EffectInstruction]:
    effects = []
    for track in timeline.tracks:
        if track.kind != _EFFECT_TRACK_KIND:
            continue
        for item in track.items:
            effects.append(
                EffectInstruction(
                    kind=item.payload.get("kind", ""),
                    shot=item.payload.get("shot"),
                    start=item.start,
                    end=item.end,
                )
            )
    return sorted(effects, key=lambda effect: effect.start)


# --- the actual composite pass (MoviePy; not exercised by the test suite) --

_ANCHOR_POSITIONS: dict[str, tuple[str, str]] = {
    "top": ("center", "top"),
    "middle": ("center", "center"),
    "bottom": ("center", "bottom"),
}


def render(
    timeline: Timeline,
    output_path: str | Path,
    *,
    fps: float = 30.0,
    font_path: str | Path = DEFAULT_FONT_PATH,
) -> Path:
    """Composite `timeline` into a single mp4 at `output_path`. The one render call.

    Requires real source media referenced by the Timeline's video-track item
    payloads (`source`/`in`/`out`) and MoviePy 2.x. Intentionally untested by
    the automated suite — see this module's docstring.
    """
    # Imported lazily so importing this module (e.g. for `build_render_plan`
    # in tests) never requires MoviePy to be importable at collection time.
    from moviepy import CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips

    plan = build_render_plan(timeline)
    if not plan.video_segments:
        raise ValueError("renderer: cannot render a Timeline with no video segments")

    subclips = []
    for segment in plan.video_segments:
        clip = VideoFileClip(segment.source).subclipped(segment.in_, segment.out_)
        clip = _apply_effects_to_clip(clip, segment, plan.effects)
        subclips.append(clip)
    video = concatenate_videoclips(subclips)

    overlays: list[Any] = [video]
    for overlay in plan.text_overlays:
        overlays.append(_build_text_clip(overlay, font_path=font_path).with_start(overlay.start))
    for emoji_overlay in plan.emoji_overlays:
        overlays.append(_build_emoji_clip(emoji_overlay, font_path=font_path).with_start(emoji_overlay.start))

    composite = CompositeVideoClip(overlays) if len(overlays) > 1 else video

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    composite.write_videofile(str(output), fps=fps)
    return output


def _build_text_clip(overlay: TextOverlay, *, font_path: str | Path) -> Any:
    from moviepy import TextClip

    position = _ANCHOR_POSITIONS.get(overlay.anchor, _ANCHOR_POSITIONS["bottom"])
    clip = TextClip(font=str(font_path), text=overlay.content, font_size=48, color="white")
    return clip.with_duration(overlay.end - overlay.start).with_position(position)


def _build_emoji_clip(overlay: EmojiOverlay, *, font_path: str | Path) -> Any:
    from moviepy import TextClip

    clip = TextClip(font=str(font_path), text=overlay.glyph, font_size=72, color="white")
    return clip.with_duration(overlay.end - overlay.start).with_position(("center", "center"))


# Effect kinds this renderer knows how to realize. Unknown kinds are a no-op
# (a director requesting an exotic effect should never crash the render —
# "AI raises the ceiling; rules are the floor").
def _zoom_in(clip: Any, duration: float) -> Any:
    return clip.resized(lambda t: 1.0 + 0.2 * (t / duration if duration > 0 else 0))


def _zoom_out(clip: Any, duration: float) -> Any:
    return clip.resized(lambda t: 1.2 - 0.2 * (t / duration if duration > 0 else 0))


_EFFECT_FUNCS: dict[str, Callable[[Any, float], Any]] = {
    "zoom_in": _zoom_in,
    "zoom_out": _zoom_out,
}


def _apply_effects_to_clip(clip: Any, segment: VideoSegment, effects: list[EffectInstruction]) -> Any:
    for effect in effects:
        if effect.shot is not None and effect.shot == segment.shot:
            effect_func = _EFFECT_FUNCS.get(effect.kind)
            if effect_func is not None:
                clip = effect_func(clip, segment.output_end - segment.output_start)
    return clip
