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

# Roboto has NO emoji glyphs — text drawn from it renders blank for emoji
# codepoints. Emoji overlays need a dedicated emoji-capable font instead;
# this one is vendored in-repo (SIL OFL 1.1, see fonts/NotoColorEmoji-LICENSE.txt).
EMOJI_FONT_PATH = Path(__file__).resolve().parent.parent / "fonts" / "NotoColorEmoji-Regular.ttf"

# EMOJI_FONT_PATH embeds exactly ONE bitmap (CBDT/CBLC) strike, at this pixel
# size — there is no "regular size" to ask for, this is it. MoviePy's
# TextClip can't use this font at all: it always probes the font via
# `PIL.ImageFont.truetype(font)` with NO size argument before ever looking at
# our `font_size`, which raises "invalid pixel size" for any single-strike
# bitmap font. So emoji are rasterized directly via Pillow (at this, the
# font's one real size) in `_build_emoji_clip` instead of going through
# `TextClip` — see that function's docstring.
_EMOJI_STRIKE_PIXEL_SIZE = 109
# Visual size (pixels, tallest dimension) emoji are downscaled to on the
# output frame after rasterizing at the strike's native (much larger) size.
_EMOJI_DISPLAY_SIZE_PX = 72

_VIDEO_TRACK_KIND = "video"
_TEXT_TRACK_KIND = "text"
_EMOJI_TRACK_KIND = "emoji"
_EFFECT_TRACK_KIND = "effect"
_TRANSITION_TRACK_KIND = "transition"
_REFRAME_TRACK_KIND = "reframe"

# Mirrors `subsystems.reframe._DEFAULT_TARGET_ASPECT` -- the canvas every
# video segment is normalized onto when no `reframe` op names a different
# one (see `_resolve_target_aspect`/`_apply_reframe_to_clip`).
_DEFAULT_TARGET_ASPECT = 9 / 16
# The canvas's larger dimension, in pixels; the other dimension is derived
# from `target_aspect` so e.g. 9:16 -> 1080x1920, 16:9 -> 1920x1080.
_CANVAS_ANCHOR_PX = 1920
# "rule_of_thirds" biases a cover-crop's vertical center to this fraction of
# the (scaled) frame's height -- above true center (0.5), toward where a
# subject's face/head usually sits, without per-frame subject tracking.
_RULE_OF_THIRDS_Y_BIAS = 1 / 3

# --- new-effect / transition tuning constants (see each `_EFFECT_FUNCS` entry) --
_DEFAULT_SPEED_RAMP_FACTOR = 1.5
_SHAKE_MAX_OFFSET_PX = 3
_KEN_BURNS_START_ZOOM = 1.1
_KEN_BURNS_END_ZOOM = 1.0
_BLUR_INTRO_DURATION_SEC = 0.5
_BLUR_INTRO_MAX_SIGMA = 10.0
_FLASH_DURATION_SEC = 0.05
_WHIP_PAN_BLUR_KERNEL_PX = 15


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
    factor: float | None = Field(default=None, description="Speed multiplier, only set for kind='speed_ramp'.")


class TransitionInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    between: list[str] = Field(description="[outgoing shot id, incoming shot id].")
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class ReframeInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = Field(description="'center_crop', 'rule_of_thirds', or 'fit'.")
    target_aspect: float = Field(gt=0)
    shot: str | None = None
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class AudioInstruction(BaseModel):
    """A music-bed excerpt to mix in under the video's own audio."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Path to the music file this excerpt is cut from.")
    in_: float = Field(ge=0, description="Start time (seconds) within the SOURCE file.")
    out_: float = Field(gt=0, description="End time (seconds) within the SOURCE file.")
    start: float = Field(ge=0, description="Start time (seconds) on the OUTPUT timeline.")
    end: float = Field(gt=0, description="End time (seconds) on the OUTPUT timeline.")
    volume: float = Field(ge=0, description="Gain multiplier applied to this excerpt.")


class RenderPlan(BaseModel):
    """A fully-resolved, MoviePy-agnostic description of one composite pass."""

    model_config = ConfigDict(extra="forbid")

    video_segments: list[VideoSegment] = Field(default_factory=list)
    text_overlays: list[TextOverlay] = Field(default_factory=list)
    emoji_overlays: list[EmojiOverlay] = Field(default_factory=list)
    effects: list[EffectInstruction] = Field(default_factory=list)
    transitions: list[TransitionInstruction] = Field(default_factory=list)
    reframes: list[ReframeInstruction] = Field(default_factory=list)
    audio: list[AudioInstruction] = Field(default_factory=list)
    duration: float = Field(ge=0, description="Total output duration (seconds).")


def build_render_plan(timeline: Timeline) -> RenderPlan:
    """Pure Timeline -> RenderPlan translation. No file IO, no MoviePy."""
    video_segments = _build_video_segments(timeline)
    text_overlays = _build_text_overlays(timeline)
    emoji_overlays = _build_emoji_overlays(timeline)
    effects = _build_effects(timeline)
    transitions = _build_transitions(timeline)
    reframes = _build_reframes(timeline)
    audio = _build_audio_instructions(timeline)

    ends = [segment.output_end for segment in video_segments]
    ends += [overlay.end for overlay in text_overlays]
    ends += [overlay.end for overlay in emoji_overlays]
    duration = max(ends, default=0.0)

    return RenderPlan(
        video_segments=video_segments,
        text_overlays=text_overlays,
        emoji_overlays=emoji_overlays,
        effects=effects,
        transitions=transitions,
        reframes=reframes,
        audio=audio,
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
                    factor=item.payload.get("factor"),
                )
            )
    return sorted(effects, key=lambda effect: effect.start)


def _build_transitions(timeline: Timeline) -> list[TransitionInstruction]:
    transitions = []
    for track in timeline.tracks:
        if track.kind != _TRANSITION_TRACK_KIND:
            continue
        for item in track.items:
            transitions.append(
                TransitionInstruction(
                    kind=item.payload.get("kind", ""),
                    between=item.payload.get("between", []),
                    start=item.start,
                    end=item.end,
                )
            )
    return sorted(transitions, key=lambda transition: transition.start)


def _build_reframes(timeline: Timeline) -> list[ReframeInstruction]:
    reframes = []
    for track in timeline.tracks:
        if track.kind != _REFRAME_TRACK_KIND:
            continue
        for item in track.items:
            reframes.append(
                ReframeInstruction(
                    kind=item.payload.get("kind", "center_crop"),
                    target_aspect=item.payload.get("target_aspect", _DEFAULT_TARGET_ASPECT),
                    shot=item.payload.get("shot"),
                    start=item.start,
                    end=item.end,
                )
            )
    return sorted(reframes, key=lambda reframe: reframe.start)


def _build_audio_instructions(timeline: Timeline) -> list[AudioInstruction]:
    instructions = []
    for track in timeline.tracks:
        if track.kind != "audio":
            continue
        for item in track.items:
            if "source" not in item.payload:
                raise ValueError(
                    f"renderer: audio item {item.id!r} has no 'source' in its payload; "
                    "the Timeline must be seeded with source media before rendering"
                )
            duration = item.end - item.start
            instructions.append(
                AudioInstruction(
                    source=item.payload["source"],
                    in_=item.payload.get("in", 0.0),
                    out_=item.payload.get("out", duration),
                    start=item.start,
                    end=item.end,
                    volume=item.payload.get("volume", 1.0),
                )
            )
    return sorted(instructions, key=lambda instruction: instruction.start)


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
    from moviepy import CompositeVideoClip

    plan = build_render_plan(timeline)
    if not plan.video_segments:
        raise ValueError("renderer: cannot render a Timeline with no video segments")

    canvas_w, canvas_h = _canvas_size(_resolve_target_aspect(plan))
    video = _build_video_track(plan, canvas_w=canvas_w, canvas_h=canvas_h)

    overlays: list[Any] = [video]
    for overlay in plan.text_overlays:
        if overlay.style == "karaoke":
            overlays.extend(_build_karaoke_clips(overlay, font_path=font_path))
        else:
            overlays.append(_build_text_clip(overlay, font_path=font_path).with_start(overlay.start))
    for emoji_overlay in plan.emoji_overlays:
        overlays.append(_build_emoji_clip(emoji_overlay).with_start(emoji_overlay.start))
    for effect in plan.effects:
        if effect.kind == "flash":
            overlays.append(_build_flash_clip(size=video.size).with_start(effect.start))

    composite = CompositeVideoClip(overlays) if len(overlays) > 1 else video

    if plan.audio:
        composite = _mix_audio(composite, plan.audio)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    composite.write_videofile(str(output), fps=fps)
    return output


def _mix_audio(clip: Any, audio_instructions: list[AudioInstruction]) -> Any:
    """Mix `audio_instructions` (music-bed excerpts) UNDER `clip`'s own audio, if any.

    A talking-head shot's own dialogue survives (`clip.audio` is kept as one
    of the layers); the music bed(s) just play alongside it at their own
    `volume`. A clip with no embedded audio at all (e.g. silent b-roll) just
    gets the music bed as its only audio.
    """
    from moviepy import AudioFileClip, CompositeAudioClip
    from moviepy.audio.fx import MultiplyVolume

    music_layers = []
    for instruction in audio_instructions:
        excerpt = AudioFileClip(instruction.source).subclipped(instruction.in_, instruction.out_)
        excerpt = excerpt.with_effects([MultiplyVolume(instruction.volume)]).with_start(instruction.start)
        music_layers.append(excerpt)

    layers = ([clip.audio] if clip.audio is not None else []) + music_layers
    if not layers:
        return clip
    mixed = layers[0] if len(layers) == 1 else CompositeAudioClip(layers)
    return clip.with_audio(mixed)


def _canvas_size(target_aspect: float) -> tuple[int, int]:
    """Output canvas (width, height) in pixels for `target_aspect` (width / height).

    Anchors the LARGER dimension at `_CANVAS_ANCHOR_PX` and derives the
    other from `target_aspect`, so every segment (regardless of its own
    native resolution/aspect) normalizes onto one consistent frame size —
    e.g. 9:16 -> 1080x1920, 16:9 -> 1920x1080, 1:1 -> 1920x1920.
    """
    if target_aspect <= 1:
        return round(_CANVAS_ANCHOR_PX * target_aspect), _CANVAS_ANCHOR_PX
    return _CANVAS_ANCHOR_PX, round(_CANVAS_ANCHOR_PX / target_aspect)


def _resolve_target_aspect(plan: RenderPlan) -> float:
    """The output canvas's aspect: the first `reframe` instruction's, if any, else the default.

    A `Template` (and so `fill_template`) names exactly one `aspect_ratio`
    for a whole edit, so in practice every `reframe` op on a given Timeline
    agrees — this just needs *a* value when at least one exists.
    """
    if plan.reframes:
        return plan.reframes[0].target_aspect
    return _DEFAULT_TARGET_ASPECT


def _build_video_track(plan: RenderPlan, *, canvas_w: int, canvas_h: int) -> Any:
    """Build the single (possibly transitioned) video clip from `plan.video_segments`.

    Every segment is first normalized onto one `(canvas_w, canvas_h)` canvas
    (see `_apply_reframe_to_clip`) — mixed-aspect, pooled footage (see
    `content.extract_pool`) would otherwise leave `CompositeVideoClip`
    sizing itself off whichever clip happens to be first and pinning the
    rest top-left, misaligned. Effects then run on the now-canvas-sized clip.

    Replaces plain `concatenate_videoclips` with a manually-positioned
    `CompositeVideoClip` because a `fade` transition needs a NEGATIVE gap
    (overlap) between exactly one adjacent pair of clips, and
    `concatenate_videoclips`'s `padding` argument is a single value applied
    uniformly between every pair (see its source: `timings + padding *
    arange(...)`) — it can't express "overlap only here, not there."
    """
    from moviepy import CompositeVideoClip, VideoFileClip
    from moviepy.video.fx import CrossFadeIn, CrossFadeOut

    clips: list[Any] = []
    for segment in plan.video_segments:
        clip = VideoFileClip(segment.source).subclipped(segment.in_, segment.out_)
        reframe = _find_reframe(plan.reframes, segment.shot)
        mode = reframe.kind if reframe is not None else "center_crop"
        clip = _apply_reframe_to_clip(clip, mode=mode, canvas_w=canvas_w, canvas_h=canvas_h)
        clip = _apply_effects_to_clip(clip, segment, plan.effects)
        clips.append(clip)

    if len(clips) == 1:
        return clips[0]

    # `overlap_before[i]` = seconds clip i's start is pulled BACK from the
    # natural end of clip i-1 (only nonzero for a `fade` transition between
    # that specific pair — see the docstring above).
    overlap_before = [0.0] * len(clips)
    for i in range(len(clips) - 1):
        outgoing_shot = plan.video_segments[i].shot
        incoming_shot = plan.video_segments[i + 1].shot
        transition = _find_transition(plan.transitions, outgoing_shot, incoming_shot)
        if transition is None:
            continue
        half_window = (transition.end - transition.start) / 2
        if transition.kind == "whip_pan":
            clips[i] = _apply_edge_blur(clips[i], window_sec=half_window, at_start=False)
            clips[i + 1] = _apply_edge_blur(clips[i + 1], window_sec=half_window, at_start=True)
        elif transition.kind == "fade":
            fade_duration = min(transition.end - transition.start, clips[i].duration, clips[i + 1].duration)
            if fade_duration > 0:
                clips[i] = clips[i].with_effects([CrossFadeOut(fade_duration)])
                clips[i + 1] = clips[i + 1].with_effects([CrossFadeIn(fade_duration)])
                overlap_before[i + 1] = fade_duration

    starts = [0.0] * len(clips)
    for i in range(1, len(clips)):
        starts[i] = starts[i - 1] + clips[i - 1].duration - overlap_before[i]

    positioned = [clip.with_start(start) for clip, start in zip(clips, starts)]
    return CompositeVideoClip(positioned, size=(canvas_w, canvas_h))


def _find_transition(
    transitions: list[TransitionInstruction], outgoing_shot: str | None, incoming_shot: str | None
) -> TransitionInstruction | None:
    for transition in transitions:
        if transition.between == [outgoing_shot, incoming_shot]:
            return transition
    return None


def _find_reframe(reframes: list[ReframeInstruction], shot: str | None) -> ReframeInstruction | None:
    return next((reframe for reframe in reframes if reframe.shot == shot and shot is not None), None)


def _apply_reframe_to_clip(clip: Any, *, mode: str, canvas_w: int, canvas_h: int) -> Any:
    """Normalize `clip` onto exactly `(canvas_w, canvas_h)`, per `mode` (see `subsystems/reframe.py`)."""
    if mode == "fit":
        return _fit_letterbox(clip, canvas_w, canvas_h)
    y_bias = _RULE_OF_THIRDS_Y_BIAS if mode == "rule_of_thirds" else 0.5
    return _cover_crop(clip, canvas_w, canvas_h, y_bias=y_bias)


def _cover_crop(clip: Any, canvas_w: int, canvas_h: int, *, y_bias: float) -> Any:
    """Scale `clip` to fully COVER `(canvas_w, canvas_h)`, then crop the excess.

    `y_bias` (0=top, 0.5=center, 1=bottom) places the crop window's vertical
    center as a fraction of the scaled frame's height, clamped so the
    window never runs past either edge.
    """
    from moviepy.video.fx import Crop

    scale = max(canvas_w / clip.w, canvas_h / clip.h)
    resized = clip.resized(scale)

    x_center = resized.w / 2
    y_center = min(max(resized.h * y_bias, canvas_h / 2), resized.h - canvas_h / 2)
    return resized.with_effects([Crop(x_center=x_center, y_center=y_center, width=canvas_w, height=canvas_h)])


def _fit_letterbox(clip: Any, canvas_w: int, canvas_h: int) -> Any:
    """Scale `clip` to fit ENTIRELY inside `(canvas_w, canvas_h)`, centered on a black background (no crop)."""
    from moviepy import ColorClip, CompositeVideoClip

    scale = min(canvas_w / clip.w, canvas_h / clip.h)
    resized = clip.resized(scale).with_position("center")
    background = ColorClip(size=(canvas_w, canvas_h), color=(0, 0, 0)).with_duration(clip.duration)
    return CompositeVideoClip([background, resized], size=(canvas_w, canvas_h))


def _build_text_clip(overlay: TextOverlay, *, font_path: str | Path) -> Any:
    from moviepy import TextClip

    position = _ANCHOR_POSITIONS.get(overlay.anchor, _ANCHOR_POSITIONS["bottom"])
    clip = TextClip(font=str(font_path), text=overlay.content, font_size=48, color="white")
    return clip.with_duration(overlay.end - overlay.start).with_position(position)


def _build_karaoke_clips(overlay: TextOverlay, *, font_path: str | Path) -> list[Any]:
    """Split `overlay.content` into words, each shown for an even slice of `[overlay.start, overlay.end)`.

    "karaoke" (per AGENTS.md's `TextStyle`: "rapid word-by-word") means one
    word on screen at a time, not the whole phrase held for the whole span —
    the static-caption path (`_build_text_clip`) already covers "the whole
    phrase, the whole span." An empty/whitespace-only `content` yields no
    clips at all rather than dividing by zero.
    """
    from moviepy import TextClip

    words = overlay.content.split()
    if not words:
        return []

    position = _ANCHOR_POSITIONS.get(overlay.anchor, _ANCHOR_POSITIONS["bottom"])
    per_word_duration = (overlay.end - overlay.start) / len(words)

    clips = []
    for index, word in enumerate(words):
        clip = TextClip(font=str(font_path), text=word, font_size=48, color="white")
        clip = clip.with_duration(per_word_duration).with_position(position)
        clips.append(clip.with_start(overlay.start + index * per_word_duration))
    return clips


def _build_emoji_clip(overlay: EmojiOverlay) -> Any:
    """Rasterize `overlay.glyph` via Pillow directly and wrap it in an ImageClip.

    Deliberately does NOT use MoviePy's `TextClip` (unlike `_build_text_clip`):
    `EMOJI_FONT_PATH` is a bitmap-strike color emoji font with only ONE
    embedded size (`_EMOJI_STRIKE_PIXEL_SIZE`), and `TextClip.__init__`
    unconditionally probes any font via `PIL.ImageFont.truetype(font)` with
    no size argument (Pillow's default size, 10) before it ever looks at our
    `font_size` — which raises `ValueError: ... invalid pixel size` for a
    single-strike font no matter what size we ask for. Pillow can still draw
    this font perfectly well directly (`embedded_color=True`) when we
    instantiate it ourselves at its one real size, so we do that and hand
    MoviePy the resulting RGBA bitmap as an `ImageClip` instead.
    """
    import numpy as np
    from moviepy import ImageClip
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(EMOJI_FONT_PATH), _EMOJI_STRIKE_PIXEL_SIZE)
    canvas_size = _EMOJI_STRIKE_PIXEL_SIZE * 2
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    ImageDraw.Draw(canvas).text((0, 0), overlay.glyph, font=font, embedded_color=True)
    glyph_bbox = canvas.getbbox() or (0, 0, 1, 1)  # fall back to a 1x1 pixel if nothing was drawn
    glyph_image = canvas.crop(glyph_bbox)

    clip = ImageClip(np.array(glyph_image)).resized(height=_EMOJI_DISPLAY_SIZE_PX)
    return clip.with_duration(overlay.end - overlay.start).with_position(("center", "center"))


def _build_flash_clip(*, size: tuple[int, int]) -> Any:
    """A `_FLASH_DURATION_SEC` solid-white frame, composited at the flashing shot's start time.

    Unlike the other effects, `flash` isn't a per-clip transform (it doesn't
    change how the shot itself plays) — it's an extra overlay, exactly like
    a text/emoji overlay, which is why it's built and composited in `render`
    rather than living in `_EFFECT_FUNCS`/`_apply_effects_to_clip`.
    """
    from moviepy import ColorClip

    return ColorClip(size=size, color=(255, 255, 255)).with_duration(_FLASH_DURATION_SEC)


def _apply_edge_blur(clip: Any, *, window_sec: float, at_start: bool) -> Any:
    """Apply `_horizontal_motion_blur` only within `window_sec` of one edge of `clip` (for `whip_pan`)."""

    def frame_filter(get_frame: Callable[[float], Any], t: float) -> Any:
        frame = get_frame(t)
        in_window = t <= window_sec if at_start else t >= (clip.duration - window_sec)
        return _horizontal_motion_blur(frame) if in_window else frame

    return clip.transform(frame_filter)


def _horizontal_motion_blur(frame: Any, *, kernel_px: int = _WHIP_PAN_BLUR_KERNEL_PX) -> Any:
    import numpy as np

    half_kernel = kernel_px // 2
    offsets = range(-half_kernel, half_kernel + 1)
    accum = sum(np.roll(frame, offset, axis=1).astype(np.float64) for offset in offsets)
    return (accum / len(offsets)).astype(frame.dtype)


def _gaussian_blur_frame(frame: Any, sigma: float) -> Any:
    import numpy as np
    from PIL import Image, ImageFilter

    if sigma <= 0:
        return frame
    blurred = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred)


# Effect kinds this renderer knows how to realize as a per-clip transform.
# ("flash" is deliberately absent -- see `_build_flash_clip`'s docstring.)
# Unknown kinds are a no-op (a director requesting an exotic effect should
# never crash the render — "AI raises the ceiling; rules are the floor").
def _zoom_in(clip: Any, duration: float, effect: EffectInstruction) -> Any:
    del effect  # unused: zoom_in has no per-instance parameters.
    return clip.resized(lambda t: 1.0 + 0.2 * (t / duration if duration > 0 else 0))


def _zoom_out(clip: Any, duration: float, effect: EffectInstruction) -> Any:
    del effect
    return clip.resized(lambda t: 1.2 - 0.2 * (t / duration if duration > 0 else 0))


def _speed_ramp(clip: Any, duration: float, effect: EffectInstruction) -> Any:
    """Speed the clip up by `effect.factor` (or `_DEFAULT_SPEED_RAMP_FACTOR`) for its full span.

    `MultiplySpeed.apply` shrinks the returned clip's `duration` by the same
    factor internally, which is the "adjust the timeline item's duration
    accordingly" this effect calls for — no separate bookkeeping needed.
    """
    del duration  # the effect spans the clip's own full duration either way.
    from moviepy.video.fx import MultiplySpeed

    factor = effect.factor if effect.factor is not None else _DEFAULT_SPEED_RAMP_FACTOR
    return clip.with_effects([MultiplySpeed(factor)])


def _shake(clip: Any, duration: float, effect: EffectInstruction) -> Any:
    del duration, effect
    import random

    return clip.with_position(
        lambda t: (
            random.uniform(-_SHAKE_MAX_OFFSET_PX, _SHAKE_MAX_OFFSET_PX),
            random.uniform(-_SHAKE_MAX_OFFSET_PX, _SHAKE_MAX_OFFSET_PX),
        )
    )


def _ken_burns(clip: Any, duration: float, effect: EffectInstruction) -> Any:
    """Pan+zoom from 110% centered-top to 100% centered, over `duration` seconds.

    MoviePy's own `.cropped()` (`vfx.Crop`) only accepts static ints/floats
    for its crop rectangle — it computes `x1/y1/x2/y2` once from whatever
    it's given, so a callable `x_center`/`y_center` can't work (see
    `Crop.apply`). A truly time-varying crop needs a per-frame `.transform`
    instead, which is what this does: resize each frame to the current zoom
    level, then crop back down to the clip's original size, sliding the
    crop window's vertical center from the top edge down to true center as
    `t` goes from 0 to `duration`.
    """
    import numpy as np
    from PIL import Image

    width, height = clip.w, clip.h

    def _zoom(t: float) -> float:
        progress = min(t / duration, 1.0) if duration > 0 else 1.0
        return _KEN_BURNS_START_ZOOM + (_KEN_BURNS_END_ZOOM - _KEN_BURNS_START_ZOOM) * progress

    def frame_filter(get_frame: Callable[[float], Any], t: float) -> Any:
        zoom = _zoom(t)
        frame = get_frame(t)
        zoomed_w, zoomed_h = max(width, round(width * zoom)), max(height, round(height * zoom))
        zoomed = np.array(Image.fromarray(frame).resize((zoomed_w, zoomed_h)))

        x1 = (zoomed_w - width) // 2  # always horizontally centered
        top_anchored_y1 = 0
        center_anchored_y1 = (zoomed_h - height) // 2
        progress = min(t / duration, 1.0) if duration > 0 else 1.0
        y1 = round(top_anchored_y1 + (center_anchored_y1 - top_anchored_y1) * progress)

        return zoomed[y1 : y1 + height, x1 : x1 + width]

    return clip.transform(frame_filter)


def _blur_intro(clip: Any, duration: float, effect: EffectInstruction) -> Any:
    del duration, effect  # the blur window is fixed (`_BLUR_INTRO_DURATION_SEC`), not the shot's own span.

    def frame_filter(get_frame: Callable[[float], Any], t: float) -> Any:
        frame = get_frame(t)
        if t >= _BLUR_INTRO_DURATION_SEC:
            return frame
        sigma = _BLUR_INTRO_MAX_SIGMA * (1 - t / _BLUR_INTRO_DURATION_SEC)
        return _gaussian_blur_frame(frame, sigma)

    return clip.transform(frame_filter)


_EFFECT_FUNCS: dict[str, Callable[[Any, float, EffectInstruction], Any]] = {
    "zoom_in": _zoom_in,
    "zoom_out": _zoom_out,
    "speed_ramp": _speed_ramp,
    "shake": _shake,
    "ken_burns": _ken_burns,
    "blur_intro": _blur_intro,
}


def _apply_effects_to_clip(clip: Any, segment: VideoSegment, effects: list[EffectInstruction]) -> Any:
    for effect in effects:
        if effect.shot is not None and effect.shot == segment.shot:
            effect_func = _EFFECT_FUNCS.get(effect.kind)
            if effect_func is not None:
                clip = effect_func(clip, segment.output_end - segment.output_start, effect)
    return clip
