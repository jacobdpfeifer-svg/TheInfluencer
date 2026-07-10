"""Video track assembly: effects, reframes, and transition application.

All MoviePy clip operations for assembling the video track live here —
effect transforms (zoom, shake, ken_burns, etc.), reframe/crop normalization,
and the transition logic (fade/whip_pan) that positions clips with overlap.

None of these functions render to a file; they return transformed MoviePy
clip objects for `compose.render` to composite.
"""

from __future__ import annotations

from typing import Any, Callable

from .plan import EffectInstruction, ReframeInstruction, RenderPlan, TransitionInstruction, VideoSegment

# --- tuning constants -------------------------------------------------------

_DEFAULT_SPEED_RAMP_FACTOR = 1.5
_SHAKE_MAX_OFFSET_PX = 3
_KEN_BURNS_START_ZOOM = 1.1
_KEN_BURNS_END_ZOOM = 1.0
_BLUR_INTRO_DURATION_SEC = 0.5
_BLUR_INTRO_MAX_SIGMA = 10.0
_WHIP_PAN_BLUR_KERNEL_PX = 15

# "rule_of_thirds" biases a cover-crop's vertical center to this fraction of
# the (scaled) frame's height — above true center (0.5), toward where a
# subject's face/head usually sits, without per-frame subject tracking.
_RULE_OF_THIRDS_Y_BIAS = 1 / 3


# --- video track assembly ---------------------------------------------------


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
    uniformly between every pair — it can't express "overlap only here, not there."
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


# --- transition helpers -----------------------------------------------------


def _find_transition(
    transitions: list[TransitionInstruction], outgoing_shot: str | None, incoming_shot: str | None
) -> TransitionInstruction | None:
    for transition in transitions:
        if transition.between == [outgoing_shot, incoming_shot]:
            return transition
    return None


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


# --- reframe helpers --------------------------------------------------------


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


# --- effect functions -------------------------------------------------------

# Effect kinds this renderer knows how to realize as a per-clip transform.
# ("flash" is deliberately absent -- see overlays._build_flash_clip's docstring.)
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
    del duration, effect  # the blur window is fixed (_BLUR_INTRO_DURATION_SEC), not the shot's own span.

    def frame_filter(get_frame: Callable[[float], Any], t: float) -> Any:
        frame = get_frame(t)
        if t >= _BLUR_INTRO_DURATION_SEC:
            return frame
        sigma = _BLUR_INTRO_MAX_SIGMA * (1 - t / _BLUR_INTRO_DURATION_SEC)
        return _gaussian_blur_frame(frame, sigma)

    return clip.transform(frame_filter)


def _gaussian_blur_frame(frame: Any, sigma: float) -> Any:
    import numpy as np
    from PIL import Image, ImageFilter

    if sigma <= 0:
        return frame
    blurred = Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred)


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
