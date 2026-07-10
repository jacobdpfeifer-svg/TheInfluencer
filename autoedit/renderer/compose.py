"""Top-level render() — the single composite pass that turns a Timeline into an mp4.

Per `.cursor/rules/subsystems.mdc`: **one and only one place renders.**
This module orchestrates plan.py (pure planning), video.py (clip assembly),
overlays.py (text/emoji), and audio.py (music mixing) into one MoviePy
`write_videofile` call — never a chain of re-encodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._fonts import DEFAULT_FONT_PATH
from .audio import _mix_audio
from .overlays import _build_emoji_clip, _build_flash_clip, _build_karaoke_clips, _build_text_clip
from .plan import RenderPlan, build_render_plan
from .video import _build_video_track

# The canvas's larger dimension, in pixels; the other dimension is derived
# from `target_aspect` so e.g. 9:16 -> 1080x1920, 16:9 -> 1920x1080.
_CANVAS_ANCHOR_PX = 1920

# Mirrors `subsystems.reframe._DEFAULT_TARGET_ASPECT` — the canvas every
# video segment is normalized onto when no `reframe` op names a different one.
_DEFAULT_TARGET_ASPECT = 9 / 16


def render(
    timeline: Any,
    output_path: str | Path,
    *,
    fps: float = 30.0,
    font_path: str | Path = DEFAULT_FONT_PATH,
) -> Path:
    """Composite `timeline` into a single mp4 at `output_path`. The one render call.

    Requires real source media referenced by the Timeline's video-track item
    payloads (`source`/`in`/`out`) and MoviePy 2.x. Intentionally untested by
    the automated suite — see the package docstring in __init__.py.
    """
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
