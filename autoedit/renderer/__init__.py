"""renderer — the single composite pass that turns a Timeline into an mp4.

Per `.cursor/rules/subsystems.mdc`: **one and only one place renders.**
Subsystems only ever rewrite `Timeline` instructions; this package is the one
place that opens/decodes/writes media, and it does so exactly once per call
(one `write_videofile`, never a chain of renders / re-encodes).

This package is split into focused submodules for maintainability:

- ``plan.py``     — pure Pydantic models + ``build_render_plan`` + ``_build_*``
                    helpers. Zero MoviePy, fully tested on fixture Timelines.
- ``video.py``    — ``_build_video_track``, all effect/transition/reframe
                    clip operations (``_EFFECT_FUNCS``, zoom, shake, …).
- ``overlays.py`` — ``_build_text_clip``, ``_build_karaoke_clips``,
                    ``_build_emoji_clip``, ``_build_flash_clip``.
- ``audio.py``    — ``_mix_audio``: music-bed mixing under the video clip.
- ``compose.py``  — top-level ``render()`` that orchestrates the above into
                    one ``CompositeVideoClip.write_videofile`` call.

All public names (and the private names accessed by the test suite) are
re-exported here so every existing import remains unchanged:

    from autoedit.renderer import render, build_render_plan, RenderPlan, …
    from autoedit import renderer; renderer._build_karaoke_clips(…)
"""

from ._fonts import DEFAULT_FONT_PATH, EMOJI_FONT_PATH
from .audio import _mix_audio
from .compose import render
from .overlays import (
    _ANCHOR_POSITIONS,
    _build_emoji_clip,
    _build_flash_clip,
    _build_karaoke_clips,
    _build_text_clip,
)
from .plan import (
    AudioInstruction,
    EffectInstruction,
    EmojiOverlay,
    ReframeInstruction,
    RenderPlan,
    TextOverlay,
    TransitionInstruction,
    VideoSegment,
    build_render_plan,
)
from .video import (
    _EFFECT_FUNCS,
    _apply_effects_to_clip,
    _build_video_track,
)

__all__ = [
    # public API
    "DEFAULT_FONT_PATH",
    "EMOJI_FONT_PATH",
    "render",
    "build_render_plan",
    "RenderPlan",
    "VideoSegment",
    "TextOverlay",
    "EmojiOverlay",
    "EffectInstruction",
    "TransitionInstruction",
    "ReframeInstruction",
    "AudioInstruction",
    # private names accessed by tests (via `from autoedit import renderer`)
    "_EFFECT_FUNCS",
    "_apply_effects_to_clip",
    "_build_video_track",
    "_mix_audio",
    "_build_karaoke_clips",
    "_build_text_clip",
    "_build_emoji_clip",
    "_build_flash_clip",
    "_ANCHOR_POSITIONS",
]
