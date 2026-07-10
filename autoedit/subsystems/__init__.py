"""Subsystems — pure `(Timeline, params) -> Timeline` functions.

Each subsystem only rewrites Timeline instructions; only `renderer.render`
ever turns a Timeline into pixels (see `.cursor/rules/subsystems.mdc`).

`TOOL_MANIFEST` is the map from an `EditOp.tool` name (see AGENTS.md's
`EditPlan` example) to the subsystem function that handles it. The executor
(a later build step) dispatches every op in a plan through this map, then
calls `renderer.render` exactly once at the end.
"""

from __future__ import annotations

from typing import Callable

from autoedit.models.timeline import Timeline
from autoedit.subsystems.cutter import CutterParams, cut
from autoedit.subsystems.effects import EffectParams, apply_effect
from autoedit.subsystems.emoji_adder import EmojiParams, add_emoji
from autoedit.subsystems.music import MusicParams, add_music
from autoedit.subsystems.reframe import ReframeParams, apply_reframe
from autoedit.subsystems.text_adder import TextParams, add_text
from autoedit.subsystems.transitions import TransitionParams, apply_transition

TOOL_MANIFEST: dict[str, Callable[[Timeline, dict], Timeline]] = {
    "cutter": cut,
    "text": add_text,
    "emoji": add_emoji,
    "effect": apply_effect,
    "transition": apply_transition,
    "reframe": apply_reframe,
    "music": add_music,
}

# What params each tool accepts, keyed the same way as `TOOL_MANIFEST`. The
# director validates an `EditOp.params` dict against this before ever
# dispatching it (see `.cursor/rules/director.mdc`: "Reject any op that names
# a tool absent from the manifest or params outside schema").
TOOL_PARAMS_MANIFEST: dict[str, type] = {
    "cutter": CutterParams,
    "text": TextParams,
    "emoji": EmojiParams,
    "effect": EffectParams,
    "transition": TransitionParams,
    "reframe": ReframeParams,
    "music": MusicParams,
}

__all__ = [
    "TOOL_MANIFEST",
    "TOOL_PARAMS_MANIFEST",
    "cut",
    "CutterParams",
    "add_text",
    "TextParams",
    "add_emoji",
    "EmojiParams",
    "apply_effect",
    "EffectParams",
    "apply_transition",
    "TransitionParams",
    "apply_reframe",
    "ReframeParams",
    "add_music",
    "MusicParams",
]
