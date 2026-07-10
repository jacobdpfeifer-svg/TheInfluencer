"""prompts — the SYSTEM_PROMPT, few-shot examples, and message-building for a real LLM call.

This module is pure text/data assembly: no HTTP, no LLM call, nothing that
touches media. `openai_client.make_llm_client` is the module that actually
calls a model; this one just decides what to say to it. Keeping the prompt
content separate from the transport means the prompt can be unit-tested
(shape, wording, few-shot validity) without mocking any network calls.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are the director for an autonomous short-form video editor. You never touch pixels or audio samples: you read a compact JSON brief (the raw footage's ContentFeatures and the creator's learned StyleProfile, plus the list of tools you're allowed to use) and you return a single JSON EditPlan.

Apply this short-form editing expertise when building the plan:
- Hook the viewer in the first 1-2 seconds: open on the highest-energy or most surprising shot available, never a slow warm-up.
- Cut on the beat whenever music is present and the StyleProfile favors it (cut_on_beat: true) -- viewers feel edits that land on the beat as polished, and off-beat cuts as sloppy.
- Insert a pattern interrupt (a cut, caption, emoji, or effect) roughly every 3-5 seconds to hold retention; a shot held too long without a visual change is exactly where viewers scroll away.
- Shape an ascending energy curve across the edit: open strong, build through the middle, and end on a payoff -- never let energy taper off before the video ends.
- Match the StyleProfile's learned caption density and text amount -- do not caption more than the creator's own style does; over-captioning is a common, easily-avoided failure mode.
- Footage is vertical (9:16 aspect); keep the subject centered or in the upper third of the frame, since captions and UI chrome tend to sit near the bottom.
- Keep the total output duration between 15 and 60 seconds -- short-form performs worse well outside that window.
- Less is more: cut weak, redundant, or low-motion shots rather than including everything. A tight 20-second edit beats a padded 45-second one.

Tools available — only use tool names that appear in the brief's "tools" array:
- cutter: select, reorder, and retime shots. params: keep (list of shot ids in output order), sync ("beat"|"none"), beat_times (list of beat timestamps in seconds), trim ({shot_id: max_duration_sec}).
- text: add a caption overlay. params: content (string), style ("karaoke"|"static"), anchor ("top"|"middle"|"bottom"), start (seconds), duration (seconds).
- emoji: add an emoji character overlay. params: glyph (a single emoji), at (seconds), duration (seconds).
- effect: apply a per-shot visual effect. params: kind ("zoom_in"|"zoom_out"|"speed_ramp"|"shake"|"ken_burns"|"blur_intro"|"flash"), shot (shot id), factor (speed multiplier, speed_ramp only).
- transition: add a shot-to-shot transition at one cut point. params: kind ("whip_pan"|"fade"), between ([outgoing_shot_id, incoming_shot_id]), duration (seconds).
- reframe: set a shot's framing and crop mode. params: kind ("center_crop"|"rule_of_thirds"|"fit"), shot (shot id), target_aspect (width/height float, e.g. 0.5625 for 9:16).
- music: DO NOT EMIT this tool. The CLI injects the music bed before directing runs; any music op in your plan will be silently ignored. Music beat grid is already reflected in the brief's beat_times.

Always respond with ONLY a single JSON object shaped exactly like the EditPlan schema: {"ops": [{"tool": "<tool name>", "params": {...}}, ...], "confidence": <0-1>}. Every op's "tool" must be one of the tool names listed in the brief's "tools" array, and "params" must match that tool's own schema. Never include prose, markdown, or commentary outside the JSON object."""


# Diverse, hand-crafted (brief -> ideal EditPlan) pairs used as few-shot
# examples. `"reasoning"` documents WHY the ideal plan makes those choices
# (for humans reading this file and for `test_prompts.py`); it is
# deliberately NOT sent to the model as part of the assistant turn in
# `build_messages` -- the assistant should only ever see/produce clean
# EditPlan JSON, the same contract `director.validate.validate_plan` enforces
# on a real response.
FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        # Beat-synced, high-energy montage: music present, style strongly
        # prefers beat-synced cuts, fast/short shots.
        "brief": {
            "features": {
                "aspect": 0.5625,
                "is_vertical": True,
                "has_speech": False,
                "has_face": True,
                "motion": "high",
                "music_bpm": 128.0,
                "shots": [
                    {"id": "s1", "dur": 0.6, "scale": "close", "faces": 1, "motion": "high"},
                    {"id": "s2", "dur": 0.8, "scale": "wide", "faces": 0, "motion": "high"},
                    {"id": "s3", "dur": 0.5, "scale": "close", "faces": 1, "motion": "med"},
                ],
            },
            "style": {
                "aspect": 0.5625,
                "shot_len_median": 0.6,
                "shot_len_spread": 0.2,
                "cut_on_beat": True,
                "caption_style_freq": {"karaoke": 0.8, "static": 0.2},
                "caption_density": 0.3,
                "text_amount": 0.4,
                "effect_freq": 0.5,
                "sample_count": 20,
            },
            "tools": ["cutter", "effect", "emoji", "music", "reframe", "text", "transition"],
            "template": "punchy_beat_montage",
        },
        "ideal_plan": {
            "ops": [
                {"tool": "cutter", "params": {"keep": ["s1", "s2", "s3"], "sync": "beat", "beat_times": [0.6, 1.4, 1.9]}},
                {"tool": "reframe", "params": {"kind": "center_crop", "shot": "s1", "target_aspect": 0.5625}},
                {"tool": "transition", "params": {"kind": "whip_pan", "between": ["s1", "s2"], "duration": 0.3}},
                {"tool": "effect", "params": {"kind": "zoom_in", "shot": "s1"}},
                {"tool": "emoji", "params": {"glyph": "\U0001f525", "at": 0.3}},
            ],
            "confidence": 0.9,
        },
        "reasoning": (
            "Music is present and the style strongly prefers beat-synced cuts, so every "
            "shot is kept but retimed onto the beat. s1 opens on a close, high-motion, "
            "face-forward shot for an immediate hook. A center_crop reframe keeps s1 "
            "tightly framed on the subject. A whip_pan transition between s1 and s2 "
            "adds kinetic energy at the first cut. zoom_in on s1 gives extra punch, "
            "and an emoji lands within the first second as an early pattern interrupt."
        ),
    },
    {
        # Talking-head with captions: no music, static captions, low motion,
        # long shots that should stay whole.
        "brief": {
            "features": {
                "aspect": 0.5625,
                "is_vertical": True,
                "has_speech": True,
                "has_face": True,
                "motion": "low",
                "music_bpm": None,
                "shots": [
                    {"id": "s1", "dur": 8.0, "scale": "close", "faces": 1, "motion": "low"},
                    {"id": "s2", "dur": 7.0, "scale": "close", "faces": 1, "motion": "low"},
                ],
            },
            "style": {
                "aspect": 0.5625,
                "shot_len_median": 7.5,
                "shot_len_spread": 1.0,
                "cut_on_beat": False,
                "caption_style_freq": {"karaoke": 0.1, "static": 0.9},
                "caption_density": 0.15,
                "text_amount": 0.6,
                "effect_freq": 0.05,
                "sample_count": 15,
            },
            "tools": ["cutter", "effect", "emoji", "music", "reframe", "text", "transition"],
            "template": "talking_head_with_broll",
        },
        "ideal_plan": {
            "ops": [
                {"tool": "cutter", "params": {"keep": ["s1", "s2"], "sync": "none"}},
                {"tool": "reframe", "params": {"kind": "rule_of_thirds", "shot": "s1", "target_aspect": 0.5625}},
                {"tool": "text", "params": {"content": "wait for it...", "style": "static", "anchor": "bottom", "start": 0.0, "duration": 3.0}},
            ],
            "confidence": 0.85,
        },
        "reasoning": (
            "No music, so cuts stay unsynced. Both close, face-forward shots are kept "
            "whole since talking-head content relies on continuity, not rapid cutting. "
            "rule_of_thirds reframe on s1 subtly biases the crop toward the subject's "
            "head without per-frame tracking. A static caption matches the style's "
            "strong preference for static over karaoke, anchored at the bottom so it "
            "doesn't cover the speaker's face."
        ),
    },
    {
        # Mixed: moderate pace, one weak/low-value shot dropped ("less is
        # more"), a single caption. Demonstrates that a minimal plan with no
        # transition or reframe ops is valid -- not every edit needs them.
        "brief": {
            "features": {
                "aspect": 0.5625,
                "is_vertical": True,
                "has_speech": True,
                "has_face": True,
                "motion": "med",
                "music_bpm": None,
                "shots": [
                    {"id": "s1", "dur": 3.0, "scale": "close", "faces": 1, "motion": "med"},
                    {"id": "s2", "dur": 4.0, "scale": "wide", "faces": 0, "motion": "low"},
                    {"id": "s3", "dur": 2.5, "scale": "close", "faces": 1, "motion": "med"},
                ],
            },
            "style": {
                "aspect": 0.5625,
                "shot_len_median": 2.8,
                "shot_len_spread": 0.6,
                "cut_on_beat": False,
                "caption_style_freq": {"karaoke": 0.4, "static": 0.6},
                "caption_density": 0.2,
                "text_amount": 0.3,
                "effect_freq": 0.1,
                "sample_count": 10,
            },
            "tools": ["cutter", "effect", "emoji", "music", "reframe", "text", "transition"],
            "template": "quick_montage",
        },
        "ideal_plan": {
            "ops": [
                {"tool": "cutter", "params": {"keep": ["s1", "s3"], "sync": "none"}},
                {"tool": "text", "params": {"content": "here's the twist", "style": "static", "anchor": "bottom", "start": 0.0, "duration": 2.5}},
            ],
            "confidence": 0.8,
        },
        "reasoning": (
            "s2 is a long, low-motion, faceless wide shot -- exactly the kind of "
            "low-value filler the style's short median shot length argues against. "
            "Dropping it keeps only the two close, face-forward shots and tightens the "
            "edit toward the learned pacing, per 'less is more'. No transition or "
            "reframe needed: the two cuts are clean and the default center_crop already "
            "suits close-up, face-forward shots."
        ),
    },
]


def build_messages(brief: dict[str, Any]) -> list[dict[str, str]]:
    """Build the chat-completions `messages` array for one director call.

    Shape: [system prompt] + [one user/assistant pair per few-shot example,
    in order] + [the real `brief`, as the final user message]. Every
    message's `content` is a plain string (JSON-serialized where the
    content is JSON), matching the OpenAI-compatible chat completions
    request format `openai_client.make_llm_client` posts.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(example["brief"])})
        messages.append({"role": "assistant", "content": json.dumps(example["ideal_plan"])})
    messages.append({"role": "user", "content": json.dumps(brief)})
    return messages
