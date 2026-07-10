"""Pure plan-building: Timeline -> RenderPlan.

All Pydantic sub-models and build_render_plan (plus its _build_* helpers) live
here. Zero MoviePy, zero file IO — this module is fully exercised by the test
suite on plain fixture Timelines.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from autoedit.models.timeline import Timeline

_VIDEO_TRACK_KIND = "video"
_TEXT_TRACK_KIND = "text"
_EMOJI_TRACK_KIND = "emoji"
_EFFECT_TRACK_KIND = "effect"
_TRANSITION_TRACK_KIND = "transition"
_REFRAME_TRACK_KIND = "reframe"

# Default output aspect ratio (width/height) when no reframe instruction names one.
_DEFAULT_TARGET_ASPECT = 9 / 16


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
