"""content.extract — Phase B: raw footage -> ContentFeatures.

Wires the shared extractors together for ONE piece of raw footage into the
compact, LLM-brief-ready `ContentFeatures` the director consumes: quantized
buckets (`motion`) and boolean flags (`is_vertical`, `has_face`, `has_speech`)
rather than raw curves — see AGENTS.md's `ContentFeatures` schema and the
director rules' "compact JSON... quantized buckets" guidance.

Per AGENTS.md, Phase B runs the SAME extractors as Phase A (`autoedit.style.
aggregate`); this module is just the Phase B wiring, pointed at one raw clip
instead of a corpus of reference videos.

`extractors.text` has no field on `ContentFeatures` yet (see AGENTS.md's
schema list), so it isn't wired in here — that's a schema gap for a future
build step, not an oversight.
"""

from __future__ import annotations

from pathlib import Path

from autoedit import ingest
from autoedit.extractors.audio import extract_audio
from autoedit.extractors.framing import FramingFeatures, ShotFraming, extract_framing
from autoedit.extractors.pacing import PacingFeatures, extract_pacing
from autoedit.models.content_features import ContentFeatures, MotionBucket
from autoedit.models.shot import Shot

# Mean per-shot motion (0-1, same scale as `extractors.pacing.PacingFeatures.
# motion_curve`) below this quantizes to "low"; at/above the second
# threshold quantizes to "high"; otherwise "med".
_LOW_MOTION_MAX = 0.01
_HIGH_MOTION_MIN = 0.05


def extract(path: str | Path) -> ContentFeatures:
    """Run the shared extractors on `path` and assemble `ContentFeatures`.

    Raises:
        ingest.ProbeError: `path` is missing/corrupt (via `ingest.probe`).
        ValueError: a downstream extractor can't measure a zero-duration file.
    """
    media = ingest.probe(path)

    pacing = extract_pacing(media.path)
    framing = extract_framing(media.path, shot_bounds=pacing.shot_bounds)
    audio = extract_audio(media.path)

    shots = _build_shots(media.path, pacing, framing)

    return ContentFeatures(
        aspect=media.width / media.height,
        has_speech=audio.has_speech,
        music_bpm=audio.bpm,
        shots=shots,
        motion=_motion_bucket(pacing.motion_curve),
        is_vertical=media.height > media.width,
        has_face=any(shot.faces > 0 for shot in shots),
    )


def _build_shots(source: str, pacing: PacingFeatures, framing: FramingFeatures) -> list[Shot]:
    """Zip pacing's shot_bounds/motion_curve with framing's per-shot measurements.

    `extract_framing` was called with `pacing.shot_bounds`, so the three
    sequences are already aligned one-to-one, in order.
    """
    return [
        _build_shot(index, source, bounds, motion, framing_shot)
        for index, (bounds, motion, framing_shot) in enumerate(
            zip(pacing.shot_bounds, pacing.motion_curve, framing.shots), start=1
        )
    ]


def _build_shot(
    index: int, source: str, bounds: tuple[float, float], motion: float, framing_shot: ShotFraming
) -> Shot:
    start, end = bounds
    return Shot(
        id=f"s{index}",
        source=source,
        in_=start,
        out_=end,
        dur=end - start,
        motion=motion,
        brightness=framing_shot.brightness,
        sharpness=framing_shot.sharpness,
        faces=framing_shot.faces,
        scale=framing_shot.scale,
    )


def _motion_bucket(motion_curve: list[float]) -> MotionBucket:
    mean_motion = sum(motion_curve) / len(motion_curve)
    if mean_motion < _LOW_MOTION_MAX:
        return "low"
    if mean_motion >= _HIGH_MOTION_MIN:
        return "high"
    return "med"
