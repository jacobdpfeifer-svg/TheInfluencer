"""content.extract — Phase B: raw footage -> ContentFeatures.

Wires the shared extractors together for ONE piece of raw footage into the
compact, LLM-brief-ready `ContentFeatures` the director consumes: quantized
buckets (`motion`) and boolean flags (`is_vertical`, `has_face`, `has_speech`)
rather than raw curves — see AGENTS.md's `ContentFeatures` schema and the
director rules' "compact JSON... quantized buckets" guidance.

Per AGENTS.md, Phase B runs the SAME extractors as Phase A (`autoedit.style.
aggregate`); this module is just the Phase B wiring, pointed at one raw clip
instead of a corpus of reference videos.

`extract_pool` extends this to a *pool* of raw clips: the whole point of the
template system (`autoedit.templates`) is pouring your footage into a
creator's slot structure, and those slots (`hook`, `b_roll_1`, ...) want
DISTINCT clips, not sub-shots of one file. It runs `extract` per clip and
merges the results into a single `ContentFeatures` whose `shots` span every
source (each `Shot` already carries its own `source`/`in`/`out`, so the
executor + renderer — which load `VideoFileClip(segment.source)` per segment
— already handle multi-source timelines with no change).

`extractors.text`'s on-screen text measurements are compressed onto
`ContentFeatures` as two whole-clip flags/buckets (`has_text`, `text_style`)
rather than a per-shot field — the same "compact flags... quantized
buckets" treatment `motion`/`has_face` already get (see AGENTS.md's
`ContentFeatures` schema and the director rules' briefing guidance); the
director doesn't need each shot's exact OCR'd text, only whether captions
are already burned in and, roughly, what style they are.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from autoedit import ingest
from autoedit.extractors.audio import extract_audio
from autoedit.extractors.framing import FramingFeatures, ShotFraming, extract_framing
from autoedit.extractors.pacing import PacingFeatures, extract_pacing
from autoedit.extractors.text import TextFeatures, extract_text
from autoedit.models.content_features import ContentFeatures, MotionBucket, TextStyleBucket
from autoedit.models.shot import Shot

# Mean per-shot motion (0-1, same scale as `extractors.pacing.PacingFeatures.
# motion_curve`) below this quantizes to "low"; at/above the second
# threshold quantizes to "high"; otherwise "med".
_LOW_MOTION_MAX = 0.01
_HIGH_MOTION_MIN = 0.05

# Motion buckets, ordered low -> high, for picking a pool's overall bucket.
_MOTION_ORDER: list[MotionBucket] = ["low", "med", "high"]


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
    text = extract_text(media.path, shot_bounds=pacing.shot_bounds)

    shots = _build_shots(media.path, pacing, framing)
    text_style = _text_style_bucket(text)

    return ContentFeatures(
        aspect=media.width / media.height,
        has_speech=audio.has_speech,
        music_bpm=audio.bpm,
        shots=shots,
        motion=_motion_bucket(pacing.motion_curve),
        is_vertical=media.height > media.width,
        has_face=any(shot.faces > 0 for shot in shots),
        beat_times=audio.beat_times,
        has_text=text_style != "none",
        text_style=text_style,
    )


def extract_pool(paths: list[str | Path]) -> ContentFeatures:
    """Extract a POOL of raw clips into one `ContentFeatures` spanning every source.

    Runs `extract` per clip, then merges. Each clip's shots keep their own
    `source`/`in`/`out` (only their ids are re-numbered to stay globally
    unique across the pool: `s1..sN` in clip-then-shot order), so downstream
    the executor/renderer edit across all sources with no change.

    Whole-video fields are merged conservatively:
      - `aspect`/`is_vertical`: from the DOMINANT clip (most shots; ties ->
        first), since one output can only have one aspect.
      - `has_speech`/`has_face`: `any` across clips.
      - `music_bpm`: the first music-y clip's BPM (has a BPM *and* no speech),
        else `None` — a best-effort guess at which clip is the music bed.
      - `motion`: the highest bucket any clip reached (a montage's energy is
        set by its punchiest footage).
      - `beat_times`: `[]`. A montage's final beat grid isn't any single
        source clip's beats; a real value waits on a dedicated music-bed
        step (see `cli.make`'s `--music` flag, which overrides this
        entirely with a real music file's beat grid before the director runs).
      - `has_text`/`text_style`: `has_text` is `any` across clips;
        `text_style` is whichever non-"none" style is most common among
        clips that have text at all (else "none").

    Raises:
        ValueError: `paths` is empty.
        ingest.ProbeError / ValueError: from `extract` on any bad clip.
    """
    if not paths:
        raise ValueError("extract_pool: needs at least one clip path")

    per_clip = [extract(path) for path in paths]

    shots = _merge_and_reindex_shots(per_clip)
    dominant = _dominant_clip(per_clip)
    text_style = _pool_text_style(per_clip)

    return ContentFeatures(
        aspect=dominant.aspect,
        has_speech=any(clip.has_speech for clip in per_clip),
        music_bpm=_pool_music_bpm(per_clip),
        shots=shots,
        motion=_pool_motion(per_clip),
        is_vertical=dominant.is_vertical,
        has_face=any(clip.has_face for clip in per_clip),
        beat_times=[],
        has_text=text_style != "none",
        text_style=text_style,
    )


def _merge_and_reindex_shots(per_clip: list[ContentFeatures]) -> list[Shot]:
    return [
        shot.model_copy(update={"id": f"s{index}"})
        for index, shot in enumerate((shot for clip in per_clip for shot in clip.shots), start=1)
    ]


def _dominant_clip(per_clip: list[ContentFeatures]) -> ContentFeatures:
    return max(per_clip, key=lambda clip: len(clip.shots))


def _pool_music_bpm(per_clip: list[ContentFeatures]) -> float | None:
    for clip in per_clip:
        if clip.music_bpm is not None and not clip.has_speech:
            return clip.music_bpm
    return None


def _pool_motion(per_clip: list[ContentFeatures]) -> MotionBucket:
    return max((clip.motion for clip in per_clip), key=_MOTION_ORDER.index)


def _pool_text_style(per_clip: list[ContentFeatures]) -> TextStyleBucket:
    styles = [clip.text_style for clip in per_clip if clip.text_style != "none"]
    if not styles:
        return "none"
    return Counter(styles).most_common(1)[0][0]


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


def _text_style_bucket(text: TextFeatures) -> TextStyleBucket:
    """Whole-clip text style: whichever non-"none" per-shot style is most common, else "none"."""
    styles = [shot.style for shot in text.shots if shot.style != "none"]
    if not styles:
        return "none"
    return Counter(styles).most_common(1)[0][0]
