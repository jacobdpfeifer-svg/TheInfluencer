"""aggregate — Phase A: per-video extractor features -> StyleProfile.

`aggregate()` is a pure function over already-computed feature JSON: given
the four shared extractors' output for each of many reference videos, it
folds them into ONE `StyleProfile` distribution (median ± spread, caption
style frequency, ...) — never a single-video snapshot (see AGENTS.md's
"known trap"). It never touches a file or a pixel itself; that's the
extractors' job.
"""

from __future__ import annotations

import statistics
import warnings

from pydantic import BaseModel, ConfigDict

from autoedit.extractors.audio import AudioFeatures
from autoedit.extractors.framing import FramingFeatures
from autoedit.extractors.pacing import PacingFeatures
from autoedit.extractors.text import TextFeatures
from autoedit.models.style_profile import CaptionStyleFreq, StyleProfile

# A cut within this many seconds of a beat counts as "on beat".
_CUT_BEAT_TOLERANCE_SEC = 0.15
# A video needs at least this fraction of its cuts landing on a beat to cast
# a "cuts on beat" vote; the overall StyleProfile.cut_on_beat is then a
# majority vote across videos that had any beats to check against at all.
_CUT_BEAT_MATCH_FRACTION = 0.5
# Below this many reference videos, warn per AGENTS.md: "a single reference
# video is one data point — StyleProfile must aggregate many."
_MIN_RECOMMENDED_SAMPLES = 3
# Neutral fallback when no video in the corpus has any on-screen captions at
# all, so `CaptionStyleFreq` (which must sum to 1.0) still has something to say.
_NEUTRAL_CAPTION_STYLE_FREQ = CaptionStyleFreq(karaoke=0.5, static=0.5)


class VideoFeatures(BaseModel):
    """Bundled output of the four shared extractors for ONE reference video.

    This is what `content.extract` (Phase B) would assemble into
    `ContentFeatures` for a single piece of raw footage; here, a list of
    these (one per reference video) is what `aggregate()` folds into a
    `StyleProfile`.
    """

    model_config = ConfigDict(extra="forbid")

    pacing: PacingFeatures
    framing: FramingFeatures
    audio: AudioFeatures
    text: TextFeatures


def aggregate(videos: list[VideoFeatures]) -> StyleProfile:
    """Fold per-video extractor features across MANY reference videos into a `StyleProfile`.

    Raises:
        ValueError: `videos` is empty (there is nothing to aggregate).
    """
    if not videos:
        raise ValueError("aggregate() requires at least one video's features")
    if len(videos) < _MIN_RECOMMENDED_SAMPLES:
        warnings.warn(
            f"Aggregating only {len(videos)} video(s); AGENTS.md: a single reference "
            "video is one data point, StyleProfile should aggregate many.",
            stacklevel=2,
        )

    shot_lengths = [length for video in videos for length in video.pacing.shot_lengths]

    return StyleProfile(
        aspect=statistics.median(video.framing.aspect for video in videos),
        shot_len_median=statistics.median(shot_lengths),
        shot_len_spread=_spread(shot_lengths),
        cut_on_beat=_majority_cut_on_beat(videos),
        caption_style_freq=_caption_style_freq(videos),
        caption_density=statistics.fmean(_video_caption_density(video) for video in videos),
        text_amount=statistics.fmean(_video_text_amount(video) for video in videos),
        # No extractor measures applied effects (zoom/pan/etc.) yet — only
        # pacing/framing/audio/text exist per AGENTS.md's build order — so
        # this is a documented placeholder, not a real measurement.
        effect_freq=0.0,
        sample_count=len(videos),
    )


def _spread(values: list[float]) -> float:
    """IQR (Q3 - Q1) when there's enough data to quantile, else population stddev."""
    if len(values) >= 4:
        q1, _, q3 = statistics.quantiles(values, n=4)
        return q3 - q1
    if len(values) >= 2:
        return statistics.pstdev(values)
    return 0.0


def _video_cuts_on_beat(pacing: PacingFeatures, audio: AudioFeatures) -> bool | None:
    """Whether this video's cuts land on beat, or None if there's nothing to check."""
    if not pacing.cuts or not audio.beat_times:
        return None
    matches = sum(
        1 for cut in pacing.cuts if min(abs(cut - beat) for beat in audio.beat_times) <= _CUT_BEAT_TOLERANCE_SEC
    )
    return (matches / len(pacing.cuts)) >= _CUT_BEAT_MATCH_FRACTION


def _majority_cut_on_beat(videos: list[VideoFeatures]) -> bool:
    votes = [v for v in (_video_cuts_on_beat(video.pacing, video.audio) for video in videos) if v is not None]
    if not votes:
        return False
    return (sum(votes) / len(votes)) > 0.5


def _caption_style_freq(videos: list[VideoFeatures]) -> CaptionStyleFreq:
    karaoke = sum(1 for video in videos for shot in video.text.shots if shot.style == "karaoke")
    static = sum(1 for video in videos for shot in video.text.shots if shot.style == "static")
    total = karaoke + static
    if total == 0:
        return _NEUTRAL_CAPTION_STYLE_FREQ
    return CaptionStyleFreq(karaoke=karaoke / total, static=static / total)


def _video_caption_density(video: VideoFeatures) -> float:
    """Caption events per second for one video (0.0 if it has no measurable duration)."""
    if video.pacing.duration <= 0:
        return 0.0
    n_events = sum(len(shot.events) for shot in video.text.shots)
    return n_events / video.pacing.duration


def _video_text_amount(video: VideoFeatures) -> float:
    """Fraction of a video's duration with on-screen text (0.0 if no duration)."""
    if video.pacing.duration <= 0:
        return 0.0
    covered = sum(event.end - event.start for shot in video.text.shots for event in shot.events)
    return covered / video.pacing.duration
