"""pacing — cut list, shot lengths, and a per-shot motion curve.

Pure measurement, per `.cursor/rules/extractors.mdc`: PySceneDetect finds cut
points; a sampled mean-abs grayscale frame diff scores motion per shot. The
same function serves Phase A (reference videos) and Phase B (raw footage).
"""

from __future__ import annotations

import statistics
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from autoedit.extractors._video import sample_frames, video_info

# How many frames to sample within each shot to score its motion. Kept small
# since we prefer ffprobe/PySceneDetect-style sparse sampling over decoding
# every frame (see extractor rules).
_MOTION_SAMPLES_PER_SHOT = 4


class PacingFeatures(BaseModel):
    """Cut-list + motion measurements for a single video."""

    model_config = ConfigDict(extra="forbid")

    duration: float = Field(gt=0, description="Total video duration in seconds.")
    cuts: list[float] = Field(
        description="Cut timestamps in seconds (shot boundaries, excluding 0 and the final duration)."
    )
    shot_bounds: list[tuple[float, float]] = Field(
        min_length=1, description="Per-shot (start, end) spans in seconds, derived from `cuts`."
    )
    shot_lengths: list[float] = Field(
        min_length=1, description="Per-shot duration in seconds, aligned with `shot_bounds`."
    )
    shot_len_median: float = Field(gt=0, description="Median shot length in seconds.")
    motion_curve: list[float] = Field(
        min_length=1,
        description="Per-shot motion magnitude in [0, 1] (mean sampled frame diff), aligned with `shot_bounds`.",
    )


def extract_pacing(
    path: str | Path,
    *,
    threshold: float = 27.0,
    min_shot_len_sec: float = 0.2,
) -> PacingFeatures:
    """Measure the cut list, shot lengths, and per-shot motion curve for `path`.

    Args:
        threshold: PySceneDetect `ContentDetector` sensitivity (lower = more cuts).
        min_shot_len_sec: shortest shot PySceneDetect is allowed to report;
            converted to frames using the video's own fps.

    Raises:
        ValueError: the video has zero duration or zero fps (unreadable).
    """
    _, _, fps, duration = video_info(path)
    if duration <= 0 or fps <= 0:
        raise ValueError(f"Cannot measure pacing for a zero-duration/zero-fps video: {path}")

    min_scene_len = max(1, round(min_shot_len_sec * fps))
    video = open_video(str(path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    if scene_list:
        shot_bounds = [(start.seconds, end.seconds) for start, end in scene_list]
        # PySceneDetect's last scene can end a frame short of the probed
        # duration; clamp so shots always cover the whole video.
        shot_bounds[-1] = (shot_bounds[-1][0], duration)
    else:
        shot_bounds = [(0.0, duration)]

    cuts = [start for start, _ in shot_bounds[1:]]
    shot_lengths = [end - start for start, end in shot_bounds]
    motion_curve = [_shot_motion(path, start, end) for start, end in shot_bounds]

    return PacingFeatures(
        duration=duration,
        cuts=cuts,
        shot_bounds=shot_bounds,
        shot_lengths=shot_lengths,
        shot_len_median=statistics.median(shot_lengths),
        motion_curve=motion_curve,
    )


def _shot_motion(path: str | Path, start: float, end: float) -> float:
    """Mean absolute grayscale frame-diff within `[start, end)`, normalized to 0-1."""
    timestamps = np.linspace(start, end, num=_MOTION_SAMPLES_PER_SHOT, endpoint=False).tolist()
    frames = sample_frames(path, timestamps, grayscale=True)
    if len(frames) < 2:
        return 0.0
    diffs = [
        float(np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))))
        for i in range(1, len(frames))
    ]
    return (sum(diffs) / len(diffs)) / 255.0
