"""Shared OpenCV frame-decoding helpers for the pacing/framing extractors.

Not an extractor itself — just `VideoCapture` bookkeeping so `pacing.py` and
`framing.py` don't each reinvent it. Turning frames into typed features is
still entirely the job of the extractor modules.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoOpenError(Exception):
    """OpenCV could not open the file as a video (missing/corrupt/unsupported)."""


def video_info(path: str | Path) -> tuple[int, int, float, float]:
    """Return `(width, height, fps, duration_seconds)` for a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoOpenError(f"OpenCV could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        return width, height, fps, duration
    finally:
        cap.release()


def sample_frames(
    path: str | Path, timestamps: list[float], *, grayscale: bool = False
) -> list[np.ndarray]:
    """Decode the nearest frame at each timestamp (seconds), skipping failures.

    Frames are returned in the same order as `timestamps`; a timestamp that
    fails to decode (e.g. past EOF) is silently omitted rather than raising,
    since callers sample speculatively and tolerate short lists.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoOpenError(f"OpenCV could not open video: {path}")
    frames: list[np.ndarray] = []
    try:
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if grayscale else frame)
    finally:
        cap.release()
    return frames
