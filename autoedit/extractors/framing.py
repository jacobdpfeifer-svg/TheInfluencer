"""framing — aspect, face count/position, shot scale, and static/moving camera.

Pure measurement via OpenCV: Haar-cascade frontal-face detection plus a
sampled grayscale frame diff, per shot span. Shared by both phases. Takes
`shot_bounds` from the pacing extractor (or a caller-supplied list); with
none given, the whole video is treated as a single shot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from autoedit.extractors._video import sample_frames, video_info
from autoedit.models.shot import ShotScale

CameraMotion = Literal["static", "moving"]

# How many frames to sample within each shot; kept small (see extractor rules
# on preferring sparse sampling over decoding every frame).
_SAMPLES_PER_SHOT = 3
_FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml"
_FACE_MIN_SIZE = (24, 24)
# A detected face bbox taller than this fraction of the frame reads as a close-up.
_CLOSE_SHOT_FACE_HEIGHT_RATIO = 0.25
# Mean sampled frame diff (0-1) at/above this reads as a moving camera/subject.
_MOVING_MOTION_THRESHOLD = 0.005


class ShotFraming(BaseModel):
    """Per-shot framing measurements."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0, description="Shot start time in seconds.")
    end: float = Field(gt=0, description="Shot end time in seconds.")
    faces: int = Field(ge=0, description="Number of faces detected in the shot.")
    face_positions: list[tuple[float, float]] = Field(
        default_factory=list, description="Normalized (x, y) centers of detected faces, each in [0, 1]."
    )
    scale: ShotScale = Field(description="'close' if a large face fills the frame, else 'wide'.")
    camera: CameraMotion = Field(description="Coarse static-vs-moving read on the shot's sampled frames.")
    brightness: float = Field(ge=0, le=1, description="Normalized mean brightness (0-1).")
    sharpness: float = Field(ge=0, description="Sharpness score (variance of Laplacian).")


class FramingFeatures(BaseModel):
    """Aspect + per-shot framing measurements for a single video."""

    model_config = ConfigDict(extra="forbid")

    aspect: float = Field(gt=0, description="Aspect ratio (width / height) of the source footage.")
    shots: list[ShotFraming] = Field(min_length=1, description="Per-shot framing measurements.")


def extract_framing(
    path: str | Path, shot_bounds: Optional[list[tuple[float, float]]] = None
) -> FramingFeatures:
    """Measure aspect ratio and per-shot face/scale/camera/brightness/sharpness for `path`.

    Args:
        shot_bounds: (start, end) spans in seconds to measure, e.g. from
            `pacing.extract_pacing(...).shot_bounds`. Defaults to the whole
            video as a single shot.

    Raises:
        ValueError: the video has zero duration (unreadable).
    """
    width, height, _, duration = video_info(path)
    if duration <= 0:
        raise ValueError(f"Cannot measure framing for a zero-duration video: {path}")

    bounds = shot_bounds if shot_bounds else [(0.0, duration)]
    shots = [_measure_shot(path, start, end, width, height) for start, end in bounds]
    return FramingFeatures(aspect=width / height, shots=shots)


def _measure_shot(path: str | Path, start: float, end: float, width: int, height: int) -> ShotFraming:
    timestamps = np.linspace(start, end, num=_SAMPLES_PER_SHOT, endpoint=False).tolist()
    frames = sample_frames(path, timestamps, grayscale=False)
    if not frames:
        frames = sample_frames(path, [(start + end) / 2], grayscale=False)
    if not frames:
        return ShotFraming(
            start=start, end=end, faces=0, face_positions=[], scale="wide", camera="static",
            brightness=0.0, sharpness=0.0,
        )

    grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    best_faces: list[tuple[int, int, int, int]] = []
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + _FACE_CASCADE_FILE)
    for gray in grays:
        detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=_FACE_MIN_SIZE)
        if len(detected) > len(best_faces):
            best_faces = [tuple(int(v) for v in box) for box in detected]

    face_positions = [((x + w / 2) / width, (y + h / 2) / height) for x, y, w, h in best_faces]
    is_close = any(h / height >= _CLOSE_SHOT_FACE_HEIGHT_RATIO for _, _, _, h in best_faces)

    brightness = float(np.mean([gray.mean() for gray in grays])) / 255.0
    sharpness = float(np.mean([cv2.Laplacian(gray, cv2.CV_64F).var() for gray in grays]))
    camera = "moving" if _mean_frame_diff(grays) >= _MOVING_MOTION_THRESHOLD else "static"

    return ShotFraming(
        start=start,
        end=end,
        faces=len(best_faces),
        face_positions=face_positions,
        scale="close" if is_close else "wide",
        camera=camera,
        brightness=brightness,
        sharpness=sharpness,
    )


def _mean_frame_diff(grays: list[np.ndarray]) -> float:
    if len(grays) < 2:
        return 0.0
    diffs = [
        float(np.mean(np.abs(grays[i].astype(np.int16) - grays[i - 1].astype(np.int16))))
        for i in range(1, len(grays))
    ]
    return (sum(diffs) / len(diffs)) / 255.0
