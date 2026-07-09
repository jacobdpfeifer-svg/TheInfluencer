"""text — on-screen text position, timing, and style class (karaoke vs static).

Pure measurement via OCR (pytesseract/Tesseract). Per extractor rules this
reports coarse position buckets and a style *class*, never the exact font or
styling — that distinction belongs to the director. Shared by both phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import pytesseract
from pydantic import BaseModel, ConfigDict, Field

from autoedit.extractors._video import sample_frames, video_info

TextAnchor = Literal["top", "middle", "bottom"]
TextStyle = Literal["karaoke", "static", "none"]

# How densely to sample frames for OCR within a shot.
_DEFAULT_SAMPLE_RATE_HZ = 5.0
# Tesseract word-confidence (0-100) below this is treated as noise, not text.
_MIN_WORD_CONFIDENCE = 30
# A shot needs at least this many distinct text events, each averaging no
# more than a third of the shot's duration, to read as "karaoke" rather than
# a single sustained ("static") caption.
_MIN_EVENTS_FOR_KARAOKE = 3


class TextEvent(BaseModel):
    """A single span of sustained on-screen text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="OCR'd on-screen text (not styling/font).")
    start: float = Field(ge=0, description="Start time in seconds.")
    end: float = Field(gt=0, description="End time in seconds.")
    anchor: TextAnchor = Field(description="Coarse vertical position bucket.")


class ShotText(BaseModel):
    """Per-shot on-screen text measurements."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0, description="Shot start time in seconds.")
    end: float = Field(gt=0, description="Shot end time in seconds.")
    events: list[TextEvent] = Field(default_factory=list, description="Detected text spans within the shot.")
    style: TextStyle = Field(description="'karaoke' (rapid word-by-word), 'static' (sustained), or 'none'.")


class TextFeatures(BaseModel):
    """Per-shot on-screen text features for a single video."""

    model_config = ConfigDict(extra="forbid")

    shots: list[ShotText] = Field(min_length=1, description="Per-shot on-screen text measurements.")


def extract_text(
    path: str | Path,
    shot_bounds: Optional[list[tuple[float, float]]] = None,
    *,
    sample_rate_hz: float = _DEFAULT_SAMPLE_RATE_HZ,
) -> TextFeatures:
    """Measure on-screen text position, timing, and style class for `path`.

    Args:
        shot_bounds: (start, end) spans in seconds, e.g. from
            `pacing.extract_pacing(...).shot_bounds`. Defaults to the whole
            video as a single shot.
        sample_rate_hz: how many frames per second to run OCR on.

    Raises:
        ValueError: the video has zero duration (unreadable).
    """
    _, _, _, duration = video_info(path)
    if duration <= 0:
        raise ValueError(f"Cannot measure on-screen text for a zero-duration video: {path}")

    bounds = shot_bounds if shot_bounds else [(0.0, duration)]
    shots = [_measure_shot_text(path, start, end, sample_rate_hz) for start, end in bounds]
    return TextFeatures(shots=shots)


def _measure_shot_text(path: str | Path, start: float, end: float, sample_rate_hz: float) -> ShotText:
    frame_dt = 1.0 / sample_rate_hz
    n_samples = max(2, round((end - start) * sample_rate_hz))
    timestamps = np.linspace(start, end, num=n_samples, endpoint=False).tolist()
    frames = sample_frames(path, timestamps, grayscale=False)

    readings = [(ts, *_ocr_frame(frame)) for ts, frame in zip(timestamps, frames)]
    events = _merge_readings_into_events(readings, frame_dt=frame_dt, shot_end=end)
    return ShotText(start=start, end=end, events=events, style=_classify_style(events, start, end))


def _ocr_frame(frame: np.ndarray) -> tuple[str, Optional[TextAnchor]]:
    """OCR a single frame; return (combined text, coarse vertical anchor)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    frame_height = frame.shape[0]

    words: list[str] = []
    y_centers: list[float] = []
    for word, conf, top, height in zip(data["text"], data["conf"], data["top"], data["height"]):
        if word.strip() and int(conf) >= _MIN_WORD_CONFIDENCE:
            words.append(word.strip())
            y_centers.append(top + height / 2)

    if not words:
        return "", None
    return " ".join(words), _anchor_bucket(sum(y_centers) / len(y_centers) / frame_height)


def _anchor_bucket(normalized_y: float) -> TextAnchor:
    if normalized_y < 1 / 3:
        return "top"
    if normalized_y > 2 / 3:
        return "bottom"
    return "middle"


def _merge_readings_into_events(
    readings: list[tuple[float, str, Optional[TextAnchor]]], *, frame_dt: float, shot_end: float
) -> list[TextEvent]:
    """Collapse consecutive identical-text readings into spans."""
    events: list[TextEvent] = []
    run_text: Optional[str] = None
    run_anchor: Optional[TextAnchor] = None
    run_start = 0.0
    run_last = 0.0

    def _flush() -> None:
        if run_text:
            events.append(
                TextEvent(text=run_text, start=run_start, end=min(run_last + frame_dt, shot_end), anchor=run_anchor)
            )

    for ts, text, anchor in readings:
        if text and text == run_text:
            run_last = ts
            continue
        _flush()
        run_text, run_anchor, run_start, run_last = (text or None), anchor, ts, ts
    _flush()
    return events


def _classify_style(events: list[TextEvent], start: float, end: float) -> TextStyle:
    if not events:
        return "none"
    if len(events) < _MIN_EVENTS_FOR_KARAOKE:
        return "static"
    avg_event_dur = sum(e.end - e.start for e in events) / len(events)
    if avg_event_dur <= (end - start) / _MIN_EVENTS_FOR_KARAOKE:
        return "karaoke"
    return "static"
