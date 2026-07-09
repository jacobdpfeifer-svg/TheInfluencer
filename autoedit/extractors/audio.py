"""audio — BPM, beat times, RMS energy curve, and a speech-vs-music flag.

Pure measurement via librosa. `has_speech` is a lightweight zero-crossing-rate
heuristic — real speech has a much higher and more variable zero-crossing
rate than tonal/percussive music (consonants/fricatives are broadband bursts;
sustained tones and clicks are not). Per extractor rules, this is a
measurement instrument, not a full voice-activity/speech-to-text model, and
it never decides anything about the edit — it only reports a flag.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Mean zero-crossing rate at/above this reads as speech rather than music.
_SPEECH_ZCR_THRESHOLD = 0.08
# Fewer detected beats than this means "no reliable tempo" -> bpm=None.
_MIN_BEATS_FOR_BPM = 2


class AudioFeatures(BaseModel):
    """Whole-track audio measurements (BPM/beats/energy are track-level, not per-shot)."""

    model_config = ConfigDict(extra="forbid")

    duration: float = Field(ge=0, description="Audio duration in seconds.")
    bpm: Optional[float] = Field(default=None, gt=0, description="Detected tempo, or null if no reliable beat.")
    beat_times: list[float] = Field(default_factory=list, description="Detected beat timestamps in seconds.")
    rms_curve: list[float] = Field(default_factory=list, description="Per-window RMS energy, aligned with `rms_times`.")
    rms_times: list[float] = Field(default_factory=list, description="Start time (seconds) of each `rms_curve` window.")
    has_speech: bool = Field(description="Zero-crossing-rate heuristic: speech present vs music/tone/silence.")


def extract_audio(path: str | Path) -> AudioFeatures:
    """Measure BPM, beat times, an RMS energy curve, and a speech-vs-music flag for `path`.

    Raises:
        FileNotFoundError: `path` does not exist or is not a regular file.

    A raw video clip with no audio stream at all is an expected shape for
    this pipeline (see `extract_pacing`/`extract_framing`, which don't
    require audio either) — so a file that exists but can't be decoded as
    audio returns a zero-valued `AudioFeatures`, same as a decodable but
    silent/empty track, rather than raising.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No such media file: {file_path}")

    try:
        with warnings.catch_warnings():
            # librosa/soundfile warn loudly while falling back through decode
            # backends before failing outright on a file with no audio
            # stream — expected here, so silence it rather than raising.
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(file_path), sr=None, mono=True)
    except Exception:
        return AudioFeatures(duration=0.0, has_speech=False)
    if len(y) == 0 or sr <= 0:
        return AudioFeatures(duration=0.0, has_speech=False)

    duration = float(len(y) / sr)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    tempo_scalar = float(np.atleast_1d(tempo)[0])
    bpm = tempo_scalar if len(beat_times) >= _MIN_BEATS_FOR_BPM and tempo_scalar > 0 else None

    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr).tolist()

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    has_speech = bool(np.mean(zcr) >= _SPEECH_ZCR_THRESHOLD) if len(zcr) else False

    return AudioFeatures(
        duration=duration,
        bpm=bpm,
        beat_times=beat_times,
        rms_curve=rms.tolist(),
        rms_times=rms_times,
        has_speech=has_speech,
    )
