"""ingest.probe — the only place a raw file path becomes a typed MediaAsset.

Shells out to `ffprobe` (a required system dependency, not a pip package —
see AGENTS.md known traps) to read container/stream metadata. This module
never decodes pixels or audio samples itself; that is the extractors' job.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from autoedit.models import MediaAsset

_FFPROBE_BIN = "ffprobe"
_FFPROBE_TIMEOUT_S = 30


class ProbeError(Exception):
    """Base class for every typed error `probe` can raise."""


class MediaNotFoundError(ProbeError):
    """Raised when `path` does not exist or is not a regular file."""


class CorruptMediaError(ProbeError):
    """Raised when ffprobe runs but the file's media is unreadable, has no
    usable streams, or has no video stream (only video assets are supported)."""


class FFprobeUnavailableError(ProbeError):
    """Raised when the `ffprobe` binary is missing from PATH or cannot be run."""


def probe(path: str | Path) -> MediaAsset:
    """Probe a media file on disk and return its MediaAsset description.

    Only video assets (a video stream must be present) are supported today,
    matching this tool's scope of editing video footage.

    Raises:
        MediaNotFoundError: `path` does not exist or is not a regular file.
        FFprobeUnavailableError: ffprobe is missing from PATH or can't be run.
        CorruptMediaError: the file is corrupt/unreadable or has no video stream.
    """
    media_path = Path(path)
    if not media_path.is_file():
        raise MediaNotFoundError(f"Media file not found: {media_path}")

    if shutil.which(_FFPROBE_BIN) is None:
        raise FFprobeUnavailableError(
            f"'{_FFPROBE_BIN}' was not found on PATH. ffmpeg/ffprobe is a required "
            "system dependency, not a pip package (see AGENTS.md known traps)."
        )

    payload = _run_ffprobe(media_path)
    return _to_media_asset(media_path, payload)


def _run_ffprobe(media_path: Path) -> dict[str, Any]:
    cmd = [
        _FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_FFPROBE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as exc:
        raise CorruptMediaError(f"ffprobe timed out probing {media_path}") from exc
    except OSError as exc:
        raise FFprobeUnavailableError(f"Failed to run ffprobe on {media_path}: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or "unknown error"
        raise CorruptMediaError(f"ffprobe failed on {media_path} (exit {result.returncode}): {stderr}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CorruptMediaError(f"ffprobe returned unparsable output for {media_path}: {exc}") from exc

    if not payload.get("streams"):
        raise CorruptMediaError(f"No media streams found in {media_path} (file may be corrupt)")

    return payload


def _to_media_asset(media_path: Path, payload: dict[str, Any]) -> MediaAsset:
    fmt = payload.get("format", {})
    streams = payload.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise CorruptMediaError(f"{media_path} has no video stream; only video assets are supported")

    duration = _parse_float(fmt.get("duration")) or _parse_float(video_stream.get("duration"))
    if duration is None or duration <= 0:
        raise CorruptMediaError(f"Could not determine a valid duration for {media_path}")

    width = _parse_int(video_stream.get("width"))
    height = _parse_int(video_stream.get("height"))
    if not width or not height:
        raise CorruptMediaError(f"Could not determine frame size for {media_path}")

    fps = _parse_frame_rate(video_stream.get("r_frame_rate")) or _parse_frame_rate(
        video_stream.get("avg_frame_rate")
    )
    if not fps or fps <= 0:
        raise CorruptMediaError(f"Could not determine a valid frame rate for {media_path}")

    codec = video_stream.get("codec_name") or "unknown"
    audio_channels = _parse_int(audio_stream.get("channels")) if audio_stream else 0

    try:
        return MediaAsset(
            path=str(media_path),
            type="video",
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            audio_channels=audio_channels or 0,
        )
    except ValidationError as exc:
        raise CorruptMediaError(f"Probed metadata for {media_path} failed validation: {exc}") from exc


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_frame_rate(value: Any) -> float | None:
    """Parse an ffprobe frame rate string like '30/1' or '30000/1001'."""
    if not value:
        return None
    try:
        frac = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None
    if frac.denominator == 0:
        return None
    return float(frac)
