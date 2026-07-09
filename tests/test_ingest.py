"""Tests for `autoedit.ingest.probe`.

`probe` is the one place this codebase decodes a real file (via `ffprobe`),
so unlike the rest of the suite it exercises a tiny (~12KB, 1s) checked-in mp4
fixture rather than pure JSON. See `tests/conftest.py::tiny_clip_path` for how
that fixture was generated.
"""

from __future__ import annotations

import shutil

import pytest

from autoedit.ingest import (
    CorruptMediaError,
    FFprobeUnavailableError,
    MediaNotFoundError,
    probe,
)
from autoedit.models import MediaAsset


def test_probes_tiny_fixture_clip(tiny_clip_path):
    asset = probe(tiny_clip_path)

    assert isinstance(asset, MediaAsset)
    assert asset.type == "video"
    assert asset.width == 64
    assert asset.height == 64
    assert asset.fps == pytest.approx(10.0)
    assert asset.duration == pytest.approx(1.0, abs=0.05)
    assert asset.codec == "h264"
    assert asset.audio_channels == 1


def test_probe_accepts_str_path(tiny_clip_path):
    asset = probe(str(tiny_clip_path))
    assert asset.path == str(tiny_clip_path)


def test_probe_raises_typed_error_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(MediaNotFoundError):
        probe(missing)


def test_probe_raises_typed_error_on_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"this is not a video file, just garbage bytes")

    with pytest.raises(CorruptMediaError):
        probe(corrupt)


def test_probe_raises_on_directory_path(tmp_path):
    with pytest.raises(MediaNotFoundError):
        probe(tmp_path)


def test_media_not_found_and_corrupt_are_probe_errors(tmp_path):
    from autoedit.ingest import ProbeError

    assert issubclass(MediaNotFoundError, ProbeError)
    assert issubclass(CorruptMediaError, ProbeError)
    assert issubclass(FFprobeUnavailableError, ProbeError)


def test_probe_raises_when_ffprobe_missing(tiny_clip_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(FFprobeUnavailableError):
        probe(tiny_clip_path)
