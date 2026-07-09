"""extract_video_features — Phase A: one reference video -> VideoFeatures.

Runs the same four shared extractors `content.extract` (Phase B) runs, just
bundled as `VideoFeatures` instead of assembled into `ContentFeatures` — see
AGENTS.md: "Phase A — Learn... reference videos -> extractors -> per-video
features -> aggregate ... -> StyleProfile". `style.aggregate.aggregate`
takes a list of exactly this, one per reference video.
"""

from __future__ import annotations

from pathlib import Path

from autoedit import ingest
from autoedit.extractors.audio import extract_audio
from autoedit.extractors.framing import extract_framing
from autoedit.extractors.pacing import extract_pacing
from autoedit.extractors.text import extract_text
from autoedit.style.aggregate import VideoFeatures


def extract_video_features(path: str | Path) -> VideoFeatures:
    """Run all four shared extractors on `path` and bundle their output.

    Raises:
        ingest.ProbeError: `path` is missing/corrupt (via `ingest.probe`).
    """
    media = ingest.probe(path)

    pacing = extract_pacing(media.path)
    framing = extract_framing(media.path, shot_bounds=pacing.shot_bounds)
    audio = extract_audio(media.path)
    text = extract_text(media.path, shot_bounds=pacing.shot_bounds)

    return VideoFeatures(pacing=pacing, framing=framing, audio=audio, text=text)
