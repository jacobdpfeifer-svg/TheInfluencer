"""Shared feature extractors — the ONLY code allowed to decode pixels or audio.

Each extractor is a pure function `(path, ...) -> typed features`. The same
extractors serve both Phase A (learning from reference videos) and Phase B
(editing raw footage); callers simply point them at different inputs. See
AGENTS.md and `.cursor/rules/extractors.mdc` for the rules these follow.
"""

from autoedit.extractors.audio import AudioFeatures, extract_audio
from autoedit.extractors.framing import FramingFeatures, ShotFraming, extract_framing
from autoedit.extractors.pacing import PacingFeatures, extract_pacing
from autoedit.extractors.text import ShotText, TextEvent, TextFeatures, extract_text

__all__ = [
    "PacingFeatures",
    "extract_pacing",
    "FramingFeatures",
    "ShotFraming",
    "extract_framing",
    "AudioFeatures",
    "extract_audio",
    "TextFeatures",
    "ShotText",
    "TextEvent",
    "extract_text",
]
