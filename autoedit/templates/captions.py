"""captions — generate_caption_copy: deterministic, footage-derived caption text.

Per AGENTS.md's inviolable law #4 ("every AI decision has a deterministic
fallback"), the director's caption COPY needs a non-LLM floor too, not just
its structure/timing. Before this module, that floor was one hardcoded
string (`heuristic._DEFAULT_CAPTION_TEXT`) or a template's flat placeholder
tag (`TextSlot.placeholder`, e.g. `"TITLE"`/`"CTA"`) shown verbatim.

`generate_caption_copy` treats a template's `placeholder` as a semantic TAG
naming the slot's PURPOSE, not literal copy, and picks a small, ordered set
of rule-based phrases from `ContentFeatures` signals (motion, faces,
speech, music) for known tags. An unrecognized tag passes through
unchanged, so a template author's own literal copy is never clobbered —
this only ever *replaces* a known purpose-tag, never invents on top of real
text. Shared by both deterministic floors: `templates.filler.fill_template`
(templated text slots) and `director.heuristic._simple_heuristic_plan` (the
floor under the floor, via the synthetic `"TITLE"` tag).

This is rule-based text SELECTION, not generation — no LLM, no randomness,
fully deterministic for a given `ContentFeatures`. A real LLM director is
free to write better, situational copy; this only ever backs it up.
"""

from __future__ import annotations

from typing import Callable

from autoedit.models.content_features import ContentFeatures

# BPM at/above this reads as "high energy" music for caption-copy purposes.
_HIGH_ENERGY_BPM = 120.0

CaptionRule = tuple[Callable[[ContentFeatures], bool], str]

_TITLE_RULES: list[CaptionRule] = [
    (lambda f: f.has_face and f.motion == "high", "wait for it \U0001f440"),
    (lambda f: f.music_bpm is not None and f.music_bpm >= _HIGH_ENERGY_BPM, "this hits different \U0001f525"),
    (lambda f: f.has_speech, "the story so far..."),
    (lambda f: f.has_face, "you had to be there \u2728"),
]
_TITLE_FALLBACK = "New content \U0001f4a5"

_CTA_RULES: list[CaptionRule] = [
    (lambda f: f.music_bpm is not None, "turn the sound on \U0001f3b6"),
    (lambda f: f.has_face, "follow for more like this \u2728"),
]
_CTA_FALLBACK = "follow for more \u2728"


def _title_copy(features: ContentFeatures) -> str:
    return next((phrase for predicate, phrase in _TITLE_RULES if predicate(features)), _TITLE_FALLBACK)


def _cta_copy(features: ContentFeatures) -> str:
    return next((phrase for predicate, phrase in _CTA_RULES if predicate(features)), _CTA_FALLBACK)


# Keyed by the placeholder TAG (case-insensitive), not literal template text.
_CAPTION_GENERATORS: dict[str, Callable[[ContentFeatures], str]] = {
    "TITLE": _title_copy,
    "CTA": _cta_copy,
}


def generate_caption_copy(features: ContentFeatures, placeholder: str) -> str:
    """Turn a purpose TAG (`"TITLE"`/`"CTA"`, case-insensitive) into footage-derived copy.

    Any other `placeholder` (a template author's own literal text, or a
    future tag this module doesn't know yet) is returned unchanged.
    """
    generator = _CAPTION_GENERATORS.get(placeholder.upper())
    return generator(features) if generator is not None else placeholder
