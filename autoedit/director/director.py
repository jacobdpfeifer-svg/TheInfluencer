"""direct — the one place an LLM is called, per `.cursor/rules/director.mdc`.

`direct(features, style)` is the whole director in one call:

    build_brief -> llm(brief) -> validate_plan -> (fallback if needed)

Any failure mode — malformed output, an unknown tool, bad params, a
too-low confidence score, or the `llm` callable itself raising — is caught
here and turned into a `heuristic_plan` result instead. Nothing this module
does ever touches pixels; its only output is a validated `EditPlan`.
"""

from __future__ import annotations

import logging
from typing import Any

from autoedit.director.brief import build_brief
from autoedit.director.heuristic import heuristic_plan
from autoedit.director.llm import LLMClient, stub_llm
from autoedit.director.validate import PlanValidationError, validate_plan
from autoedit.models.content_features import ContentFeatures
from autoedit.models.plan import EditPlan
from autoedit.models.style_profile import StyleProfile
from autoedit.subsystems import TOOL_MANIFEST
from autoedit.templates.matcher import match_template

logger = logging.getLogger(__name__)

# Below this confidence, prefer the deterministic heuristic over a "valid
# but unsure" model plan (`.cursor/rules/director.mdc`: "On invalid output
# OR confidence below threshold, fall back to heuristic_plan").
_DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def direct(
    features: ContentFeatures,
    style: StyleProfile,
    *,
    llm: LLMClient = stub_llm,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    manifest: dict[str, Any] = TOOL_MANIFEST,
) -> EditPlan:
    """Turn `features` + `style` into a validated `EditPlan`, via `llm` or the heuristic fallback."""
    matched = match_template(features, style)
    brief = build_brief(features, style, manifest, template_name=matched.name)

    try:
        raw_response = llm(brief)
    except Exception:
        logger.exception("director: llm call raised; falling back to heuristic_plan. brief=%r", brief)
        return heuristic_plan(style, features)

    logger.info("director: prompt(brief)=%r response=%r", brief, raw_response)

    try:
        plan = validate_plan(raw_response)
    except PlanValidationError:
        logger.warning("director: llm response failed validation; falling back to heuristic_plan.", exc_info=True)
        return heuristic_plan(style, features)

    if plan.confidence < confidence_threshold:
        logger.info(
            "director: plan confidence %.2f below threshold %.2f; falling back to heuristic_plan.",
            plan.confidence,
            confidence_threshold,
        )
        return heuristic_plan(style, features)

    return plan
