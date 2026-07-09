"""The director — ContentFeatures + StyleProfile + tool manifest -> EditPlan.

The only module allowed to call an LLM (see `.cursor/rules/director.mdc`).
`direct()` is the single entry point most callers want; the rest are
exposed for testing/composability.
"""

from autoedit.director.brief import build_brief
from autoedit.director.director import direct
from autoedit.director.heuristic import HEURISTIC_CONFIDENCE, heuristic_plan
from autoedit.director.llm import LLMClient, stub_llm
from autoedit.director.validate import PlanValidationError, validate_plan

__all__ = [
    "direct",
    "build_brief",
    "heuristic_plan",
    "HEURISTIC_CONFIDENCE",
    "stub_llm",
    "LLMClient",
    "validate_plan",
    "PlanValidationError",
]
