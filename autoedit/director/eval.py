"""eval — an OFFLINE harness for measuring a raw LLMClient's plan quality.

Deliberately bypasses `director.direct`'s safety net: `direct` silently
swallows any failure (bad JSON, unknown tool, low confidence...) and falls
back to `heuristic_plan`, which is exactly right for production but useless
for *measuring* a model — a harness that also falls back on failure would
report 100% success no matter how bad the model is. `run_eval` instead
calls the LLM directly and records exactly how it failed, if it did.

"Offline" per AGENTS.md's testing rule: every case here is `ContentFeatures`
+ `StyleProfile` JSON (an `EvalCase`), never a real video or a render — the
harness measures the director's JSON-in/JSON-out contract only, the same
contract `direct()` itself is built on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from autoedit.director.brief import build_brief
from autoedit.director.llm import LLMClient
from autoedit.director.validate import PlanValidationError, validate_plan
from autoedit.models.content_features import ContentFeatures
from autoedit.models.style_profile import StyleProfile
from autoedit.review import check_plan
from autoedit.subsystems import TOOL_MANIFEST


class EvalCase(BaseModel):
    """One (features, style) pair to direct and score -- see `fixtures/eval_cases/` for examples."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Human-readable case id, e.g. 'hook_montage_high_energy'.")
    features: ContentFeatures
    style: StyleProfile


class EvalCaseResult(BaseModel):
    """The outcome for one `EvalCase`: whether the raw LLM output was even usable, and if so, how good."""

    model_config = ConfigDict(extra="forbid")

    name: str
    valid: bool = Field(description="True iff the raw response passed `validate_plan` (schema + tool/params).")
    confidence: float | None = Field(default=None, description="The plan's own confidence, if `valid`.")
    review_passed: bool | None = Field(default=None, description="`review.check_plan(...).passed`, if `valid`.")
    warnings: list[str] = Field(default_factory=list, description="`review.check_plan(...).warnings`, if `valid`.")
    error: str | None = Field(default=None, description="Why it's invalid: the llm call's or validation's exception.")


class EvalReport(BaseModel):
    """Aggregate results across every `EvalCase` in a run."""

    model_config = ConfigDict(extra="forbid")

    cases: list[EvalCaseResult]

    @property
    def total_count(self) -> int:
        return len(self.cases)

    @property
    def valid_count(self) -> int:
        return sum(1 for case in self.cases if case.valid)

    @property
    def review_passed_count(self) -> int:
        return sum(1 for case in self.cases if case.review_passed)

    @property
    def valid_rate(self) -> float:
        return self.valid_count / self.total_count if self.total_count else 0.0


def run_eval(cases: list[EvalCase], llm: LLMClient, *, manifest: dict[str, Any] = TOOL_MANIFEST) -> EvalReport:
    """Direct every case's raw `llm` response and score it -- no heuristic fallback (see module docstring)."""
    return EvalReport(cases=[_eval_one(case, llm, manifest) for case in cases])


def _eval_one(case: EvalCase, llm: LLMClient, manifest: dict[str, Any]) -> EvalCaseResult:
    brief = build_brief(case.features, case.style, manifest)

    try:
        raw_response = llm(brief)
    except Exception as exc:
        return EvalCaseResult(name=case.name, valid=False, error=f"llm call raised: {exc}")

    try:
        plan = validate_plan(raw_response)
    except PlanValidationError as exc:
        return EvalCaseResult(name=case.name, valid=False, error=f"validation failed: {exc}")

    review = check_plan(plan, case.style, features=case.features)
    return EvalCaseResult(
        name=case.name, valid=True, confidence=plan.confidence, review_passed=review.passed, warnings=review.warnings
    )


def load_eval_cases(cases_dir: str | Path) -> list[EvalCase]:
    """Load one `EvalCase` per `*.json` file directly under `cases_dir`, sorted by filename."""
    directory = Path(cases_dir)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"eval: no *.json case files found under {directory}")
    return [EvalCase.model_validate(json.loads(path.read_text())) for path in paths]


def format_report(report: EvalReport) -> str:
    """A short, human-readable summary + one line per case, for CLI printing."""
    lines = [
        f"{report.valid_count}/{report.total_count} valid "
        f"({report.valid_rate:.0%}), {report.review_passed_count}/{report.total_count} passed review"
    ]
    for case in report.cases:
        if not case.valid:
            lines.append(f"  FAIL  {case.name}: {case.error}")
        elif not case.review_passed:
            lines.append(f"  WARN  {case.name}: confidence={case.confidence:.2f} warnings={case.warnings}")
        else:
            lines.append(f"  OK    {case.name}: confidence={case.confidence:.2f}")
    return "\n".join(lines)
