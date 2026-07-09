"""validate_plan — the ONLY gate a candidate EditPlan must pass before dispatch.

Per `.cursor/rules/director.mdc`: "Always validate the model's output
against the EditPlan Pydantic schema. Reject any op that names a tool absent
from the manifest or params outside schema." This does two layers of
checking, both required:

1. Structural: does `raw` even parse as an `EditPlan` (right shape/types)?
2. Semantic: does every op's `tool` exist in the manifest, and do its
   `params` match that tool's own params schema (`CutterParams`, etc.)?

Never touches an LLM or a Timeline — pure validation over already-received
data.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from autoedit.models.plan import EditPlan
from autoedit.subsystems import TOOL_PARAMS_MANIFEST


class PlanValidationError(Exception):
    """Raised when a candidate EditPlan fails schema, tool-manifest, or params validation."""


def validate_plan(raw: Any, *, params_manifest: dict[str, type] = TOOL_PARAMS_MANIFEST) -> EditPlan:
    """Parse and validate `raw` (arbitrary LLM output) into a trustworthy `EditPlan`.

    Raises `PlanValidationError` on any structural, tool-name, or params
    problem. Never raises anything else.
    """
    try:
        plan = EditPlan.model_validate(raw)
    except ValidationError as exc:
        raise PlanValidationError(f"malformed EditPlan: {exc}") from exc

    for op in plan.ops:
        if op.tool not in params_manifest:
            raise PlanValidationError(f"unknown tool {op.tool!r}: not in the tool manifest")
        params_model = params_manifest[op.tool]
        try:
            params_model.model_validate(op.params)
        except ValidationError as exc:
            raise PlanValidationError(f"invalid params for tool {op.tool!r}: {exc}") from exc

    return plan
