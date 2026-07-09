"""EditOp / EditPlan — the director's output contract.

An EditPlan is the ONLY thing an LLM (or the heuristic fallback) is allowed to
produce. It names tools by string; whether a tool actually exists is checked
against the tool manifest at validation/execution time, not here (this module
has no knowledge of the manifest, to keep it a pure data contract).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EditOp(BaseModel):
    """A single operation to hand to a subsystem, by tool name."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1, description="Name of the tool/subsystem to dispatch to.")
    params: dict[str, Any] = Field(default_factory=dict, description="Tool-specific parameters.")


class EditPlan(BaseModel):
    """A full, ordered plan of operations plus the director's confidence in it."""

    model_config = ConfigDict(extra="forbid")

    ops: list[EditOp] = Field(min_length=1, description="Ordered list of operations to execute.")
    confidence: float = Field(ge=0, le=1, description="Director's confidence in this plan, 0-1.")
