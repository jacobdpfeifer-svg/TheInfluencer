"""The stubbed LLM interface — the ONE seam where a real model would plug in.

Per `.cursor/rules/director.mdc`: "The LLM call sits behind a stubbed
interface so the whole pipeline runs and tests with AI disabled." `LLMClient`
is any callable `brief -> raw response`; the response is deliberately typed
as `Any` because a real model can return malformed JSON, a bare string,
`None`, or anything else — `director.validate.validate_plan` is what turns
that into a trustworthy `EditPlan` (or rejects it).

`stub_llm` is the default: it always returns a deliberately unusable
response, so `director.direct` always falls through to `heuristic_plan`
until a real client is wired in. This is what AGENTS.md means by "If the AI
is disabled, a heuristic planner produces a valid plan and the program still
works end to end" — "disabled" and "stubbed" are the same code path.
"""

from __future__ import annotations

from typing import Any, Callable

LLMClient = Callable[[dict[str, Any]], Any]


def stub_llm(brief: dict[str, Any]) -> Any:
    """Default LLM client: always returns an unusable response (AI disabled).

    Deliberately fails `EditPlan` validation (`ops` requires at least one
    op) so callers exercise the exact same fallback path a real model's
    malformed output would trigger.
    """
    del brief  # unused: the stub never actually looks at the brief.
    return {"ops": [], "confidence": 0.0}
