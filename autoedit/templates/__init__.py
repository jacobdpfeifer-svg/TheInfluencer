"""templates — builtin timeline skeletons, loaded once into TEMPLATE_REGISTRY.

A `Template` (see `autoedit.models.template`) names a fixed slot structure.
`matcher.match_template` picks the best-fitting one for a given footage +
style, and `filler.fill_template` pours real shots into its slots to produce
an `EditPlan`. Both are deterministic — no LLM — so together they form a
richer "floor" the director's heuristic fallback stands on (see
`director/heuristic.py` and AGENTS.md's architecture section).
"""

from __future__ import annotations

import json
from pathlib import Path

from autoedit.models.template import Template

_BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"


def _load_builtin_templates() -> dict[str, Template]:
    registry: dict[str, Template] = {}
    for path in sorted(_BUILTIN_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        template = Template.model_validate(data)
        registry[template.name] = template
    return registry


# Loaded once, at import time, from every `*.json` file in `builtin/`.
TEMPLATE_REGISTRY: dict[str, Template] = _load_builtin_templates()


def load_template(name: str, *, registry: dict[str, Template] = TEMPLATE_REGISTRY) -> Template:
    """Look up a template by name. Raises `KeyError` (listing available names) on a miss."""
    try:
        return registry[name]
    except KeyError:
        raise KeyError(f"no template named {name!r}; available: {sorted(registry.keys())}") from None


def list_templates(*, registry: dict[str, Template] = TEMPLATE_REGISTRY) -> list[str]:
    """Names of every registered template, sorted."""
    return sorted(registry.keys())


__all__ = ["TEMPLATE_REGISTRY", "load_template", "list_templates"]
