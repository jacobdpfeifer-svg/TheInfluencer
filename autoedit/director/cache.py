"""cached_llm_client — a deterministic, disk-backed cache around any LLMClient.

Wraps any `LLMClient` (a real model, or `stub_llm`) so an IDENTICAL brief
never pays for a second network call. This matters for two callers:

- `director.eval.run_eval` (the offline eval harness) is meant to be re-run
  often as prompts/templates change — without a cache, every re-run of every
  unchanged case still burns a real API call.
- `cli.make`'s `--cache-dir` flag, for the same reason during iteration.

Keyed on a stable hash of the brief's own JSON (sorted keys, so dict key
order never busts the cache) — nothing time- or request-id-based, so a
cache hit only ever happens for an EXACT repeat of a brief already seen, and
a changed brief is always a genuine cache miss, never a stale hit. A failed
call (the wrapped `llm` raises) is never cached, so a transient error can't
poison future calls with the same brief.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autoedit.director.llm import LLMClient


def cached_llm_client(client: LLMClient, cache_dir: str | Path) -> LLMClient:
    """Wrap `client` with a JSON-file cache under `cache_dir` (one file per unique brief)."""
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)

    def call(brief: dict[str, Any]) -> Any:
        cache_path = directory / f"{_brief_key(brief)}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        response = client(brief)
        cache_path.write_text(json.dumps(response))
        return response

    return call


def _brief_key(brief: dict[str, Any]) -> str:
    canonical = json.dumps(brief, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
