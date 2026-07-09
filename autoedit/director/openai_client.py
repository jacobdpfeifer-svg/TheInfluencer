"""openai_client — a raw-HTTP LLMClient for any OpenAI-compatible chat completions API.

Named for the request/response *shape* (OpenAI's chat completions format),
not the `openai` SDK: this module makes its own HTTP call via `httpx` rather
than depending on the `openai` or `anthropic` packages, per the audit's
"no SDK dependency" requirement. It happens to also work against
Anthropic's own OpenAI-SDK-compatible endpoint
(https://docs.anthropic.com/en/api/openai-sdk), which is why
`autoedit.cli`'s default `--base-url` points at it.

Per `.cursor/rules/director.mdc`: "Log the exact prompt + response on every
call for reproducibility" (done here at DEBUG level) and "Low temperature
for structural decisions" (the default here, 0.3). Validating the response
is `director.validate.validate_plan`'s job, not this module's — see
`autoedit.director.llm.LLMClient`'s docstring: the response type is `Any`.
Network errors, HTTP error statuses, and malformed JSON/response shapes are
all left to propagate as exceptions, which is exactly what `director.direct`
catches to fall back to `heuristic_plan`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from autoedit.director.llm import LLMClient
from autoedit.director.prompts import build_messages

logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_PATH = "/chat/completions"
# Low temperature for structural decisions (assignment, ordering) per
# `.cursor/rules/director.mdc`; higher is only appropriate for caption/text
# generation, which isn't what this call is for.
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_TIMEOUT_SEC = 60.0


def make_llm_client(
    api_key: str,
    model: str,
    base_url: str,
    temperature: float = _DEFAULT_TEMPERATURE,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
) -> LLMClient:
    """Build an `LLMClient` that POSTs a brief to `base_url`'s chat completions endpoint.

    The returned callable matches `director.llm.LLMClient` (`brief -> Any`):
    it builds the messages via `prompts.build_messages`, posts them, and
    returns the parsed JSON content of the model's reply — nothing more.
    """
    endpoint = base_url.rstrip("/") + _CHAT_COMPLETIONS_PATH
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def call(brief: dict[str, Any]) -> Any:
        messages = build_messages(brief)
        payload = {"model": model, "messages": messages, "temperature": temperature}
        logger.debug("openai_client: POST %s model=%s messages=%r", endpoint, model, messages)

        response = httpx.post(endpoint, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        logger.debug("openai_client: response=%r", data)

        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    return call
