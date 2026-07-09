"""Tests for `director.openai_client.make_llm_client` — all HTTP is mocked, no real network calls."""

from __future__ import annotations

import json

import httpx
import pytest

from autoedit.director.llm import LLMClient
from autoedit.director.openai_client import make_llm_client

_BRIEF = {"features": {"aspect": 0.5625, "shots": []}, "style": {"cut_on_beat": False}, "tools": ["cutter"]}


def _chat_completions_response(content: str, *, status_code: int = 200, url: str = "https://api.example.com/v1/chat/completions") -> httpx.Response:
    """Build a real `httpx.Response` (bound to a request, so `raise_for_status` behaves) without any network I/O."""
    request = httpx.Request("POST", url)
    return httpx.Response(status_code, json={"choices": [{"message": {"content": content}}]}, request=request)


class TestMakeLlmClientReturnsAnLlmClient:
    def test_returns_a_callable(self) -> None:
        client = make_llm_client(api_key="sk-test", model="claude-sonnet-4-6", base_url="https://api.anthropic.com/v1")
        assert callable(client)

    def test_the_callable_matches_the_llmclient_type_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "autoedit.director.openai_client.httpx.post",
            lambda *a, **k: _chat_completions_response(json.dumps({"ops": [], "confidence": 0.0})),
        )
        client: LLMClient = make_llm_client(api_key="k", model="m", base_url="https://api.example.com/v1")
        result = client(_BRIEF)
        assert isinstance(result, dict)


class TestSuccessfulCall:
    def test_posts_to_the_chat_completions_endpoint_with_expected_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def fake_post(url, *, headers, json, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            captured["timeout"] = timeout
            return _chat_completions_response(json_lib.dumps({"ops": [], "confidence": 0.0}))

        import json as json_lib

        monkeypatch.setattr("autoedit.director.openai_client.httpx.post", fake_post)

        client = make_llm_client(api_key="sk-test", model="my-model", base_url="https://api.example.com/v1")
        client(_BRIEF)

        assert captured["url"] == "https://api.example.com/v1/chat/completions"
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        assert captured["json"]["model"] == "my-model"
        assert captured["json"]["temperature"] == pytest.approx(0.3)
        assert captured["json"]["messages"][0]["role"] == "system"

    def test_returns_the_parsed_json_content_as_a_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expected_plan = {"ops": [{"tool": "cutter", "params": {"keep": ["s1"]}}], "confidence": 0.9}
        monkeypatch.setattr(
            "autoedit.director.openai_client.httpx.post",
            lambda *a, **k: _chat_completions_response(json.dumps(expected_plan)),
        )

        client = make_llm_client(api_key="k", model="m", base_url="https://api.example.com/v1")
        result = client(_BRIEF)

        assert result == expected_plan

    def test_strips_a_trailing_slash_from_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return _chat_completions_response("{}")

        monkeypatch.setattr("autoedit.director.openai_client.httpx.post", fake_post)

        client = make_llm_client(api_key="k", model="m", base_url="https://api.example.com/v1/")
        client(_BRIEF)

        assert captured["url"] == "https://api.example.com/v1/chat/completions"

    def test_custom_temperature_is_forwarded_in_the_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def fake_post(url, **kwargs):
            captured.update(kwargs)
            return _chat_completions_response("{}")

        monkeypatch.setattr("autoedit.director.openai_client.httpx.post", fake_post)

        client = make_llm_client(api_key="k", model="m", base_url="https://api.example.com/v1", temperature=0.9)
        client(_BRIEF)

        assert captured["json"]["temperature"] == pytest.approx(0.9)


class TestErrorPropagation:
    """The director's `try/except` around the `llm` call is what should catch these — this module must NOT swallow them."""

    def test_a_network_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_post(*args, **kwargs):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr("autoedit.director.openai_client.httpx.post", fake_post)
        client = make_llm_client(api_key="k", model="m", base_url="https://api.example.com/v1")

        with pytest.raises(httpx.ConnectError):
            client(_BRIEF)

    def test_an_http_error_status_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "autoedit.director.openai_client.httpx.post",
            lambda *a, **k: _chat_completions_response("{}", status_code=401),
        )
        client = make_llm_client(api_key="bad-key", model="m", base_url="https://api.example.com/v1")

        with pytest.raises(httpx.HTTPStatusError):
            client(_BRIEF)

    def test_malformed_json_in_the_message_content_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "autoedit.director.openai_client.httpx.post",
            lambda *a, **k: _chat_completions_response("this is not valid json"),
        )
        client = make_llm_client(api_key="k", model="m", base_url="https://api.example.com/v1")

        with pytest.raises(json.JSONDecodeError):
            client(_BRIEF)

    def test_an_unexpected_response_shape_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
        monkeypatch.setattr(
            "autoedit.director.openai_client.httpx.post",
            lambda *a, **k: httpx.Response(200, json={"unexpected": "shape"}, request=request),
        )
        client = make_llm_client(api_key="k", model="m", base_url="https://api.example.com/v1")

        with pytest.raises((KeyError, IndexError)):
            client(_BRIEF)
