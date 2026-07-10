"""Tests for autoedit/director/cache.py.

Pure, filesystem-only tests (a tmp_path cache dir, never real media/network).
"""

from __future__ import annotations

from autoedit.director.cache import cached_llm_client


class TestCachedLlmClient:
    def test_cache_miss_calls_through_and_returns_the_response(self, tmp_path):
        calls = []

        def client(brief):
            calls.append(brief)
            return {"ops": [{"tool": "cutter", "params": {"keep": ["s1"]}}], "confidence": 0.9}

        cached = cached_llm_client(client, tmp_path)
        result = cached({"features": {"a": 1}})

        assert result == {"ops": [{"tool": "cutter", "params": {"keep": ["s1"]}}], "confidence": 0.9}
        assert len(calls) == 1

    def test_cache_hit_on_an_identical_brief_never_calls_through_again(self, tmp_path):
        calls = []

        def client(brief):
            calls.append(brief)
            return {"ops": [], "confidence": 0.5}

        cached = cached_llm_client(client, tmp_path)
        cached({"features": {"a": 1}})
        cached({"features": {"a": 1}})
        cached({"features": {"a": 1}})

        assert len(calls) == 1

    def test_cache_miss_for_a_genuinely_different_brief(self, tmp_path):
        calls = []

        def client(brief):
            calls.append(brief)
            return {"ops": [], "confidence": 0.5}

        cached = cached_llm_client(client, tmp_path)
        cached({"features": {"a": 1}})
        cached({"features": {"a": 2}})

        assert len(calls) == 2

    def test_cache_key_is_independent_of_dict_key_order(self, tmp_path):
        calls = []

        def client(brief):
            calls.append(brief)
            return {"ops": [], "confidence": 0.5}

        cached = cached_llm_client(client, tmp_path)
        cached({"a": 1, "b": 2})
        cached({"b": 2, "a": 1})

        assert len(calls) == 1

    def test_a_raised_call_is_never_cached(self, tmp_path):
        calls = []

        def flaky_client(brief):
            calls.append(brief)
            raise RuntimeError("transient")

        cached = cached_llm_client(flaky_client, tmp_path)
        for _ in range(3):
            try:
                cached({"features": {"a": 1}})
            except RuntimeError:
                pass

        assert len(calls) == 3  # every call re-attempted; nothing false-cached from a failure

    def test_persists_across_separate_wrapper_instances_over_the_same_dir(self, tmp_path):
        calls = []

        def client(brief):
            calls.append(brief)
            return {"ops": [], "confidence": 0.5}

        cached_llm_client(client, tmp_path)({"features": {"a": 1}})
        cached_llm_client(client, tmp_path)({"features": {"a": 1}})  # a brand-new wrapper, same dir

        assert len(calls) == 1

    def test_creates_the_cache_directory_if_missing(self, tmp_path):
        cache_dir = tmp_path / "nested" / "cache"
        assert not cache_dir.exists()

        cached_llm_client(lambda brief: {"ops": [], "confidence": 0.5}, cache_dir)({"a": 1})

        assert cache_dir.exists()
        assert len(list(cache_dir.glob("*.json"))) == 1
