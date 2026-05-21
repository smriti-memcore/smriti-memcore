"""Tests for QueryRewriter — lexical + LLM query variant generation."""

import pytest


class TestExpandResult:
    def test_dataclass_shape(self):
        from smriti_memcore.query_rewriter import ExpandResult
        r = ExpandResult(variants=["q"], used_mode="auto")
        assert r.variants == ["q"]
        assert r.used_mode == "auto"
        assert r.fallback is False


class TestAutoMode:
    def test_raw_variant_always_first(self):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter()
        result = qr.expand("how do we handle WAL recovery")
        assert result.variants[0] == "how do we handle WAL recovery"
        assert result.used_mode == "auto"
        assert result.fallback is False

    def test_stop_stripped_variant_present(self):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter()
        result = qr.expand("how do we handle WAL recovery")
        # At least one variant should not contain "how" / "do" / "we" stop words
        non_raw = [v for v in result.variants if v != "how do we handle WAL recovery"]
        assert any("how" not in v.lower().split() for v in non_raw)

    def test_variants_deduped(self):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter()
        # A query with no stop words — stop-stripped == raw, content-words == raw, all collapse
        result = qr.expand("FAISS HNSW")
        assert len(result.variants) == len(set(result.variants))

    def test_variant_count_bounded(self):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter()
        result = qr.expand("how does smriti recall work")
        assert 1 <= len(result.variants) <= 3

    def test_empty_query(self):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter()
        result = qr.expand("")
        assert result.variants == [""]


class TestNoneMode:
    def test_passthrough(self):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter()
        result = qr.expand("how do we handle WAL recovery", mode="none")
        assert result.variants == ["how do we handle WAL recovery"]
        assert result.used_mode == "none"
        assert result.fallback is False


class TestLLMMode:
    """LLM mode uses LLMInterface.generate_json and caches results."""

    @pytest.fixture
    def fake_llm(self):
        """Minimal stub: records call count, returns 3 paraphrases."""
        class FakeLLM:
            def __init__(self):
                self.calls = 0
                # Match the real LLMInterface attribute name (llm_interface.py:42)
                self.default_model = "fake-model"
                self.fail = False
                self.return_value = ["paraphrase 1", "paraphrase 2", "paraphrase 3"]

            def generate_json(self, prompt, **kwargs):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("simulated LLM failure")
                return self.return_value

        return FakeLLM()

    def test_llm_called_once_per_unique_query(self, fake_llm):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter(llm=fake_llm)
        r1 = qr.expand("hard query", mode="llm")
        r2 = qr.expand("hard query", mode="llm")  # cache hit
        assert fake_llm.calls == 1
        assert r1.variants == r2.variants
        assert r1.fallback is False

    def test_llm_variants_include_raw_first(self, fake_llm):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter(llm=fake_llm)
        r = qr.expand("hard query", mode="llm")
        assert r.variants[0] == "hard query"
        assert "paraphrase 1" in r.variants

    def test_llm_failure_falls_back_to_auto(self, fake_llm):
        from smriti_memcore.query_rewriter import QueryRewriter
        fake_llm.fail = True
        qr = QueryRewriter(llm=fake_llm)
        r = qr.expand("how do we handle WAL", mode="llm")
        assert r.fallback is True
        assert r.used_mode == "auto"
        assert r.variants[0] == "how do we handle WAL"
        # auto path produces >= 1 variant
        assert len(r.variants) >= 1

    def test_llm_no_llm_configured_falls_back(self):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter(llm=None)
        r = qr.expand("anything", mode="llm")
        assert r.fallback is True
        assert r.used_mode == "auto"

    def test_llm_empty_response_falls_back(self, fake_llm):
        from smriti_memcore.query_rewriter import QueryRewriter
        fake_llm.return_value = []
        qr = QueryRewriter(llm=fake_llm)
        r = qr.expand("query", mode="llm")
        assert r.fallback is True

    def test_llm_malformed_response_falls_back(self, fake_llm):
        from smriti_memcore.query_rewriter import QueryRewriter
        fake_llm.return_value = "not a list"
        qr = QueryRewriter(llm=fake_llm)
        r = qr.expand("query", mode="llm")
        assert r.fallback is True

    def test_llm_filters_empty_and_duplicate_variants(self, fake_llm):
        from smriti_memcore.query_rewriter import QueryRewriter
        fake_llm.return_value = ["", "   ", "good paraphrase", "good paraphrase", "another"]
        qr = QueryRewriter(llm=fake_llm)
        r = qr.expand("raw query", mode="llm")
        assert "" not in r.variants
        assert r.variants.count("good paraphrase") == 1
        assert "good paraphrase" in r.variants
        assert "another" in r.variants

    def test_cache_key_includes_model_name(self, fake_llm):
        """Changing the LLM default_model must invalidate cached variants.

        LLMInterface attribute is `default_model` (llm_interface.py:42).
        """
        from smriti_memcore.query_rewriter import QueryRewriter
        # fake_llm fixture uses .default_model to match the real LLMInterface
        qr = QueryRewriter(llm=fake_llm)
        qr.expand("q", mode="llm")
        fake_llm.default_model = "different-model"
        qr.expand("q", mode="llm")
        assert fake_llm.calls == 2  # not a cache hit

    def test_cache_key_includes_prompt_version(self, fake_llm):
        """Bumping prompt_version invalidates cache."""
        from smriti_memcore.query_rewriter import QueryRewriter
        qr1 = QueryRewriter(llm=fake_llm, prompt_version="v1")
        qr2 = QueryRewriter(llm=fake_llm, prompt_version="v2")
        qr1.expand("q", mode="llm")
        qr2.expand("q", mode="llm")
        assert fake_llm.calls == 2

    def test_lru_eviction_when_cache_full(self, fake_llm):
        from smriti_memcore.query_rewriter import QueryRewriter
        qr = QueryRewriter(llm=fake_llm, cache_size=2)
        qr.expand("q1", mode="llm")        # 1 call
        qr.expand("q2", mode="llm")        # 2 calls
        qr.expand("q3", mode="llm")        # 3 calls — evicts q1
        qr.expand("q1", mode="llm")        # 4 calls — cache miss, LRU evicted
        qr.expand("q2", mode="llm")        # 4 still — q2 was evicted by q1
        assert fake_llm.calls == 5

    def test_llm_accepts_dict_with_variants_key(self, fake_llm):
        """The robust prompt asks for {'variants': [...]} — accept that shape."""
        from smriti_memcore.query_rewriter import QueryRewriter
        fake_llm.return_value = {"variants": ["paraphrase a", "paraphrase b"]}
        qr = QueryRewriter(llm=fake_llm)
        r = qr.expand("raw query", mode="llm")
        assert r.fallback is False
        assert r.used_mode == "llm"
        assert "paraphrase a" in r.variants
        assert "paraphrase b" in r.variants

    def test_llm_dict_without_variants_key_falls_back(self, fake_llm):
        """A dict without a 'variants' list key is treated as malformed."""
        from smriti_memcore.query_rewriter import QueryRewriter
        fake_llm.return_value = {"other_key": ["x", "y"]}
        qr = QueryRewriter(llm=fake_llm)
        r = qr.expand("raw query", mode="llm")
        assert r.fallback is True
        assert r.used_mode == "auto"

    def test_llm_error_dict_falls_back(self, fake_llm):
        """generate_json returns {'error': ...} on JSON parse failure — must fall back."""
        from smriti_memcore.query_rewriter import QueryRewriter
        fake_llm.return_value = {"error": "Failed to parse JSON"}
        qr = QueryRewriter(llm=fake_llm)
        r = qr.expand("raw query", mode="llm")
        assert r.fallback is True
        assert r.used_mode == "auto"
