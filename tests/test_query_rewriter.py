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
