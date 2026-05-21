"""Tests for SnippetExtractor — sentence-level snippet generation.

Uses `vector_store` and `make_memory` fixtures from tests/conftest.py.
"""
import numpy as np
import pytest


class TestExtractResult:
    def test_dataclass_shape(self):
        from smriti_memcore.snippet import ExtractResult
        r = ExtractResult(used_mode="auto")
        assert r.used_mode == "auto"
        assert r.fallback is False


class TestNoneMode:
    def test_clears_snippet_and_returns(self, vector_store, make_memory):
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("anything")
        m.snippet = "leftover from prior recall"
        extractor = SnippetExtractor(vector_store=vector_store)
        result = extractor.extract(m, ["q"], np.zeros(384), mode="none")
        assert m.snippet is None
        assert result.used_mode == "none"


class TestThresholdShortCircuit:
    def test_short_content_not_extracted(self, vector_store, make_memory):
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("short.")  # well below 300 chars
        m.snippet = "stale"
        extractor = SnippetExtractor(vector_store=vector_store, min_chars=300)
        extractor.extract(m, ["q"], np.zeros(384), mode="auto")
        # State-leak guard: snippet always cleared, even on short-circuit
        assert m.snippet is None

    def test_exactly_at_min_chars_still_short_circuits(self, vector_store, make_memory):
        """Spec §5.4: `≤ min_chars` short-circuits (inclusive)."""
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("x" * 300)
        extractor = SnippetExtractor(vector_store=vector_store, min_chars=300)
        extractor.extract(m, ["x"], np.zeros(384), mode="auto")
        assert m.snippet is None


class TestStateLeakGuard:
    def test_extract_always_clears_snippet_on_entry(self, vector_store, make_memory):
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("anything short")
        m.snippet = "stale from prior call"
        extractor = SnippetExtractor(vector_store=vector_store)
        # Even mode='none' must clear (spec §5.2)
        extractor.extract(m, [], np.zeros(384), mode="none")
        assert m.snippet is None


class TestAutoModeLexical:
    @pytest.fixture
    def long_memory(self, make_memory):
        # 4 sentences, well over 300 chars
        content = (
            "Smriti uses FAISS for vector search. "
            "The palace organises memories into rooms with weighted graph edges. "
            "Consolidation runs every five minutes by default. "
            "Migration v3 strips embeddings from palace.json to reduce file size."
        ) * 2  # make it long enough to exceed threshold
        return make_memory(content)

    def test_picks_positive_score_sentence(self, vector_store, long_memory):
        """A query with strong lexical overlap should yield a snippet with the matching sentence."""
        from smriti_memcore.snippet import SnippetExtractor
        extractor = SnippetExtractor(vector_store=vector_store)
        # Query overlaps with "FAISS" sentence
        extractor.extract(long_memory, ["FAISS vector search"], np.zeros(384), mode="auto")
        assert long_memory.snippet is not None
        assert "FAISS" in long_memory.snippet

    def test_picks_up_to_max_sentences(self, vector_store, long_memory):
        from smriti_memcore.snippet import SnippetExtractor
        extractor = SnippetExtractor(vector_store=vector_store, max_sentences=2)
        # Query overlaps with two distinct sentences
        extractor.extract(long_memory, ["FAISS consolidation"], np.zeros(384), mode="auto")
        # Snippet should be present and contain at most 2 source sentences
        assert long_memory.snippet is not None
        # Joined with " … " between non-adjacent picks
        # We can't assert exact text, but should reference both topics
        assert "FAISS" in long_memory.snippet or "consolidation" in long_memory.snippet.lower()

    def test_excludes_zero_score_sentences(self, vector_store, long_memory):
        """Spec §5.4 — only positive-score sentences picked; never zero-score filler."""
        from smriti_memcore.snippet import SnippetExtractor
        extractor = SnippetExtractor(vector_store=vector_store, max_sentences=2)
        # Query overlaps with only ONE sentence (palace)
        extractor.extract(long_memory, ["palace rooms"], np.zeros(384), mode="auto")
        assert long_memory.snippet is not None
        # No reference to unrelated sentences (FAISS, consolidation, migration)
        assert "FAISS" not in long_memory.snippet
        assert "consolidation" not in long_memory.snippet.lower()
        assert "Migration" not in long_memory.snippet

    def test_sentences_in_document_order(self, vector_store, make_memory):
        """Multiple picks must be re-ordered to original document order."""
        from smriti_memcore.snippet import SnippetExtractor
        content = (
            "ALPHA is the first interesting sentence about FAISS. "
            "Some boring middle filler sentence with nothing useful. "
            "Some more boring middle filler sentence with nothing useful. "
            "Some more boring middle filler sentence with nothing useful. "
            "OMEGA is the last interesting sentence about FAISS."
        )
        m = make_memory(content)
        # content is ~282 chars — use min_chars=200 to ensure extraction runs
        extractor = SnippetExtractor(vector_store=vector_store, max_sentences=2, min_chars=200)
        extractor.extract(m, ["FAISS"], np.zeros(384), mode="auto")
        # ALPHA must appear before OMEGA in the assembled snippet
        assert m.snippet is not None
        a = m.snippet.find("ALPHA")
        o = m.snippet.find("OMEGA")
        assert a >= 0 and o >= 0 and a < o


class TestCosineFallback:
    """Spec §5.5 — when no sentence shares any query token, pick the sentence with the
    highest cosine similarity to the raw-query embedding.

    Tests use MOCKED `vector_store.embed()` (spec §9) so behavior is deterministic and
    not dependent on the sentence-transformer model.
    """

    def _make_mock_vector_store(self, embedding_map: dict, monkeypatch):
        """Return a mock vector_store whose .embed(text) returns embedding_map[text]."""
        import numpy as np

        class MockVS:
            def embed(self, text):
                # Default to a zero vector if the test forgot to populate this string
                return np.array(embedding_map.get(text, [0.0] * 384), dtype=np.float32)
        return MockVS()

    def test_zero_overlap_uses_cosine_floor(self, make_memory, monkeypatch):
        """Mocked embeddings so we know exactly which sentence the cosine floor must pick."""
        import numpy as np
        from smriti_memcore.snippet import SnippetExtractor

        # Three sentences with NO lexical overlap to the query "xyzzy quux"
        s_a = "Apples grow on trees in the orchard."
        s_b = "Bananas are imported from tropical regions year-round."
        s_c = "Cherries ripen in late summer in the northern hemisphere."
        # Repeat 3x to push content well above 300-char threshold
        content = (s_a + " " + s_b + " " + s_c + " ") * 3
        m = make_memory(content)

        # Embeddings: query is closest to s_b (Bananas/tropical sentence).
        query_emb = np.array([1.0, 0.0, 0.0] + [0.0] * 381, dtype=np.float32)
        # Each sentence embedded by mock vector_store.embed() must yield a vector
        # whose dot product with query_emb gives the desired ordering.
        embedding_map = {
            s_a: [0.1, 0.0, 0.0] + [0.0] * 381,
            s_b: [0.9, 0.0, 0.0] + [0.0] * 381,   # closest
            s_c: [0.3, 0.0, 0.0] + [0.0] * 381,
        }
        mock_vs = self._make_mock_vector_store(embedding_map, monkeypatch)
        extractor = SnippetExtractor(vector_store=mock_vs)

        extractor.extract(m, ["xyzzy quux"], query_emb, mode="auto")
        assert m.snippet is not None
        # The mock guarantees s_b is the highest-cosine sentence
        assert "Bananas" in m.snippet

    def test_zero_overlap_returns_single_sentence(self, make_memory):
        """Cosine fallback returns exactly one sentence — not top-2."""
        import numpy as np
        from smriti_memcore.snippet import SnippetExtractor

        s_a = "Apples grow on trees in the orchard."
        s_b = "Bananas are imported from tropical regions year-round."
        content = (s_a + " " + s_b + " ") * 4
        m = make_memory(content)

        query_emb = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        class MockVS:
            def embed(self, text):
                return np.array([0.5] + [0.0] * 383, dtype=np.float32)
        extractor = SnippetExtractor(vector_store=MockVS(), max_sentences=2)
        extractor.extract(m, ["xyzzy"], query_emb, mode="auto")
        assert m.snippet is not None
        # No " … " separator means a single sentence was picked
        assert "…" not in m.snippet
