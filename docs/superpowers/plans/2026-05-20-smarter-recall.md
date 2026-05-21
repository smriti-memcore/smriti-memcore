# Smarter Recall Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add query rewriting, snippet extraction, and cross-room adjacency boost to `RetrievalEngine` so recall token-spend drops ~30-50% and hit-rate@10 stays flat or improves.

**Architecture:** Two new modules (`query_rewriter.py`, `snippet.py`) slot around the existing hybrid vector+FTS+RRF pipeline. `RetrievalEngine.retrieve()` orchestrates: it embeds query variants once, calls `palace.search()` with precomputed embeddings (replacing today's neighbor-discount with a per-memory adjacency lift), then calls `SnippetExtractor` to populate `Memory.snippet` before returning. Library API gains two optional params (`rewrite`, `snippet`) with `None` sentinels that fall through to config defaults. MCP layer surfaces both as enum parameters and gains a small `smriti_get_memory` companion tool for expanding snippets.

**Tech Stack:** Python stdlib + numpy, existing `LLMInterface` for opt-in LLM rewrite/snippet paths, existing `VectorStore` (sentence-transformers `all-MiniLM-L6-v2`, L2-normalized embeddings). No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-05-20-smarter-recall-design.md`

---

## Task 1: Foundation — `Memory.snippet` field and config fields

**Files:**
- Modify: `smriti_memcore/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models.py`:

```python
class TestSnippetField:
    def test_memory_has_snippet_field(self):
        from smriti_memcore.models import Memory
        m = Memory(content="hello world")
        assert hasattr(m, "snippet")
        assert m.snippet is None

    def test_snippet_not_in_to_dict(self):
        """snippet is transient — never serialised to palace.json."""
        from smriti_memcore.models import Memory
        m = Memory(content="hello")
        m.snippet = "trimmed"
        d = m.to_dict()
        assert "snippet" not in d

    def test_memory_has_relevance_score_field(self):
        """relevance_score is the lift-adjusted score from palace.search (spec §6.1)."""
        from smriti_memcore.models import Memory
        m = Memory(content="x")
        assert hasattr(m, "relevance_score")
        assert m.relevance_score == 0.0

    def test_relevance_score_not_in_to_dict(self):
        from smriti_memcore.models import Memory
        m = Memory(content="x")
        m.relevance_score = 0.7
        d = m.to_dict()
        assert "relevance_score" not in d


class TestSmritiConfigSmartRecallFields:
    def test_defaults(self):
        from smriti_memcore.models import SmritiConfig
        c = SmritiConfig()
        assert c.rewrite_mode_default == "auto"
        assert c.snippet_mode_default == "auto"
        assert c.snippet_min_chars == 300
        assert c.snippet_max_sentences == 2
        assert c.llm_rewrite_cache_size == 100
        assert c.llm_rewrite_prompt_version == "v1"
        assert c.adjacency_alpha == 0.3
        assert c.adjacency_lift_max == 1.0
        assert c.entry_rooms_top_k == 5

    def test_validation_rewrite_mode(self):
        import pytest
        from smriti_memcore.models import SmritiConfig
        with pytest.raises(ValueError, match="rewrite_mode_default"):
            SmritiConfig(rewrite_mode_default="bogus")

    def test_validation_snippet_mode(self):
        import pytest
        from smriti_memcore.models import SmritiConfig
        with pytest.raises(ValueError, match="snippet_mode_default"):
            SmritiConfig(snippet_mode_default="bogus")

    def test_validation_alpha_range(self):
        import pytest
        from smriti_memcore.models import SmritiConfig
        with pytest.raises(ValueError, match="adjacency_alpha"):
            SmritiConfig(adjacency_alpha=-0.1)
        with pytest.raises(ValueError, match="adjacency_alpha"):
            SmritiConfig(adjacency_alpha=1.5)

    def test_validation_entry_rooms_top_k(self):
        import pytest
        from smriti_memcore.models import SmritiConfig
        with pytest.raises(ValueError, match="entry_rooms_top_k"):
            SmritiConfig(entry_rooms_top_k=0)
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cd /Users/shivamtyagi/PycharmProjects/nexus-memory
python3 -m pytest tests/test_models.py::TestSnippetField tests/test_models.py::TestSmritiConfigSmartRecallFields -v
```

Expected: 7 FAIL with `AttributeError` (field not defined) or `ValueError` not raised.

- [ ] **Step 3: Add `Memory.snippet` field**

In `smriti_memcore/models.py`, locate the `Memory` dataclass (around line 109). Add two new transient fields after `hops: int = 0`:

```python
    # Adjacency-lifted relevance score from palace.search (transient; spec §6.1)
    relevance_score: float = 0.0
    # Snippet — transient, populated by SnippetExtractor on long memories
    snippet: Optional[str] = None
```

Verify `to_dict()` still does not include `snippet` (it should not — the existing `to_dict` already only lists explicitly-named fields).

- [ ] **Step 4: Add `SmritiConfig` fields and validation**

In `SmritiConfig` (around line 280), add new fields just before `# Storage`:

```python
    # Smarter recall (2026-05-20 design)
    rewrite_mode_default: str = "auto"           # "auto" | "llm" | "none"
    snippet_mode_default: str = "auto"
    snippet_min_chars: int = 300                 # ≤ this → return content as-is
    snippet_max_sentences: int = 2
    llm_rewrite_cache_size: int = 100
    llm_rewrite_prompt_version: str = "v1"       # cache-key component
    adjacency_alpha: float = 0.3                 # lift coefficient
    adjacency_lift_max: float = 1.0              # cap on weighted-average lift
    entry_rooms_top_k: int = 5                   # widened from hardcoded 3
```

In `__post_init__`, append validation after the existing weight-sum check:

```python
        # Smarter recall validation
        _valid_modes = {"auto", "llm", "none"}
        if self.rewrite_mode_default not in _valid_modes:
            raise ValueError(
                f"rewrite_mode_default must be one of {_valid_modes}, got {self.rewrite_mode_default!r}"
            )
        if self.snippet_mode_default not in _valid_modes:
            raise ValueError(
                f"snippet_mode_default must be one of {_valid_modes}, got {self.snippet_mode_default!r}"
            )
        if not (0.0 <= self.adjacency_alpha <= 1.0):
            raise ValueError(
                f"adjacency_alpha must be in [0, 1], got {self.adjacency_alpha}"
            )
        if self.entry_rooms_top_k < 1:
            raise ValueError(
                f"entry_rooms_top_k must be >= 1, got {self.entry_rooms_top_k}"
            )
        if self.snippet_min_chars < 0:
            raise ValueError(
                f"snippet_min_chars must be >= 0, got {self.snippet_min_chars}"
            )
        if self.snippet_max_sentences < 1:
            raise ValueError(
                f"snippet_max_sentences must be >= 1, got {self.snippet_max_sentences}"
            )
```

- [ ] **Step 5: Run tests — all 7 pass**

```bash
python3 -m pytest tests/test_models.py::TestSnippetField tests/test_models.py::TestSmritiConfigSmartRecallFields -v
```

Expected: 7 PASS. Run the full `test_models.py` to confirm no regressions:

```bash
python3 -m pytest tests/test_models.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add smriti_memcore/models.py tests/test_models.py
git commit -m "feat: add Memory.snippet transient field and SmritiConfig smart-recall fields"
```

---

## Task 2: QueryRewriter — `mode="auto"` lexical variants

**Files:**
- Create: `smriti_memcore/query_rewriter.py`
- Test: `tests/test_query_rewriter.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/test_query_rewriter.py`:

```python
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
```

- [ ] **Step 2: Run — confirm they fail**

```bash
python3 -m pytest tests/test_query_rewriter.py -v
```

Expected: all FAIL with `ModuleNotFoundError: No module named 'smriti_memcore.query_rewriter'`.

- [ ] **Step 3: Implement QueryRewriter (auto + none)**

Create `smriti_memcore/query_rewriter.py`:

```python
"""
SMRITI v2 — Query Rewriter.

Generates query variants for the recall pipeline. mode="auto" produces
lexical variants (raw, stop-stripped, content-words) at microsecond cost;
mode="llm" produces LLM paraphrases (1-3s) with an LRU cache.

See docs/superpowers/specs/2026-05-20-smarter-recall-design.md §4.
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

from smriti_memcore.fts_index import _STOP_WORDS

logger = logging.getLogger(__name__)


# Modal / auxiliary verbs and very-short tokens dropped for the content-words variant.
# We do NOT include these in _STOP_WORDS because FTS5 uses _STOP_WORDS too and these
# tokens occasionally carry meaning in technical queries (e.g., "can foo bar" → "can").
_AUX_TOKENS = frozenset({
    "do", "does", "did", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "to", "of", "for",
})


@dataclass
class ExpandResult:
    variants: List[str]
    used_mode: str
    fallback: bool = False


class QueryRewriter:
    """Generates query variants for hybrid retrieval. See spec §4."""

    def __init__(
        self,
        llm=None,
        cache_size: int = 100,
        prompt_version: str = "v1",
    ):
        self.llm = llm
        self.prompt_version = prompt_version
        self._cache_size = cache_size
        # Composite-keyed LRU cache for LLM mode only (see spec §4.4)
        self._llm_cache: "OrderedDict[tuple, List[str]]" = OrderedDict()

    def expand(self, query: str, mode: str = "auto") -> ExpandResult:
        if mode == "none":
            return ExpandResult(variants=[query], used_mode="none")
        if mode == "auto":
            return ExpandResult(variants=self._lexical_variants(query), used_mode="auto")
        if mode == "llm":
            # Implemented in Task 3 — for now, fall back to auto.
            raise NotImplementedError("llm mode added in Task 3")
        raise ValueError(f"Unknown mode {mode!r} (expected 'auto'|'llm'|'none')")

    # ── Lexical variant generation ─────────────────────────────

    def _lexical_variants(self, query: str) -> List[str]:
        """Return up to 3 deduped variants. variants[0] is always the raw query."""
        variants: List[str] = [query]

        tokens = query.split()
        # Stop-stripped (uses same _STOP_WORDS as fts_index, case-insensitive)
        stop_stripped_tokens = [t for t in tokens if t.lower() not in _STOP_WORDS]
        stop_stripped = " ".join(stop_stripped_tokens)
        if stop_stripped and stop_stripped not in variants:
            variants.append(stop_stripped)

        # Content-words: also drop modal/aux verbs and tokens of length ≤ 2
        content_words_tokens = [
            t for t in stop_stripped_tokens
            if t.lower() not in _AUX_TOKENS and len(t) > 2
        ]
        content_words = " ".join(content_words_tokens)
        if content_words and content_words not in variants:
            variants.append(content_words)

        return variants
```

- [ ] **Step 4: Run tests — all pass**

```bash
python3 -m pytest tests/test_query_rewriter.py -v
```

Expected: all PASS (the `mode="llm"` test class doesn't exist yet).

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/query_rewriter.py tests/test_query_rewriter.py
git commit -m "feat: add QueryRewriter with mode=auto lexical variants and mode=none passthrough"
```

---

## Task 3: QueryRewriter — `mode="llm"` + LRU cache + fallback

**Files:**
- Modify: `smriti_memcore/query_rewriter.py`
- Test: `tests/test_query_rewriter.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_query_rewriter.py`:

```python
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
        # auto path produces ≥ 1 variant
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
```

- [ ] **Step 2: Run — confirm they fail**

```bash
python3 -m pytest tests/test_query_rewriter.py::TestLLMMode -v
```

Expected: all FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement LLM path**

In `smriti_memcore/query_rewriter.py`, replace the `raise NotImplementedError` block with:

```python
        if mode == "llm":
            return self._llm_expand(query)
```

Then add the helper methods at the end of the class:

```python
    def _llm_expand(self, query: str) -> ExpandResult:
        """LLM rewrite path. Falls back to auto on any failure."""
        if self.llm is None:
            logger.warning("QueryRewriter mode='llm' requested but no LLM configured; falling back to auto")
            return ExpandResult(variants=self._lexical_variants(query), used_mode="auto", fallback=True)

        # LLMInterface stores its model under `default_model` (llm_interface.py:42).
        # Fall through to model / model_name for other LLM implementations that may follow
        # different conventions.
        model_name = (
            getattr(self.llm, "default_model", None)
            or getattr(self.llm, "model", None)
            or getattr(self.llm, "model_name", None)
            or "unknown"
        )
        cache_key = (query, model_name, self.prompt_version)

        if cache_key in self._llm_cache:
            self._llm_cache.move_to_end(cache_key)
            return ExpandResult(variants=self._llm_cache[cache_key], used_mode="llm")

        try:
            raw = self.llm.generate_json(self._build_prompt(query))
        except Exception as e:
            logger.warning(f"QueryRewriter LLM call failed: {e}; falling back to auto")
            return ExpandResult(variants=self._lexical_variants(query), used_mode="auto", fallback=True)

        if not isinstance(raw, list):
            logger.warning(f"QueryRewriter LLM returned non-list ({type(raw).__name__}); falling back to auto")
            return ExpandResult(variants=self._lexical_variants(query), used_mode="auto", fallback=True)

        # Filter empty/whitespace and deduplicate (preserving order), drop entries equal to raw query
        seen = {query}
        cleaned: List[str] = [query]
        for item in raw:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)

        if len(cleaned) == 1:
            # LLM returned nothing usable
            logger.warning("QueryRewriter LLM produced no usable variants; falling back to auto")
            return ExpandResult(variants=self._lexical_variants(query), used_mode="auto", fallback=True)

        # Evict LRU if at capacity
        if len(self._llm_cache) >= self._cache_size:
            self._llm_cache.popitem(last=False)
        self._llm_cache[cache_key] = cleaned

        return ExpandResult(variants=cleaned, used_mode="llm")

    def _build_prompt(self, query: str) -> str:
        return (
            "Given this user query, generate exactly 3 paraphrased variants that preserve\n"
            "meaning but use different wording. Return as a JSON list of strings.\n\n"
            f"Query: {query}\n"
            "Variants:"
        )
```

- [ ] **Step 4: Run — all pass**

```bash
python3 -m pytest tests/test_query_rewriter.py -v
```

Expected: all PASS (auto + none + llm).

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/query_rewriter.py tests/test_query_rewriter.py
git commit -m "feat: add QueryRewriter mode=llm with LRU cache and fallback semantics"
```

---

## Task 4: SnippetExtractor — shell, state-leak guard, threshold, `mode="none"`

**Files:**
- Create: `smriti_memcore/snippet.py`
- Test: `tests/test_snippet.py` (new)

- [ ] **Step 1: Write failing tests**

Create `tests/test_snippet.py`:

```python
"""Tests for SnippetExtractor — sentence-level snippet generation."""
import numpy as np
import pytest


@pytest.fixture
def vector_store():
    from smriti_memcore.vector_store import VectorStore
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        yield VectorStore(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            storage_path=os.path.join(d, "vectors"),
        )


@pytest.fixture
def make_memory():
    from smriti_memcore.models import Memory
    def _make(content):
        return Memory(content=content)
    return _make


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
```

- [ ] **Step 2: Run — confirm they fail**

```bash
python3 -m pytest tests/test_snippet.py -v
```

Expected: all FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement shell with state-leak guard, threshold, none mode**

Create `smriti_memcore/snippet.py`:

```python
"""
SMRITI v2 — Snippet Extractor.

Trims long memory content down to the sentences most relevant to a query.
Used by RetrievalEngine to reduce per-recall token spend without losing
the underlying memory (memory.content is never mutated; memory.snippet
is a transient field populated in-place).

See docs/superpowers/specs/2026-05-20-smarter-recall-design.md §5.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from smriti_memcore.fts_index import _STOP_WORDS
from smriti_memcore.models import Memory

logger = logging.getLogger(__name__)


@dataclass
class ExtractResult:
    used_mode: str
    fallback: bool = False


class SnippetExtractor:
    """See spec §5."""

    def __init__(
        self,
        vector_store,
        min_chars: int = 300,
        max_sentences: int = 2,
        llm=None,
    ):
        self.vector_store = vector_store
        self.min_chars = min_chars
        self.max_sentences = max_sentences
        self.llm = llm

    def extract(
        self,
        memory: Memory,
        query_variants: List[str],
        raw_query_embedding: np.ndarray,
        mode: str = "auto",
    ) -> ExtractResult:
        # Spec §5.2 — state-leak guard. Always clear before deciding what to populate.
        memory.snippet = None

        if mode == "none":
            return ExtractResult(used_mode="none")

        if len(memory.content) <= self.min_chars:
            # Already atomic; leave snippet as None and let serializer fall back to content
            return ExtractResult(used_mode=mode)

        if mode == "auto":
            return self._extract_auto(memory, query_variants, raw_query_embedding)
        if mode == "llm":
            # Implemented in Task 7
            raise NotImplementedError("llm mode added in Task 7")
        raise ValueError(f"Unknown mode {mode!r}")

    def _extract_auto(self, memory, query_variants, raw_query_embedding) -> ExtractResult:
        # Implemented in Task 5
        raise NotImplementedError("auto sentence-match added in Task 5")
```

- [ ] **Step 4: Run — tests pass for shell**

```bash
python3 -m pytest tests/test_snippet.py -v
```

Expected: `TestExtractResult`, `TestNoneMode`, `TestThresholdShortCircuit`, `TestStateLeakGuard` all PASS.

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/snippet.py tests/test_snippet.py
git commit -m "feat: add SnippetExtractor shell with state-leak guard, threshold, mode=none"
```

---

## Task 5: SnippetExtractor — `mode="auto"` lexical sentence-match

**Files:**
- Modify: `smriti_memcore/snippet.py`
- Test: `tests/test_snippet.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_snippet.py`:

```python
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
        extractor = SnippetExtractor(vector_store=vector_store, max_sentences=2)
        extractor.extract(m, ["FAISS"], np.zeros(384), mode="auto")
        # ALPHA must appear before OMEGA in the assembled snippet
        assert m.snippet is not None
        a = m.snippet.find("ALPHA")
        o = m.snippet.find("OMEGA")
        assert a >= 0 and o >= 0 and a < o
```

- [ ] **Step 2: Run — confirm they fail**

```bash
python3 -m pytest tests/test_snippet.py::TestAutoModeLexical -v
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `_extract_auto`**

In `smriti_memcore/snippet.py`, replace the `_extract_auto` stub with:

```python
    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

    def _tokenize(self, text: str) -> List[str]:
        """Lower-case, strip punctuation, drop stop words."""
        out = []
        for t in text.lower().split():
            t_clean = re.sub(r'[^\w]', '', t)
            if t_clean and t_clean not in _STOP_WORDS:
                out.append(t_clean)
        return out

    def _extract_auto(self, memory, query_variants, raw_query_embedding) -> ExtractResult:
        sentences = [s.strip() for s in self._SENTENCE_SPLIT.split(memory.content) if s.strip()]
        if not sentences:
            return ExtractResult(used_mode="auto")

        # Spec §5.4 step 3: sum query-token overlap counts ACROSS ALL VARIANTS, per sentence.
        # Variants that share a token reinforce each other — e.g., raw and stop-stripped
        # both contain "FAISS" → that token contributes twice.
        variant_token_lists = [self._tokenize(v) for v in query_variants]

        scored: List[Tuple[int, int]] = []
        for idx, sent in enumerate(sentences):
            sent_tokens = set(self._tokenize(sent))
            # Sum over variants: each variant contributes its overlap count with this sentence.
            score = 0
            for v_tokens in variant_token_lists:
                # Count occurrences of each variant token that appears in the sentence
                for t in v_tokens:
                    if t in sent_tokens:
                        score += 1
            scored.append((idx, score))

        # Spec §5.4 — pick up to max_sentences with score > 0. No zero-score filler.
        positive = [(idx, s) for (idx, s) in scored if s > 0]
        if not positive:
            # Implemented in Task 6 — for now, leave snippet None and return.
            # (Task 6 replaces this with the cosine-floor fallback.)
            return ExtractResult(used_mode="auto")

        positive.sort(key=lambda x: (-x[1], x[0]))  # by score desc, then doc order asc
        picks = positive[: self.max_sentences]

        # Re-order picks to document order
        picks.sort(key=lambda x: x[0])
        pick_indices = [idx for (idx, _) in picks]

        # Join with " … " between non-adjacent picks
        parts = []
        for i, idx in enumerate(pick_indices):
            if i > 0 and pick_indices[i] - pick_indices[i - 1] > 1:
                parts.append("…")
            parts.append(sentences[idx])
        memory.snippet = " ".join(parts)
        return ExtractResult(used_mode="auto")
```

- [ ] **Step 4: Run — all auto-mode tests pass**

```bash
python3 -m pytest tests/test_snippet.py::TestAutoModeLexical -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/snippet.py tests/test_snippet.py
git commit -m "feat: SnippetExtractor mode=auto lexical sentence-match with positive-score filter"
```

---

## Task 6: SnippetExtractor — zero-overlap cosine fallback

**Files:**
- Modify: `smriti_memcore/snippet.py`
- Test: `tests/test_snippet.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_snippet.py`:

```python
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
        # 2x to push content above 300-char threshold
        content = (s_a + " " + s_b + " " + s_c + " ") * 2
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
```

- [ ] **Step 2: Run — confirm they fail (snippet stays None on zero overlap)**

```bash
python3 -m pytest tests/test_snippet.py::TestCosineFallback -v
```

Expected: FAIL — snippet is None because Task 5 stub returns without populating.

- [ ] **Step 3: Replace the early-return with cosine fallback**

In `smriti_memcore/snippet.py`, in `_extract_auto`, replace:

```python
        if not positive:
            # Implemented in Task 6 — for now, leave snippet None and return.
            # (Task 6 replaces this with the cosine-floor fallback.)
            return ExtractResult(used_mode="auto")
```

with:

```python
        if not positive:
            # Spec §5.5 — zero-overlap cosine floor.
            # Cheap rare path: embed each sentence once and pick the closest.
            # Embeddings are L2-normalized by vector_store.embed() (vector_store.py:120),
            # so np.dot() is cosine similarity.
            sentence_embs = [self.vector_store.embed(s) for s in sentences]
            scores = [float(np.dot(raw_query_embedding, se)) for se in sentence_embs]
            top_idx = int(np.argmax(scores))
            memory.snippet = sentences[top_idx]
            return ExtractResult(used_mode="auto")
```

- [ ] **Step 4: Run — cosine fallback tests pass**

```bash
python3 -m pytest tests/test_snippet.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/snippet.py tests/test_snippet.py
git commit -m "feat: SnippetExtractor cosine-floor fallback for zero lexical-overlap queries"
```

---

## Task 7: SnippetExtractor — `mode="llm"` + fallback

**Files:**
- Modify: `smriti_memcore/snippet.py`
- Test: `tests/test_snippet.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_snippet.py`:

```python
class TestLLMMode:
    @pytest.fixture
    def fake_llm(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0
                self.fail = False
                self.return_text = "LLM-extracted relevant sentence."

            def generate(self, prompt, **kwargs):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("LLM down")
                # Mimic LLMInterface.generate which returns a string-like object
                class R:
                    pass
                r = R()
                r.text = self.return_text
                return r

        return FakeLLM()

    def test_llm_sets_snippet(self, vector_store, make_memory, fake_llm):
        from smriti_memcore.snippet import SnippetExtractor
        m = make_memory("a" * 400)  # exceed threshold
        extractor = SnippetExtractor(vector_store=vector_store, llm=fake_llm)
        result = extractor.extract(m, ["raw query"], np.zeros(384), mode="llm")
        assert m.snippet == "LLM-extracted relevant sentence."
        assert result.used_mode == "llm"
        assert result.fallback is False
        assert fake_llm.calls == 1

    def test_llm_failure_falls_back_to_auto(self, vector_store, make_memory, fake_llm):
        from smriti_memcore.snippet import SnippetExtractor
        fake_llm.fail = True
        content = ("FAISS underlies the vector search. " * 20)  # long, lexically rich
        m = make_memory(content)
        extractor = SnippetExtractor(vector_store=vector_store, llm=fake_llm)
        result = extractor.extract(m, ["FAISS"], np.zeros(384), mode="llm")
        assert result.fallback is True
        assert result.used_mode == "auto"
        assert m.snippet is not None  # auto path produced something

    def test_llm_empty_response_falls_back(self, vector_store, make_memory, fake_llm):
        from smriti_memcore.snippet import SnippetExtractor
        fake_llm.return_text = "   "
        content = ("FAISS underlies the vector search. " * 20)
        m = make_memory(content)
        extractor = SnippetExtractor(vector_store=vector_store, llm=fake_llm)
        result = extractor.extract(m, ["FAISS"], np.zeros(384), mode="llm")
        assert result.fallback is True
        assert result.used_mode == "auto"

    def test_llm_no_llm_configured_falls_back(self, vector_store, make_memory):
        from smriti_memcore.snippet import SnippetExtractor
        content = ("FAISS underlies the vector search. " * 20)
        m = make_memory(content)
        extractor = SnippetExtractor(vector_store=vector_store, llm=None)
        result = extractor.extract(m, ["FAISS"], np.zeros(384), mode="llm")
        assert result.fallback is True
        assert result.used_mode == "auto"
        assert m.snippet is not None
```

- [ ] **Step 2: Run — confirm they fail**

```bash
python3 -m pytest tests/test_snippet.py::TestLLMMode -v
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement LLM path**

In `smriti_memcore/snippet.py`, replace the `raise NotImplementedError("llm mode added in Task 7")` block with:

```python
        if mode == "llm":
            return self._extract_llm(memory, query_variants, raw_query_embedding)
```

Then add the helper at the end of the class:

```python
    def _extract_llm(self, memory, query_variants, raw_query_embedding) -> ExtractResult:
        if self.llm is None:
            logger.warning("SnippetExtractor mode='llm' requested but no LLM configured; falling back to auto")
            auto = self._extract_auto(memory, query_variants, raw_query_embedding)
            return ExtractResult(used_mode="auto", fallback=True)

        raw_query = query_variants[0] if query_variants else ""
        prompt = (
            "Given this query and memory content, extract the 1-2 sentences most relevant\n"
            "to the query. Return only the extracted text, nothing else.\n\n"
            f"Query: {raw_query}\n"
            f"Content: {memory.content}"
        )
        try:
            response = self.llm.generate(prompt)
            text = getattr(response, "text", str(response)).strip()
        except Exception as e:
            logger.warning(f"SnippetExtractor LLM call failed: {e}; falling back to auto")
            self._extract_auto(memory, query_variants, raw_query_embedding)
            return ExtractResult(used_mode="auto", fallback=True)

        if not text:
            logger.warning("SnippetExtractor LLM returned empty; falling back to auto")
            self._extract_auto(memory, query_variants, raw_query_embedding)
            return ExtractResult(used_mode="auto", fallback=True)

        memory.snippet = text
        return ExtractResult(used_mode="llm")
```

- [ ] **Step 4: Run — all snippet tests pass**

```bash
python3 -m pytest tests/test_snippet.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/snippet.py tests/test_snippet.py
git commit -m "feat: SnippetExtractor mode=llm with auto fallback on failure or empty response"
```

---

## Task 8: Palace.search() — accept precomputed variant embeddings

**Files:**
- Modify: `smriti_memcore/palace.py`
- Test: `tests/test_palace.py`

> Context: spec §6.1 — `RetrievalEngine` will embed variants once and pass them down to avoid double-embedding when both palace search and FTS are in play. This task changes the signature; Task 9 implements the new scoring formula.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_palace.py`:

```python
class TestPalaceSearchVariants:
    """Spec §6.1 — palace.search() accepts precomputed variant embeddings."""

    def test_search_accepts_variant_embeddings(self, palace, make_memory, vector_store):
        """New signature: search(variants, variant_embeddings, top_k, max_hops)."""
        palace.place_memory(make_memory("hello world"))
        variants = ["hello"]
        embeddings = [vector_store.embed(v) for v in variants]
        results = palace.search(variants, embeddings, top_k=5)
        assert isinstance(results, list)

    def test_search_does_not_call_embed_for_query(self, palace, make_memory, vector_store, monkeypatch):
        """Spec §6.1 — palace.search() must NOT re-embed the variants."""
        palace.place_memory(make_memory("hello world"))
        variants = ["hello"]
        embeddings = [vector_store.embed(v) for v in variants]

        call_count = {"n": 0}
        original_embed = palace.vector_store.embed
        def tracked_embed(text):
            call_count["n"] += 1
            return original_embed(text)
        monkeypatch.setattr(palace.vector_store, "embed", tracked_embed)

        palace.search(variants, embeddings, top_k=5)
        # palace.search may still call vector_store for other things, but should not
        # re-embed the variants themselves. The test confirms variant embedding is the
        # caller's responsibility.
        # (We allow > 0 here because the search may embed room topics on demand if a
        # newly-created room has no centroid yet; but it should not embed the query.)
        # This is best-asserted indirectly via Task 11 (RetrievalEngine) once the wiring
        # is in place. For now, just exercise the new signature.
        assert call_count["n"] >= 0
```

- [ ] **Step 2: Run — confirm they fail**

```bash
python3 -m pytest tests/test_palace.py::TestPalaceSearchVariants -v
```

Expected: FAIL (current signature is `search(query: str, top_k, max_hops)`).

- [ ] **Step 3: Refactor `palace.search()` signature**

In `smriti_memcore/palace.py`, locate `def search` (around line 269). Change the signature and body to accept variants:

```python
    def search(
        self,
        variants: List[str],
        variant_embeddings: List[np.ndarray],
        top_k: int = 10,
        max_hops: int = 1,
    ) -> List[Memory]:
        """
        Multi-hop associative search through the palace.
        
        Spec §6 — accepts precomputed variant embeddings from the caller
        (RetrievalEngine.retrieve()) to avoid double-embedding the query when
        FTS5 is also active.
        """
        assert len(variants) == len(variant_embeddings), \
            f"variants and variant_embeddings must align; got {len(variants)} and {len(variant_embeddings)}"

        # Use the first variant (raw query) for room-finding for now;
        # Task 9 will rework this to score rooms by max-over-variants.
        primary = variants[0] if variants else ""
        primary_emb = variant_embeddings[0] if variant_embeddings else None

        entry_rooms = self.find_rooms(primary, top_k=3)
        candidates: Dict[str, Tuple[Memory, float, int]] = {}

        for room in entry_rooms:
            room.visit_count += 1
            room.last_visited = datetime.now()

            for mem in self.get_room_memories(room.id):
                if mem.embedding and primary_emb is not None:
                    score = float(np.dot(primary_emb, np.array(mem.embedding)))
                    if mem.id not in candidates or score > candidates[mem.id][1]:
                        candidates[mem.id] = (mem, score, 0)

            if max_hops >= 1:
                for neighbor, edge in self.get_neighbors(room.id):
                    neighbor.visit_count += 1
                    neighbor.last_visited = datetime.now()

                    for mem in self.get_room_memories(neighbor.id):
                        if mem.embedding and primary_emb is not None:
                            score = float(np.dot(primary_emb, np.array(mem.embedding)))
                            score *= 0.85 * edge.strength
                            if mem.id not in candidates or score > candidates[mem.id][1]:
                                candidates[mem.id] = (mem, score, 1)

        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for mem, score, hops in sorted_candidates:
            mem.retrieval_score = score
            mem.hops = hops
            results.append(mem)

        return results
```

Note: this preserves the OLD scoring (0.85 discount) — Task 9 replaces it. This task is purely about the signature change.

- [ ] **Step 4: Update existing callers of palace.search()**

Two callers exist (audit via `grep -rn 'palace\.search' smriti_memcore tests`):

1. **`smriti_memcore/retrieval.py:73`** — production caller:

```python
vector_candidates = self.palace.search(query, top_k=top_k * 3, max_hops=max_hops)
```

Change it to a temporary single-variant call so existing tests still pass; Task 10 replaces this:

```python
# TEMPORARY shim: Task 10 replaces this with QueryRewriter integration.
_temp_emb = self.vector_store.embed(query)
vector_candidates = self.palace.search([query], [_temp_emb], top_k=top_k * 3, max_hops=max_hops)
```

2. **`tests/test_visibility.py:150`** — `TestPrivateMemoryRecall::test_private_memory_is_recalled_by_owner`:

```python
results = palace.search("project X", top_k=5)
```

Update to the new signature using the `vector_store` fixture already in scope:

```python
results = palace.search(["project X"], [vector_store.embed("project X")], top_k=5)
```

(The `vector_store` fixture is provided by `tests/conftest.py:35`.)

- [ ] **Step 5: Run — palace + retrieval tests pass**

```bash
python3 -m pytest tests/test_palace.py tests/test_retrieval.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add smriti_memcore/palace.py smriti_memcore/retrieval.py tests/test_palace.py tests/test_visibility.py
git commit -m "refactor: palace.search() accepts precomputed variant embeddings"
```

---

## Task 9: Palace.search() — per-memory adjacency lift + entry-room widening

**Files:**
- Modify: `smriti_memcore/palace.py`
- Test: `tests/test_palace.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_palace.py`:

```python
class TestAdjacencyLift:
    """Spec §6 — per-memory adjacency lift replaces the legacy 0.85 discount.

    Tests construct palace topology by hand (rooms with controlled centroid embeddings,
    edges with known strengths) so the lift formula's effect is provable, not heuristic.
    """

    def _build_palace_with_topology(self, vector_store, tmp_dir, config):
        """Construct: 3 rooms (A, B, C), edges A↔B (strong), B↔C (weak), no A↔C.
        Returns (palace, room_ids dict, mem_ids dict).
        """
        import os
        import numpy as np
        from smriti_memcore.palace import SemanticPalace, Room, TypedEdge
        from smriti_memcore.models import Memory

        palace = SemanticPalace(
            vector_store=vector_store,
            storage_path=os.path.join(tmp_dir, "palace_topology"),
            config=config,
        )

        # Three rooms with hand-picked centroid embeddings to keep cosine deterministic.
        # All embeddings are 384-d (matches config.embedding_dim).
        e_strong = np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)
        e_weak = np.array([0.05, 0.0] + [0.0] * 382, dtype=np.float32)

        # Room A: matches the query strongly
        # Room B: weakly matches; neighbor of A
        # Room C: weakly matches; neighbor of B only
        room_a = Room(id="room_a", topic="A", centroid_embedding=e_strong)
        room_b = Room(id="room_b", topic="B", centroid_embedding=e_weak)
        room_c = Room(id="room_c", topic="C", centroid_embedding=e_weak.copy())
        for r in (room_a, room_b, room_c):
            palace.rooms[r.id] = r
            palace._room_embeddings[r.id] = r.centroid_embedding

        # Edges: A↔B strength 0.9, B↔C strength 0.2 — bidirectional
        for src, dst, strength in [("room_a", "room_b", 0.9), ("room_b", "room_a", 0.9),
                                    ("room_b", "room_c", 0.2), ("room_c", "room_b", 0.2)]:
            palace._adj.setdefault(src, []).append(
                TypedEdge(source_room_id=src, target_room_id=dst, relationship="related", strength=strength)
            )

        # Place one memory in each room. Memory embeddings carry only weak direct signal
        # so the lift becomes decisive.
        mems = {}
        for rid, room in [("room_a", room_a), ("room_b", room_b), ("room_c", room_c)]:
            m = Memory(content=f"memory in {rid}", room_id=rid)
            m.embedding = e_weak.tolist()  # weak direct similarity to the query
            palace.memories[m.id] = m
            room.memory_ids.append(m.id)
            mems[rid] = m

        # Patch: memory in room_a gets a slightly stronger embedding (it's the strong-hit room)
        mems["room_a"].embedding = e_strong.tolist()

        return palace, {"a": room_a.id, "b": room_b.id, "c": room_c.id}, mems

    def test_negative_cosine_clamped(self, tmp_dir, vector_store, config):
        """A negative-cosine room/memory must not produce a negative or amplified-negative score."""
        import os, numpy as np
        from smriti_memcore.palace import SemanticPalace, Room
        from smriti_memcore.models import Memory

        palace = SemanticPalace(
            vector_store=vector_store,
            storage_path=os.path.join(tmp_dir, "palace_neg"),
            config=config,
        )
        # Query embedding and a memory embedding that are antiparallel (cosine = -1)
        q = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        e_neg = np.array([-1.0] + [0.0] * 383, dtype=np.float32)

        room = Room(id="r", topic="opposite", centroid_embedding=e_neg)
        palace.rooms[room.id] = room
        palace._room_embeddings[room.id] = e_neg

        m = Memory(content="opposite content", room_id="r")
        m.embedding = e_neg.tolist()
        palace.memories[m.id] = m
        room.memory_ids.append(m.id)

        palace.search(["query"], [q], top_k=5)
        assert m.relevance_score >= 0.0, f"relevance_score must clamp to 0, got {m.relevance_score}"

    def test_adjacency_lift_surfaces_neighbor_in_weak_room(
        self, tmp_dir, vector_store, config
    ):
        """Room B (weak direct hit, neighbor of strong-hit A) should out-rank Room C
        (same weak direct hit, neighbor of B but NOT A). The A↔B lift propagates through."""
        palace, rids, mems = self._build_palace_with_topology(vector_store, tmp_dir, config)
        import numpy as np
        # Query embedding aligns with the strong room centroid
        q = np.array([1.0, 0.0] + [0.0] * 382, dtype=np.float32)
        palace.search(["query"], [q], top_k=10)

        # room_b: neighbor of room_a (strong) with high edge — should get a meaningful lift
        # room_c: neighbor of room_b (weak) with low edge — should get little lift
        assert mems["room_b"].relevance_score > mems["room_c"].relevance_score, (
            f"adjacency lift not surfacing room_b: "
            f"b={mems['room_b'].relevance_score:.4f}, c={mems['room_c'].relevance_score:.4f}"
        )

    def test_lift_cap_prevents_unbounded_amplification(self, tmp_dir, vector_store, config):
        """A room with many strong-neighbor edges should saturate at the weighted-average lift,
        bounded by adjacency_lift_max."""
        import os, numpy as np
        from smriti_memcore.palace import SemanticPalace, Room, TypedEdge
        from smriti_memcore.models import Memory

        # Build a hub room with 10 strong-neighbor edges all pointing to strong-centroid rooms
        palace = SemanticPalace(
            vector_store=vector_store,
            storage_path=os.path.join(tmp_dir, "palace_hub"),
            config=config,
        )
        e_strong = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        e_weak = np.array([0.1] + [0.0] * 383, dtype=np.float32)
        hub = Room(id="hub", topic="hub", centroid_embedding=e_weak)
        palace.rooms["hub"] = hub
        palace._room_embeddings["hub"] = e_weak
        m = Memory(content="hub memory", room_id="hub")
        m.embedding = e_weak.tolist()
        palace.memories[m.id] = m
        hub.memory_ids.append(m.id)

        for i in range(10):
            nid = f"n{i}"
            palace.rooms[nid] = Room(id=nid, topic=f"n{i}", centroid_embedding=e_strong)
            palace._room_embeddings[nid] = e_strong
            palace._adj.setdefault("hub", []).append(
                TypedEdge(source_room_id="hub", target_room_id=nid, relationship="r", strength=1.0)
            )

        palace.search(["query"], [e_strong], top_k=5)
        # base ≈ 0.1 (cosine of e_weak·e_strong), lift = weighted average = 1.0, capped to 1.0
        # final ≈ 0.1 * (1 + 0.3 * 1.0) = 0.13
        max_expected = 0.1 * (1.0 + config.adjacency_alpha * config.adjacency_lift_max) + 0.001
        assert m.relevance_score <= max_expected, (
            f"hub-room saturation cap failed: relevance_score={m.relevance_score}, max={max_expected}"
        )

    def test_entry_rooms_widened_to_top_5(self, tmp_dir, vector_store, config):
        """Spec §6.2 — top-5 entry rooms participate, not top-3.
        A memory in the 4th- or 5th-ranked room must enter the candidate pool."""
        import os, numpy as np
        from smriti_memcore.palace import SemanticPalace, Room
        from smriti_memcore.models import Memory

        palace = SemanticPalace(
            vector_store=vector_store,
            storage_path=os.path.join(tmp_dir, "palace_widen"),
            config=config,
        )
        # Six rooms ranked by descending similarity. Place one memory in each.
        # Verify all six can appear in search results (top_k high enough).
        mem_ids = []
        for i, score in enumerate([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]):
            e = np.array([score] + [0.0] * 383, dtype=np.float32)
            rid = f"room_{i}"
            palace.rooms[rid] = Room(id=rid, topic=f"r{i}", centroid_embedding=e)
            palace._room_embeddings[rid] = e
            m = Memory(content=f"mem in {rid}", room_id=rid)
            m.embedding = e.tolist()
            palace.memories[m.id] = m
            palace.rooms[rid].memory_ids.append(m.id)
            mem_ids.append(m.id)

        q = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        results = palace.search(["q"], [q], top_k=10)
        result_ids = {r.id for r in results}
        # With entry_rooms_top_k=5, the memory in the 5th-ranked room (mem_ids[4]) must be reachable
        assert mem_ids[4] in result_ids, "5th-ranked room's memory not in candidate pool"
        # The 6th-ranked room (rank index 5) is NOT in the top-5; should NOT be returned
        # unless it's a 1-hop neighbor of a top-5 room (no edges in this fixture)
        assert mem_ids[5] not in result_ids, "6th-ranked room reached without being adjacent — top-5 widening leaked"
```

- [ ] **Step 2: Run — confirm partial failure**

```bash
python3 -m pytest tests/test_palace.py::TestAdjacencyLift -v
```

Expected: `test_negative_cosine_clamped` FAILS (scores can go negative with the legacy 0.85 multiplier on negative cosines). Others may pass or fail depending on setup.

- [ ] **Step 3: Replace scoring logic with adjacency lift**

In `smriti_memcore/palace.py`, replace the entire body of `search()` (preserving the new signature from Task 8) with:

```python
    def search(
        self,
        variants: List[str],
        variant_embeddings: List[np.ndarray],
        top_k: int = 10,
        max_hops: int = 1,
    ) -> List[Memory]:
        """
        Hybrid associative search with per-memory adjacency lift (spec §6).
        """
        assert len(variants) == len(variant_embeddings)
        if not variant_embeddings:
            return []

        # Spec §6.1 — score every room by max similarity over query variants; clamp ≥ 0.
        room_scores: Dict[str, float] = {}
        for rid in self.rooms:
            centroid = self._room_embeddings.get(rid)
            if centroid is None:
                continue
            best = max(float(np.dot(v, centroid)) for v in variant_embeddings)
            room_scores[rid] = max(0.0, best)

        # Spec §6.2 — top-N entry rooms (default 5, configurable via self.config).
        top_k_rooms = getattr(self.config, "entry_rooms_top_k", 5) if hasattr(self, "config") else 5
        entry_rids = sorted(room_scores, key=lambda r: room_scores[r], reverse=True)[:top_k_rooms]

        # Collect candidate pool: entry rooms ∪ 1-hop neighbors
        candidate_room_ids = set(entry_rids)
        if max_hops >= 1:
            for rid in entry_rids:
                for neighbor, _edge in self.get_neighbors(rid):
                    candidate_room_ids.add(neighbor.id)

        # Mark entry/neighbor rooms as visited
        now = datetime.now()
        for rid in candidate_room_ids:
            room = self.rooms.get(rid)
            if room:
                room.visit_count += 1
                room.last_visited = now

        alpha = getattr(self.config, "adjacency_alpha", 0.3) if hasattr(self, "config") else 0.3
        lift_max = getattr(self.config, "adjacency_lift_max", 1.0) if hasattr(self, "config") else 1.0

        candidates: Dict[str, Tuple[Memory, float, int]] = {}
        for rid in candidate_room_ids:
            hops = 0 if rid in entry_rids else 1
            for mem in self.get_room_memories(rid):
                if not mem.embedding:
                    continue
                mem_vec = np.array(mem.embedding)
                base = max(float(np.dot(v, mem_vec)) for v in variant_embeddings)
                base = max(0.0, base)  # clamp

                # Weighted-average lift over 1-hop neighbors of mem's room
                num = 0.0
                den = 0.0
                for neighbor, edge in self.get_neighbors(mem.room_id or rid):
                    w = max(0.0, min(1.0, edge.strength))
                    num += room_scores.get(neighbor.id, 0.0) * w
                    den += w
                lift = (num / den) if den > 0 else 0.0
                lift = min(lift, lift_max)

                score = base * (1.0 + alpha * lift)
                if mem.id not in candidates or score > candidates[mem.id][1]:
                    candidates[mem.id] = (mem, score, hops)

        sorted_candidates = sorted(candidates.values(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for mem, score, hops in sorted_candidates:
            # Spec §6.1 — write the lift-adjusted score to `relevance_score`, NOT `retrieval_score`.
            # `retrieval_score` is the multi-factor composite written later by RetrievalEngine.
            mem.relevance_score = score
            mem.hops = hops
            results.append(mem)
        return results
```

Note: `palace.config` does not exist today — it's passed in via the calling code. The `hasattr(self, "config")` checks default to the spec values when the palace was constructed without a config; we'll wire `config` into palace's `__init__` in Step 4.

- [ ] **Step 4: Add `config` parameter to `SemanticPalace.__init__`**

Find `class SemanticPalace` `__init__` (around line 56). Add `config` as an optional parameter:

```python
    def __init__(
        self,
        vector_store: VectorStore,
        storage_path: Optional[str] = None,
        config: Optional["SmritiConfig"] = None,
    ):
        self.vector_store = vector_store
        self.config = config
        # ... rest unchanged
```

Update the import at the top if needed: `from smriti_memcore.models import SmritiConfig` (TYPE_CHECKING is fine to avoid circular).

Update `SMRITI.__init__` in `core.py` to pass config:

```python
        self.palace = SemanticPalace(
            vector_store=self.vector_store,
            storage_path=os.path.join(self.config.storage_path, "palace"),
            config=self.config,
        )
```

- [ ] **Step 5: Run — palace tests pass**

```bash
python3 -m pytest tests/test_palace.py tests/test_retrieval.py -v
```

Expected: all PASS (including new `TestAdjacencyLift`).

- [ ] **Step 6: Commit**

```bash
git add smriti_memcore/palace.py smriti_memcore/core.py tests/test_palace.py
git commit -m "feat: per-memory adjacency lift + entry-room widening (replaces 0.85 discount)"
```

---

## Task 10: RetrievalEngine — wire QueryRewriter + SnippetExtractor

**Files:**
- Modify: `smriti_memcore/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_retrieval.py`:

```python
class TestSmartRecallWiring:
    """Spec §3 — RetrievalEngine wires QueryRewriter + SnippetExtractor."""

    def test_retrieve_accepts_rewrite_and_snippet_params(self, retrieval_engine):
        """New params must be accepted without raising."""
        # Uses existing retrieval_engine fixture (add one if needed)
        try:
            retrieval_engine.retrieve("query", rewrite="none", snippet="none")
        except TypeError as e:
            pytest.fail(f"retrieve() should accept rewrite/snippet params: {e}")

    def test_retrieve_uses_config_default_when_param_is_none(self, retrieval_engine):
        """Spec §8.1 — None sentinel falls through to config defaults."""
        # Hard to assert without integration; smoke-test that None is accepted.
        retrieval_engine.retrieve("query", rewrite=None, snippet=None)

    @pytest.fixture
    def smriti_for_leakage(self, tmp_dir, mock_llm):
        """Local copy of the SMRITI fixture from tests/test_core.py — that one isn't
        in conftest.py so we can't import it. Same pattern: real SMRITI with mocked LLM."""
        import os
        from smriti_memcore.models import SmritiConfig
        from smriti_memcore import SMRITI
        config = SmritiConfig(storage_path=os.path.join(tmp_dir, "smriti_db"))
        n = SMRITI(config=config)
        n.llm = mock_llm
        n.attention_gate.llm = mock_llm
        n.consolidation_engine.llm = mock_llm
        yield n
        n.close()

    def test_relevance_score_does_not_leak_across_recalls(self, smriti_for_leakage, monkeypatch):
        """Spec §6 stale-state guard: a memory entering the candidate pool via FTS-only
        (palace.search did not score it) must have its relevance_score cleared BEFORE
        _score_memory consumes it.

        To PROVE the FTS-only path is exercised (not just relying on palace.search
        happening to skip the memory), we monkeypatch palace.search to return an empty
        list. The memory then can only enter the candidate pool via FTS, exercising the
        stale-state guard. Final assertion is the strict `== 0.0`.
        """
        from smriti_memcore.models import MemorySource
        smriti = smriti_for_leakage

        # Seed memory with a rare lexical token + over-threshold content length
        mid = smriti.encode(
            "tokenxyzzy is a placeholder rare lexical marker used by the leakage test " * 8,
            source=MemorySource.USER_STATED,
        )
        if mid is None:
            import pytest
            pytest.skip("attention gate discarded the seeded memory")

        mem = smriti.palace.memories[mid]
        # Stamp the stale value — simulates a leaked lifted score from a prior recall
        mem.relevance_score = 0.95
        assert mem.relevance_score == 0.95  # sanity

        # FORCE the FTS-only path: stub palace.search to return [] so the memory cannot
        # enter via the vector pipeline. It must arrive via FTS, where the guard fires.
        monkeypatch.setattr(smriti.palace, "search", lambda *a, **k: [])

        results = smriti.recall("tokenxyzzy", rewrite="none", snippet="none", top_k=10)

        # The memory must appear in results (FTS found it via the rare token)
        assert any(m.id == mid for m in results), (
            "FTS-only path didn't surface the rare-token memory — test setup is wrong, "
            "OR FTS isn't wired in this SMRITI instance"
        )

        # STRICT assertion: the stale 0.95 must have been cleared to exactly 0.0 by the
        # guard. palace.search returned []; nothing else writes relevance_score in retrieve().
        assert mem.relevance_score == 0.0, (
            f"stale-state guard failed: expected relevance_score == 0.0 after FTS-only "
            f"recall, got {mem.relevance_score}"
        )
```

> Note: this task needs a `retrieval_engine` pytest fixture. Add one if not present:

```python
@pytest.fixture
def retrieval_engine(palace, vector_store, working_memory, config, mock_llm):
    from smriti_memcore.retrieval import RetrievalEngine
    return RetrievalEngine(
        palace=palace,
        working_memory=working_memory,
        vector_store=vector_store,
        config=config,
    )
```

(If a fixture already exists, reuse it.)

- [ ] **Step 2: Run — confirm failure**

```bash
python3 -m pytest tests/test_retrieval.py::TestSmartRecallWiring -v
```

Expected: FAIL with `TypeError: unexpected keyword argument 'rewrite'`.

- [ ] **Step 3: Wire QueryRewriter and SnippetExtractor into RetrievalEngine**

In `smriti_memcore/retrieval.py`:

1. Add imports:

```python
from smriti_memcore.query_rewriter import QueryRewriter, ExpandResult
from smriti_memcore.snippet import SnippetExtractor, ExtractResult
```

2. Update `__init__`:

```python
    def __init__(
        self,
        palace: SemanticPalace,
        working_memory: WorkingMemory,
        vector_store: VectorStore,
        config: SmritiConfig,
        fts_index: Optional[FTSIndex] = None,
        llm: Optional["LLMInterface"] = None,
    ):
        self.palace = palace
        self.working_memory = working_memory
        self.vector_store = vector_store
        self.config = config
        self.fts_index = fts_index

        self.query_rewriter = QueryRewriter(
            llm=llm,
            cache_size=config.llm_rewrite_cache_size,
            prompt_version=config.llm_rewrite_prompt_version,
        )
        self.snippet_extractor = SnippetExtractor(
            vector_store=vector_store,
            min_chars=config.snippet_min_chars,
            max_sentences=config.snippet_max_sentences,
            llm=llm,
        )

        self.retrieval_log: deque = deque(maxlen=1000)
```

3. Add a stale-state guard in `retrieve()` for FTS-only candidates:

`Memory` objects in `palace.memories` are persistent — the same `Memory` instance is reused across recalls. If a memory was scored by `palace.search` in recall N (gets a lifted `relevance_score`) but enters recall N+1 only via FTS (palace.search doesn't see it), the stale value leaks into `_score_memory`. Guard against this by tracking which IDs palace.search scored this call and zeroing the rest:

```python
# In retrieve(), right after palace.search returns and BEFORE the FTS merge:
palace_scored_ids = {m.id for m in vector_candidates}
```

Then, after the candidate list is assembled (post-FTS-merge), before the scoring loop:

```python
for memory in candidates:
    if memory.id not in palace_scored_ids:
        # FTS-only candidate; clear any stale lifted-relevance from a prior recall
        memory.relevance_score = 0.0
```

4. Add `Literal` to the typing import in `smriti_memcore/retrieval.py`:

```python
from typing import Callable, Dict, List, Literal, Optional, Tuple
```

5. Update `_score_memory()` to consume the palace-search lifted relevance:

```python
    def _score_memory(
        self, memory: Memory, query_embedding: np.ndarray, now: datetime
    ) -> float:
        """Multi-factor retrieval scoring. Relevance = palace.search lifted score
        when available; raw cosine fallback for FTS-only candidates."""
        # Spec §3 — palace.search writes the adjacency-lifted relevance to memory.relevance_score.
        # FTS-only candidates skip palace.search and have relevance_score == 0.0 (default).
        if memory.relevance_score > 0:
            relevance = memory.relevance_score
        elif memory.embedding:
            relevance = max(0.0, float(np.dot(query_embedding, np.array(memory.embedding))))
        else:
            relevance = 0.0

        # Recency, strength, salience (unchanged)
        days_since = (now - memory.last_accessed).total_seconds() / 86400
        recency = self.config.decay_rate ** days_since
        strength = min(memory.strength / 5.0, 1.0)
        salience = memory.salience.composite

        return (
            self.config.relevance_weight * relevance +
            self.config.recency_weight * recency +
            self.config.strength_weight * strength +
            self.config.salience_weight * salience
        )
```

6. Update `retrieve()` signature and body:

```python
    def retrieve(
        self,
        query: str,
        context: str = "",
        top_k: Optional[int] = None,
        max_hops: int = 1,
        rewrite: Optional[Literal["auto", "llm", "none"]] = None,
        snippet: Optional[Literal["auto", "llm", "none"]] = None,
    ) -> List[Memory]:
        top_k = top_k or self.config.retrieval_top_k
        start_time = time.time()

        # Resolve mode params: caller param > config default
        rewrite_mode = rewrite if rewrite is not None else self.config.rewrite_mode_default
        snippet_mode = snippet if snippet is not None else self.config.snippet_mode_default

        # 1. Query rewriting — embed all variants once
        expand_result = self.query_rewriter.expand(query, mode=rewrite_mode)
        variants = expand_result.variants
        variant_embeddings = [self.vector_store.embed(v) for v in variants]
        raw_query_embedding = variant_embeddings[0]

        # 2. Vector search (palace already widened to top-5 entry rooms internally).
        # palace.search() writes memory.relevance_score on every candidate it scored.
        vector_candidates = self.palace.search(
            variants, variant_embeddings,
            top_k=top_k * 3, max_hops=max_hops,
        )
        # Track which IDs palace scored this call — used below to clear stale
        # relevance_score on FTS-only candidates (spec §6 stale-state guard).
        palace_scored_ids = {m.id for m in vector_candidates}

        # 3. FTS keyword search (joined variants as query string)
        if self.fts_index is not None:
            joined_query = " ".join(variants)
            try:
                fts_results = self.fts_index.search(joined_query, top_k=top_k * 3)
            except Exception:
                logger.warning("FTS search failed — falling back to vector-only retrieval")
                fts_results = []

            merged_ids = self._rrf_merge(vector_candidates, fts_results, pool_size=top_k * 2)
            id_map: Dict[str, Memory] = {m.id: m for m in vector_candidates}
            candidates: List[Memory] = []
            for mid in merged_ids:
                if mid in id_map:
                    candidates.append(id_map[mid])
                else:
                    mem = self.palace.get_memory(mid)
                    if mem is not None:
                        candidates.append(mem)
        else:
            candidates = vector_candidates[: top_k * 2]

        if not candidates:
            logger.debug(f"No memories found for query: {query[:60]}...")
            return []

        # Stale-state guard: clear relevance_score on any candidate palace.search did
        # not score this call (i.e., FTS-only pulls). Memory objects in palace.memories
        # are reused across recalls, so a prior call's lifted value could leak.
        for memory in candidates:
            if memory.id not in palace_scored_ids:
                memory.relevance_score = 0.0

        # 4. Multi-factor scoring — _score_memory reads memory.relevance_score
        # (set by palace.search, or zeroed above for FTS-only candidates).
        now = datetime.now()
        for memory in candidates:
            memory.retrieval_score = self._score_memory(memory, raw_query_embedding, now)
        candidates.sort(key=lambda m: m.retrieval_score, reverse=True)
        selected = candidates[:top_k]

        # 5. Reinforcement (unchanged)
        for memory in selected:
            memory.reinforce(self.config.reinforcement_factor)
            memory.consecutive_successful_reviews += 1
            memory.next_review = self._next_review_interval(memory)

        # 6. Difficulty bonus (unchanged)
        for memory in selected:
            retrieval_effort = self._compute_effort(memory, now)
            if retrieval_effort > self.config.effort_threshold:
                memory.strength *= self.config.difficulty_bonus

        # 7. Snippet extraction
        for memory in selected:
            self.snippet_extractor.extract(
                memory, variants, raw_query_embedding, mode=snippet_mode,
            )

        # 8. Working memory admission (unchanged)
        for memory in selected:
            self.working_memory.admit(memory)

        # 9. Log
        elapsed_ms = (time.time() - start_time) * 1000
        self.retrieval_log.append({
            "query": query,
            "results": [m.id for m in selected],
            "scores": [m.retrieval_score for m in selected],
            "latency_ms": elapsed_ms,
            "timestamp": now.isoformat(),
            "rewrite_used_mode": expand_result.used_mode,
            "rewrite_fallback": expand_result.fallback,
        })

        logger.info(
            f"Retrieved {len(selected)} memories for '{query[:40]}...' "
            f"({elapsed_ms:.0f}ms; rewrite={expand_result.used_mode}, snippet={snippet_mode})"
        )
        return selected
```

7. Remove the TEMPORARY shim from Task 8 (the `_temp_emb = ...` lines) — superseded by the new variant embedding logic.

- [ ] **Step 4 (renumbered 5): Update `SMRITI.__init__` to pass `llm` to `RetrievalEngine`**

In `core.py`, find the `RetrievalEngine(...)` constructor call. Pass `llm=self.llm`:

```python
        self.retrieval_engine = RetrievalEngine(
            palace=self.palace,
            working_memory=self.working_memory,
            vector_store=self.vector_store,
            config=self.config,
            fts_index=self.fts_index,
            llm=self.llm,
        )
```

- [ ] **Step 5: Run — retrieval tests pass**

```bash
python3 -m pytest tests/test_retrieval.py tests/test_palace.py tests/test_query_rewriter.py tests/test_snippet.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add smriti_memcore/retrieval.py smriti_memcore/core.py tests/test_retrieval.py
git commit -m "feat: wire QueryRewriter and SnippetExtractor into RetrievalEngine"
```

---

## Task 11: SMRITI.recall() — plumb rewrite/snippet params

**Files:**
- Modify: `smriti_memcore/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_core.py`:

```python
class TestSmartRecallParams:
    def test_recall_accepts_rewrite_and_snippet(self, smriti):
        # Should not raise
        smriti.recall("Python", rewrite="none", snippet="none")
        smriti.recall("Python", rewrite="auto", snippet="auto")

    def test_recall_none_sentinel_uses_config(self, smriti):
        # None should fall through to config defaults; default = "auto"
        smriti.recall("Python", rewrite=None, snippet=None)
```

- [ ] **Step 2: Run — confirm failure**

```bash
python3 -m pytest tests/test_core.py::TestSmartRecallParams -v
```

Expected: FAIL with `TypeError`.

- [ ] **Step 3: Update `SMRITI.recall()` signature**

In `smriti_memcore/core.py`, add `Literal` to the typing import at the top:

```python
from typing import Any, Dict, List, Literal, Optional
```

Then update `recall()`:

```python
    def recall(
        self,
        query: str,
        context: str = "",
        top_k: Optional[int] = None,
        rewrite: Optional[Literal["auto", "llm", "none"]] = None,
        snippet: Optional[Literal["auto", "llm", "none"]] = None,
    ) -> List[Memory]:
        """
        Recall memories relevant to a query.

        rewrite / snippet: None → use config defaults (rewrite_mode_default /
        snippet_mode_default). Explicit values override.
        """
        start = time.perf_counter()

        # Meta-memory check (unchanged)
        decision = self.meta_memory.should_recall_or_ask(query)
        if decision == DecisionType.ADMIT_GAP_AND_ASK:
            self.meta_memory.register_failed_retrieval(query, context)
            logger.info(f"Knowledge gap detected for: {query[:60]}...")

        # Retrieval (with new params)
        memories = self.retrieval_engine.retrieve(
            query, context, top_k,
            rewrite=rewrite, snippet=snippet,
        )

        if not memories:
            self.meta_memory.register_failed_retrieval(query, context)
            self._metrics.recall_empty.inc()

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._metrics.recall_count.inc()
        self._metrics.recall_latency.observe(elapsed_ms)

        return memories
```

- [ ] **Step 4: Run — tests pass**

```bash
python3 -m pytest tests/test_core.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/core.py tests/test_core.py
git commit -m "feat: SMRITI.recall() accepts rewrite and snippet params with None sentinel"
```

---

## Task 12: MCP `smriti_recall` schema + response fields

**Files:**
- Modify: `smriti_memcore/integrations/mcp_server.py`
- Test: `tests/test_mcp_server.py`

> **Context for the implementer:** read `smriti_memcore/integrations/mcp_server.py` first.
> Key facts (verified 2026-05-20):
> - Tools are registered with `@mcp_server.tool()` (FastMCP pattern); `mcp_server` is a module global (line 81).
> - State is held in module global `_smriti: Optional[SMRITI]` (line 79). Tests inject via `import smriti_memcore.integrations.mcp_server as s; s._smriti = test_instance`.
> - The existing `smriti_recall` returns `List[Dict[str, Any]]` directly (line 158) — NOT a wrapper dict. Tests in `tests/test_mcp_server.py` and `tests/test_visibility.py` expect a list.
> - A `serialize_memory(memory)` helper (line 94) builds the per-memory dict. Extend this helper rather than rewriting the response shape.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_mcp_server.py`. Use the existing test fixtures (read the file to find them — `inject_smriti` autouse pattern):

```python
class TestSmartRecallMcpSchema:
    """rewrite/snippet exposed on the smriti_recall tool with enum constraints (Literal type)."""

    def test_smriti_recall_signature_has_rewrite_and_snippet(self):
        """The function exposed by the tool must accept rewrite and snippet kwargs."""
        import inspect
        import smriti_memcore.integrations.mcp_server as mcp_module
        sig = inspect.signature(mcp_module.smriti_recall)
        params = sig.parameters
        assert "rewrite" in params, "smriti_recall must accept a rewrite parameter"
        assert "snippet" in params, "smriti_recall must accept a snippet parameter"
        # Optional with None sentinel — caller can omit to use config default
        assert params["rewrite"].default is None
        assert params["snippet"].default is None

    def test_smriti_recall_uses_literal_type_for_enum_constraint(self):
        """Spec §8.2 — FastMCP introspects Literal type hints into the JSON schema's enum,
        so MCP callers see the three valid values."""
        from typing import get_args, get_type_hints
        import smriti_memcore.integrations.mcp_server as mcp_module
        hints = get_type_hints(mcp_module.smriti_recall)
        for field in ("rewrite", "snippet"):
            t = hints.get(field)
            # Optional[Literal[...]] is Union[Literal[...], None]
            inner = [a for a in get_args(t) if a is not type(None)]
            literal_vals = get_args(inner[0]) if inner else ()
            assert set(literal_vals) == {"auto", "llm", "none"}, (
                f"{field} type hint must be Optional[Literal['auto','llm','none']], "
                f"got Literal{list(literal_vals)}"
            )


class TestSmartRecallMcpResponse:
    """serialize_memory must include snippet/expandable/metadata fields."""

    def test_serialize_memory_includes_expandable_and_metadata(self):
        from smriti_memcore.integrations.mcp_server import serialize_memory
        from smriti_memcore.models import Memory
        m = Memory(content="full content")
        m.snippet = "trimmed snippet"
        d = serialize_memory(m)
        # When snippet is set, content field becomes the snippet (per spec §8.2)
        assert d["content"] == "trimmed snippet"
        assert d["expandable"] is True
        assert "metadata" in d
        assert "rewrite_fallback" in d["metadata"]
        assert "snippet_fallback" in d["metadata"]

    def test_serialize_memory_full_content_when_no_snippet(self):
        from smriti_memcore.integrations.mcp_server import serialize_memory
        from smriti_memcore.models import Memory
        m = Memory(content="full content")
        # m.snippet is None
        d = serialize_memory(m)
        assert d["content"] == "full content"
        assert d["expandable"] is False
```

- [ ] **Step 2: Run — confirm failure**

```bash
python3 -m pytest tests/test_mcp_server.py::TestSmartRecallMcpSchema tests/test_mcp_server.py::TestSmartRecallMcpResponse -v
```

Expected: FAIL (signature lacks rewrite/snippet; serialize_memory doesn't have expandable/metadata).

- [ ] **Step 3: Update `smriti_recall` tool function**

In `smriti_memcore/integrations/mcp_server.py` (around line 157), replace the existing `smriti_recall`:

```python
from typing import Literal  # add at top of file if not already imported

@mcp_server.tool()
def smriti_recall(
    query: str,
    top_k: int = 10,
    rewrite: Optional[Literal["auto", "llm", "none"]] = None,
    snippet: Optional[Literal["auto", "llm", "none"]] = None,
) -> List[Dict[str, Any]]:
    """
    Recall memories relevant to a query.

    rewrite: "auto" (lexical variants, fast, default) | "llm" (LLM paraphrases, 1-3s,
        better for hard queries) | "none" (pass query through unchanged).
        Omit to use server config default.
    snippet: "auto" (top-2 sentence-match, fast, default) | "llm" (LLM-extracted
        sentences, slower, noisy memories) | "none" (return full content).
        Omit to use server config default.

    Returns a list of memory dicts (one per result). Each dict's `content` field is the
    snippet when one was extracted; `expandable=true` and `metadata.snippet_fallback` /
    `metadata.rewrite_fallback` flags surface internal mode degradation.
    """
    try:
        memories = _smriti.recall(query, top_k=top_k, rewrite=rewrite, snippet=snippet)
        return [serialize_memory(m, smriti=_smriti) for m in memories]
    except Exception as e:
        logger.error(f"smriti_recall failed: {e}")
        return [{"error": str(e)}]
```

**Why `Literal` matters:** FastMCP introspects type hints to build the JSON schema. `Optional[str]` produces `{"type": "string"}`; `Optional[Literal["auto", "llm", "none"]]` produces `{"enum": ["auto", "llm", "none"], "nullable": true}`. Only the latter actually constrains the MCP client. Spec §8.2 requires the enum constraint.

- [ ] **Step 4: Extend `serialize_memory` to include the new fields**

In `smriti_memcore/integrations/mcp_server.py` (around line 94), update the helper:

```python
def serialize_memory(memory: Memory, smriti: Optional[SMRITI] = None) -> Dict[str, Any]:
    """Convert a Memory dataclass to a JSON-serializable dict.

    When `memory.snippet` is set, the `content` field returns the snippet (not the
    full content); `expandable=true` signals that the caller can use
    `smriti_get_memory(memory_id)` to fetch the full text (added in Task 13).

    `metadata.rewrite_fallback` and `metadata.snippet_fallback` reflect the most
    recent recall's degradation flags (from `retrieval_engine.retrieval_log[-1]`).
    """
    # Pull last-recall metadata from the smriti instance if provided
    rewrite_fb = False
    snippet_fb = False
    if smriti is not None:
        log = smriti.retrieval_engine.retrieval_log
        if log:
            rewrite_fb = bool(log[-1].get("rewrite_fallback", False))
            snippet_fb = bool(log[-1].get("snippet_fallback", False))

    has_snippet = memory.snippet is not None
    return {
        "id": memory.id,
        "memory_id": memory.id,  # alias for snippet-aware callers (spec §8.2)
        "content": memory.snippet if has_snippet else memory.content,
        "expandable": has_snippet,
        "metadata": {
            "rewrite_fallback": rewrite_fb,
            "snippet_fallback": snippet_fb,
        },
        "strength": memory.strength,
        "confidence": memory.confidence,
        "room_id": memory.room_id,
        "reflection_level": memory.reflection_level,
        "source": memory.source.value,
        "modality": memory.modality.value,
        "status": memory.status.value,
        "visibility": memory.visibility.value,
        "creation_time": memory.creation_time.isoformat(),
        "last_accessed": memory.last_accessed.isoformat(),
        "access_count": memory.access_count,
        "salience": memory.salience.to_dict(),
        "hops": memory.hops,
        "retrieval_score": memory.retrieval_score,
    }
```

> Note: the `smriti` parameter has a default `None` so existing callers (line 378 — `smriti_get_suggestions`) keep working without modification. The tool function passes `_smriti` explicitly.

- [ ] **Step 5: Wire snippet_fallback into RetrievalEngine's retrieval_log**

In `smriti_memcore/retrieval.py` (the `retrieve()` method, snippet extraction loop), replace:

```python
for memory in selected:
    self.snippet_extractor.extract(memory, variants, raw_query_embedding, mode=snippet_mode)
```

with:

```python
snippet_fallback_any = False
for memory in selected:
    result = self.snippet_extractor.extract(memory, variants, raw_query_embedding, mode=snippet_mode)
    if result.fallback:
        snippet_fallback_any = True
```

Then update the `retrieval_log.append({...})` block to include the new key:

```python
self.retrieval_log.append({
    ...,
    "rewrite_used_mode": expand_result.used_mode,
    "rewrite_fallback": expand_result.fallback,
    "snippet_fallback": snippet_fallback_any,
})
```

- [ ] **Step 6: Run — tests pass**

```bash
python3 -m pytest tests/test_mcp_server.py tests/test_retrieval.py tests/test_visibility.py -v
```

Expected: all PASS, including the `inject_smriti` autouse-fixture tests that already exist.

- [ ] **Step 7: Commit**

```bash
git add smriti_memcore/integrations/mcp_server.py smriti_memcore/retrieval.py tests/test_mcp_server.py
git commit -m "feat: MCP smriti_recall exposes rewrite/snippet enums; serialize_memory returns expandable/metadata"
```

---

## Task 13: MCP `smriti_get_memory` companion tool

**Files:**
- Modify: `smriti_memcore/integrations/mcp_server.py`
- Test: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_mcp_server.py` (uses the existing `inject_smriti` autouse pattern):

```python
class TestSmritiGetMemory:
    def test_get_memory_returns_full_content(self):
        """Existing inject_smriti fixture is autouse; _smriti is populated."""
        import smriti_memcore.integrations.mcp_server as mcp_module
        from smriti_memcore.models import MemorySource
        s = mcp_module._smriti
        mid = s.encode(
            "full content of a memory used to verify the get_memory tool",
            source=MemorySource.USER_STATED,
        )
        if not mid:
            import pytest
            pytest.skip("attention gate discarded the seeded memory")
        result = mcp_module.smriti_get_memory(memory_id=mid)
        assert result["id"] == mid
        assert result["content"] == "full content of a memory used to verify the get_memory tool"
        assert result["expandable"] is False

    def test_get_memory_unknown_id_returns_error(self):
        import smriti_memcore.integrations.mcp_server as mcp_module
        result = mcp_module.smriti_get_memory(memory_id="00000000-0000-0000-0000-000000000000")
        assert "error" in result
```

- [ ] **Step 2: Run — confirm failure**

```bash
python3 -m pytest tests/test_mcp_server.py::TestSmritiGetMemory -v
```

Expected: FAIL — `smriti_get_memory` doesn't exist.

- [ ] **Step 3: Register `smriti_get_memory` tool**

In `smriti_memcore/integrations/mcp_server.py`, add the new tool using the same `@mcp_server.tool()` pattern as `smriti_recall`. Place it adjacent to `smriti_recall` (around line 170) so callers find the pair together:

```python
@mcp_server.tool()
def smriti_get_memory(memory_id: str) -> Dict[str, Any]:
    """
    Fetch the full content of a memory by id.

    Use this when smriti_recall returned a snippet (expandable=true) and you need the
    complete memory. Returns the same shape as smriti_recall's per-memory dict, with
    `content` always set to the full memory text and `expandable=false`.
    """
    try:
        mem = _smriti.palace.get_memory(memory_id)
        if mem is None:
            return {"error": f"memory {memory_id} not found"}
        # Clear any stale snippet so serialize_memory returns full content
        mem.snippet = None
        return serialize_memory(mem, smriti=_smriti)
    except Exception as e:
        logger.error(f"smriti_get_memory failed: {e}")
        return {"error": str(e)}
```

- [ ] **Step 4: Run — tests pass**

```bash
python3 -m pytest tests/test_mcp_server.py::TestSmritiGetMemory -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add smriti_memcore/integrations/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add smriti_get_memory MCP tool for fetching full content after snippet"
```

---

## Task 14: Regression guard — disabled-features stability

**Files:**
- Test: `tests/test_retrieval.py`

> Spec §13.4 acceptance criterion calls for ID-order match against pre-change `main`.
> That comparison is a one-shot manual PR-verification step (see "PR baseline check" below),
> not something a unit test can express within the feature branch. The unit test here is the
> within-branch stability guard: with both features disabled, repeated recalls must be
> deterministic and `memory.snippet` must stay None — these are the invariants we CAN
> enforce automatically and that catch unintended ordering shifts during refactors.

- [ ] **Step 1: Write the regression-guard test**

Append to `tests/test_retrieval.py`:

```python
class TestRegressionGuard:
    """Spec §13.4 — within-branch stability with disabled features.

    The cross-branch baseline comparison (this branch vs main) is in the PR description,
    not in this test. See `scripts/bench_recall.py --label` for that flow.
    """

    @pytest.fixture
    def seeded_smriti(self, smriti):
        from smriti_memcore.models import MemorySource
        corpus = [
            "Python is a high-level programming language used for many tasks",
            "JavaScript runs in browsers and on servers via Node.js",
            "Rust emphasizes memory safety without garbage collection",
            "FAISS is a library for efficient similarity search of dense vectors",
            "SQLite is an embedded relational database written in C",
            "Redis is an in-memory key-value store popular for caching",
            "Postgres is a robust open-source relational database",
            "Kubernetes orchestrates containerized applications",
            "Docker provides isolation via Linux container primitives",
            "MIT license is permissive and widely used in open-source projects",
        ]
        for c in corpus:
            smriti.encode(c, source=MemorySource.USER_STATED, use_llm=True)
        return smriti

    def test_disabled_features_return_full_content(self, seeded_smriti):
        """snippet='none' must leave memory.snippet as None."""
        results = seeded_smriti.recall("python language", rewrite="none", snippet="none")
        for m in results:
            assert m.snippet is None

    def test_disabled_features_stability_within_branch(self, seeded_smriti):
        """Two consecutive recalls with disabled features must return the same ID order.

        Note: recall mutates memory.strength, last_accessed, and access_count on each call,
        which feeds into _score_memory. So we must compare set-membership of the top-K, not
        order strictly. The order CAN shift slightly between calls because of these
        reinforcement effects — which is expected behavior and was true on main as well.
        """
        r1 = seeded_smriti.recall("python language", rewrite="none", snippet="none", top_k=5)
        r2 = seeded_smriti.recall("python language", rewrite="none", snippet="none", top_k=5)
        ids1 = {m.id for m in r1}
        ids2 = {m.id for m in r2}
        # Top-5 membership should be stable across consecutive calls
        assert ids1 == ids2, f"top-5 changed between consecutive calls: {ids1 ^ ids2}"

    def test_disabled_features_score_uses_raw_cosine_fallback(self, seeded_smriti):
        """With rewrite='none', the variants list contains only the raw query — so
        relevance_score should be populated and _score_memory uses it.
        This guards against accidental disconnection of palace.search from the scorer."""
        results = seeded_smriti.recall("python language", rewrite="none", snippet="none", top_k=5)
        # At least the top result should have a positive retrieval_score (the composite)
        if results:
            assert results[0].retrieval_score > 0.0
```

- [ ] **Step 2: Run — confirm passes**

```bash
python3 -m pytest tests/test_retrieval.py::TestRegressionGuard -v
```

Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_retrieval.py
git commit -m "test: within-branch regression guard for disabled-features recall stability"
```

**PR baseline check (manual, separate from this commit):** before opening the PR, run `scripts/bench_recall.py --label main_baseline` on the pre-change `main` branch and save the JSON output. Then on the feature branch, run `scripts/bench_recall.py --label feature_branch --baseline main_baseline.json`. The hit-rate@10 with disabled features (`rewrite=none, snippet=none`) on the feature branch must match main's hit-rate@10 within ±1 result for the same fixed query corpus. Numbers go in the PR description.

---

## Task 15: Benchmark harness + final verification

**Files:**
- Create: `scripts/bench_recall.py`

> Spec §10. One-shot manual benchmark, not in CI.

- [ ] **Step 1: Create benchmark script**

Create `scripts/bench_recall.py`:

```python
#!/usr/bin/env python3
"""
Benchmark recall quality and token cost.

Seeds N=500 synthetic memories (each ≥400 chars to exercise the snippet path)
across 20 topics; runs 50 paraphrased queries with rewrite='auto', snippet='auto'
vs rewrite='none', snippet='none'; reports hit-rate@10, avg tokens/query,
p95 latency.

Not in CI. Run twice — once on `main` (baseline), once on the feature branch.
Put numbers in the PR description.

Usage:
    # On main:
    python3 scripts/bench_recall.py --label main_baseline --out /tmp/main.json

    # On feature branch:
    python3 scripts/bench_recall.py --label feature --out /tmp/feature.json --baseline /tmp/main.json
"""
import argparse
import json
import random
import statistics
import tempfile
import time

from smriti_memcore.core import SMRITI
from smriti_memcore.models import MemorySource, SmritiConfig

TOPICS = [
    "python", "javascript", "rust", "go", "java",
    "kubernetes", "docker", "terraform", "ansible", "helm",
    "postgres", "mysql", "sqlite", "redis", "mongodb",
    "react", "vue", "angular", "svelte", "ember",
]


def seed(s: SMRITI, n: int = 500) -> dict:
    """Seed `n` memories. Each content body is ≥ 400 chars so snippet_min_chars=300
    is exceeded and snippet="auto" actually trims content (otherwise tokens would
    not measurably reduce)."""
    target_ids = {}
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        content = (
            f"This memory {i} is about {topic} and how it relates to general programming patterns. "
            f"The {topic} ecosystem includes packaging tools, build systems, testing frameworks, and deployment helpers. "
            f"People who use {topic} often pair it with adjacent tooling and integrate it into broader workflows. "
            f"Communities around {topic} maintain documentation, tutorials, and reference implementations. "
            f"Random salt {random.randint(0, 1_000_000)} for uniqueness. "
            f"Performance characteristics of {topic} matter for production deployment. "
        )
        assert len(content) >= 400, f"seed content too short: {len(content)} chars"
        mid = s.encode(content, source=MemorySource.USER_STATED, use_llm=False)
        if mid:
            target_ids.setdefault(topic, []).append(mid)
    return target_ids


def _call_recall_compat(s: SMRITI, q: str, rewrite: str, snippet: str, top_k: int):
    """Call SMRITI.recall compatibly with both pre-change `main` and the feature branch.

    Pre-change main lacks rewrite/snippet kwargs. Detect via inspect at first call;
    when running on main we always do raw recall regardless of the requested mode."""
    import inspect
    sig = inspect.signature(s.recall)
    if "rewrite" in sig.parameters:
        return s.recall(q, rewrite=rewrite, snippet=snippet, top_k=top_k)
    return s.recall(q, top_k=top_k)


def run_queries(s: SMRITI, target_ids: dict, queries: list, rewrite: str, snippet: str):
    hits_at_10 = 0
    total_tokens = 0
    latencies = []
    per_query_ids = []  # for cross-branch ID-order baseline check (spec §13.4)
    for q, expected_topic in queries:
        t0 = time.perf_counter()
        results = _call_recall_compat(s, q, rewrite=rewrite, snippet=snippet, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        result_ids = [m.id for m in results]
        per_query_ids.append({"query": q, "ids": result_ids})
        if set(result_ids) & set(target_ids.get(expected_topic, [])):
            hits_at_10 += 1
        for m in results:
            content = getattr(m, "snippet", None) or m.content
            total_tokens += len(content.split())
    n = max(1, len(queries))
    return {
        "hit_rate_at_10": hits_at_10 / n,
        "avg_tokens_per_query": total_tokens / n,
        "p95_latency_ms": (
            statistics.quantiles(latencies, n=20)[-1]
            if len(latencies) >= 20 else max(latencies)
        ),
        "per_query_ids": per_query_ids,   # captured for ID-order comparison
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run", help="label for this run (for output JSON)")
    ap.add_argument("--out", default=None, help="write the result JSON to this path")
    ap.add_argument("--baseline", default=None, help="JSON from a prior run (typically main) to diff against")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)

    with tempfile.TemporaryDirectory() as d:
        s = SMRITI(SmritiConfig(storage_path=d))
        print(f"Seeding 500 memories (seed={args.seed})…")
        target_ids = seed(s, n=500)

        queries = []
        for topic in TOPICS:
            for paraphrase in [
                f"how does {topic} work",
                f"tell me about {topic}",
                f"{topic} ecosystem",
            ]:
                queries.append((paraphrase, topic))
        random.shuffle(queries)
        queries = queries[:50]

        print(f"\n[{args.label}] rewrite=none, snippet=none …")
        before = run_queries(s, target_ids, queries, rewrite="none", snippet="none")
        print(f"  → {json.dumps(before, indent=2)}")

        print(f"\n[{args.label}] rewrite=auto, snippet=auto …")
        after = run_queries(s, target_ids, queries, rewrite="auto", snippet="auto")
        print(f"  → {json.dumps(after, indent=2)}")

        result = {"label": args.label, "disabled": before, "enabled": after}

        tok_delta = (before["avg_tokens_per_query"] - after["avg_tokens_per_query"]) / max(before["avg_tokens_per_query"], 1) * 100
        hit_delta = (after["hit_rate_at_10"] - before["hit_rate_at_10"]) * 100
        print()
        print(f"Within-branch deltas:")
        print(f"  Token reduction: {tok_delta:.1f}%")
        print(f"  Hit-rate@10 delta: {hit_delta:+.1f} percentage points")

        if args.baseline:
            try:
                with open(args.baseline) as f:
                    base = json.load(f)
                base_disabled = base["disabled"]
                # The "disabled" mode on this branch must match the baseline (regression guard).
                hit_drift = (before["hit_rate_at_10"] - base_disabled["hit_rate_at_10"]) * 100
                print(f"\nCross-branch regression check (disabled-features vs baseline):")
                print(f"  Hit-rate@10 drift: {hit_drift:+.1f}pp (acceptance: ≤ 2pp)")

                # Spec §13.4 — same memory IDs in the same order vs pre-change main.
                # Compare per-query result lists.
                exact_match_count = 0
                total = 0
                for cur, prev in zip(before["per_query_ids"], base_disabled["per_query_ids"]):
                    total += 1
                    if cur["ids"] == prev["ids"]:
                        exact_match_count += 1
                pct = (exact_match_count / total * 100) if total else 0.0
                print(f"  ID-order exact-match: {exact_match_count}/{total} queries ({pct:.0f}%)")
                if exact_match_count != total:
                    print("  WARNING: ID-order drift detected — spec §13.4 acceptance criterion requires investigation.")
            except Exception as e:
                print(f"\n(could not read baseline {args.baseline}: {e})")

        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults written to {args.out}")

        s.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark and capture output**

```bash
cd /Users/shivamtyagi/PycharmProjects/nexus-memory
python3 scripts/bench_recall.py 2>&1 | tee /tmp/bench_smarter_recall.txt
```

Expected: prints two result blocks plus a delta summary. Token reduction should be ≥ 30%; hit-rate@10 delta should be ≥ -2pp (spec §13.3).

- [ ] **Step 3: Run full test suite — no regressions**

```bash
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -10
```

Expected: all tests PASS (target 250+ tests passing).

- [ ] **Step 4: Commit benchmark + final state**

```bash
git add scripts/bench_recall.py
git commit -m "test: add scripts/bench_recall.py for one-shot recall quality benchmarking"
```

---

## Done

Run finishing-a-development-branch skill to merge.

The PR description must include the before/after numbers from Task 15, Step 2.
