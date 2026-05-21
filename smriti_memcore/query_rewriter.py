"""
SMRITI v2 — Query Rewriter.

Generates query variants for the recall pipeline. mode="auto" produces
lexical variants (raw, stop-stripped, content-words) at microsecond cost;
mode="llm" produces LLM paraphrases (1-3s) with an LRU cache.

See docs/superpowers/specs/2026-05-20-smarter-recall-design.md §4.
"""
from __future__ import annotations

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
            return self._llm_expand(query)
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

    # ── LLM variant generation ─────────────────────────────────

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

        # Accept either {"variants": [...]} (preferred — matches generate_json's dict-recovery
        # fallback for prose-wrapped responses) or a raw list (some prompts emit it directly).
        # generate_json returns {"error": ...} on parse failure; that hits the non-list branch
        # below and falls back cleanly.
        if isinstance(raw, dict) and isinstance(raw.get("variants"), list):
            items = raw["variants"]
        elif isinstance(raw, list):
            items = raw
        else:
            logger.warning(
                f"QueryRewriter LLM returned unexpected shape ({type(raw).__name__}); "
                "falling back to auto"
            )
            return ExpandResult(variants=self._lexical_variants(query), used_mode="auto", fallback=True)

        # Filter empty/whitespace and deduplicate (preserving order), drop entries equal to raw query
        seen = {query}
        cleaned: List[str] = [query]
        for item in items:
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
            "Given this user query, generate exactly 3 paraphrased variants that\n"
            "preserve meaning but use different wording. Return a JSON object of\n"
            "the form {\"variants\": [\"...\", \"...\", \"...\"]} — no prose, no markdown.\n\n"
            f"Query: {query}\n"
            "Output JSON:"
        )
