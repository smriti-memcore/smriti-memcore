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
from typing import List, Optional, Tuple

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
            return self._extract_llm(memory, query_variants, raw_query_embedding)
        raise ValueError(f"Unknown mode {mode!r}")

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
            # Spec §5.5 — zero-overlap cosine floor.
            # Cheap rare path: embed each sentence once and pick the closest.
            # Embeddings are L2-normalized by vector_store.embed() (vector_store.py:120),
            # so np.dot() is cosine similarity.
            sentence_embs = [self.vector_store.embed(s) for s in sentences]
            scores = [float(np.dot(raw_query_embedding, se)) for se in sentence_embs]
            top_idx = int(np.argmax(scores))
            memory.snippet = sentences[top_idx]
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

    def _extract_llm(self, memory, query_variants, raw_query_embedding) -> ExtractResult:
        if self.llm is None:
            logger.warning("SnippetExtractor mode='llm' requested but no LLM configured; falling back to auto")
            # _extract_auto mutates memory.snippet directly — we discard its return value
            # because the outer ExtractResult below carries the fallback flag.
            self._extract_auto(memory, query_variants, raw_query_embedding)
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
