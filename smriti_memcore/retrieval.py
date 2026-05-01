"""
SMRITI v2 — Retrieval Engine.
Multi-hop associative retrieval through the Semantic Palace with
retrieval strengthening (testing effect) and effort-based desirable
difficulty bonus.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from smriti_memcore.fts_index import FTSIndex

import numpy as np

from smriti_memcore.models import Memory, SmritiConfig
from smriti_memcore.palace import SemanticPalace
from smriti_memcore.working_memory import WorkingMemory
from smriti_memcore.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Active retrieval that strengthens memories (the testing effect).
    
    The key innovation: every retrieval is a WRITE operation.
    This implements the most powerful human memory principle — retrieval
    practice — which no existing AI system uses.
    """

    def __init__(
        self,
        palace: SemanticPalace,
        working_memory: WorkingMemory,
        vector_store: VectorStore,
        config: SmritiConfig,
        fts_index: Optional["FTSIndex"] = None,
    ):
        self.palace = palace
        self.working_memory = working_memory
        self.vector_store = vector_store
        self.config = config
        self.fts_index = fts_index

        # Retrieval log (for salience weight learning, bounded to prevent memory leaks)
        self.retrieval_log: deque = deque(maxlen=1000)

    def retrieve(
        self,
        query: str,
        context: str = "",
        top_k: Optional[int] = None,
        max_hops: int = 1,
    ) -> List[Memory]:
        """
        Full retrieval pipeline:
        1. Navigate the Semantic Palace (multi-hop)
        2. Multi-factor scoring
        3. Retrieval strengthening (testing effect)
        4. Effort-based desirable difficulty bonus
        5. Admit top results to Working Memory
        """
        top_k = top_k or self.config.retrieval_top_k
        start_time = time.time()

        # Step 1a — vector search (wider pool: top_k*3)
        vector_candidates = self.palace.search(query, top_k=top_k * 3, max_hops=max_hops)

        if self.fts_index is not None:
            # Step 1b — FTS keyword search
            try:
                fts_results = self.fts_index.search(query, top_k=top_k * 3)
            except Exception:
                logger.warning("FTS search failed — falling back to vector-only retrieval")
                fts_results = []

            # Step 1c — RRF merge → ordered list of IDs
            merged_ids = self._rrf_merge(
                vector_candidates, fts_results, pool_size=top_k * 2
            )

            # Step 1d — reconstruct Memory objects; fetch FTS-only IDs from palace
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

        # 2. Multi-factor scoring
        query_embedding = self.vector_store.embed(query)
        now = datetime.now()

        for memory in candidates:
            memory.retrieval_score = self._score_memory(memory, query_embedding, now)

        # Sort by composite retrieval score
        candidates.sort(key=lambda m: m.retrieval_score, reverse=True)
        selected = candidates[:top_k]

        # 3. THE KEY INNOVATION: Retrieval strengthens the memory
        for memory in selected:
            memory.reinforce(self.config.reinforcement_factor)
            # Recalculate spaced repetition interval
            memory.consecutive_successful_reviews += 1
            memory.next_review = self._next_review_interval(memory)

        # 4. Desirable difficulty: EFFORT-based bonus
        for memory in selected:
            retrieval_effort = self._compute_effort(memory, now)
            if retrieval_effort > self.config.effort_threshold:
                # Hard but found — extra reinforcement
                memory.strength *= self.config.difficulty_bonus
                logger.debug(
                    f"Difficulty bonus applied to {memory.id} "
                    f"(effort={retrieval_effort:.2f})"
                )

        # 5. Admit to working memory
        for memory in selected:
            self.working_memory.admit(memory)

        # Log for salience weight learning
        elapsed_ms = (time.time() - start_time) * 1000
        self.retrieval_log.append({
            "query": query,
            "results": [m.id for m in selected],
            "scores": [m.retrieval_score for m in selected],
            "latency_ms": elapsed_ms,
            "timestamp": now.isoformat(),
        })

        logger.info(
            f"Retrieved {len(selected)} memories for '{query[:40]}...' "
            f"({elapsed_ms:.0f}ms)"
        )
        return selected

    def retrieve_by_id(self, memory_id: str) -> Optional[Memory]:
        """Direct retrieval by ID — still strengthens the memory."""
        memory = self.palace.get_memory(memory_id)
        if memory:
            memory.reinforce(self.config.reinforcement_factor)
        return memory

    # ── Scoring ──────────────────────────────────────────

    def _score_memory(
        self, memory: Memory, query_embedding: np.ndarray, now: datetime
    ) -> float:
        """Multi-factor retrieval scoring."""
        # Relevance (cosine similarity)
        if memory.embedding:
            relevance = float(np.dot(query_embedding, np.array(memory.embedding)))
        else:
            relevance = 0.0

        # Recency (exponential decay)
        days_since = (now - memory.last_accessed).total_seconds() / 86400
        recency = self.config.decay_rate ** days_since

        # Strength
        strength = min(memory.strength / 5.0, 1.0)  # Normalize to [0, 1]

        # Salience
        salience = memory.salience.composite

        # Composite score
        score = (
            self.config.relevance_weight * relevance +
            self.config.recency_weight * recency +
            self.config.strength_weight * strength +
            self.config.salience_weight * salience
        )

        return score

    def _rrf_merge(
        self,
        vector_candidates: List[Memory],
        fts_results: List[Tuple[str, float]],
        pool_size: int,
        k: int = 60,
    ) -> List[str]:
        scores: Dict[str, float] = defaultdict(float)
        for rank, memory in enumerate(vector_candidates):
            scores[memory.id] += 1.0 / (k + rank + 1)
        for rank, (memory_id, _) in enumerate(fts_results):
            scores[memory_id] += 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda mid: scores[mid], reverse=True)[:pool_size]

    def _compute_effort(self, memory: Memory, now: datetime) -> float:
        """
        Compute retrieval effort — NOT the same as low relevance.
        
        High effort = multi-hop traversal + long time since access + low strength.
        This is what "desirable difficulty" actually measures.
        """
        hop_effort = memory.hops * 0.5  # Each hop = 0.5 effort
        time_effort = min(
            (now - memory.last_accessed).total_seconds() / (30 * 86400),  # months
            1.0
        )
        weakness_effort = max(0, 1.0 - memory.strength)

        return hop_effort + time_effort + weakness_effort

    def _next_review_interval(self, memory: Memory) -> datetime:
        """Calculate next spaced repetition review time."""
        from datetime import timedelta

        # Base interval expands exponentially with successful reviews
        base_days = 2 ** memory.consecutive_successful_reviews

        # Cap at 180 days
        base_days = min(base_days, 180)

        # Context-shift factor would compress the interval
        # (for now, use strength as a proxy)
        shift_factor = max(0.5, memory.strength / 2.0)

        interval_days = base_days * shift_factor
        return datetime.now() + timedelta(days=interval_days)

    # ── Stats ────────────────────────────────────────────

    def stats(self) -> dict:
        """Retrieval engine statistics."""
        if not self.retrieval_log:
            return {"total_retrievals": 0}

        latencies = [r["latency_ms"] for r in self.retrieval_log]
        return {
            "total_retrievals": len(self.retrieval_log),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "avg_results_per_query": sum(
                len(r["results"]) for r in self.retrieval_log
            ) / len(self.retrieval_log),
        }
