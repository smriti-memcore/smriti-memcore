"""
SMRITI v2 — Core Integration.
The main SMRITI class that orchestrates all components into a unified API.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from typing import Any, Dict, List, Optional

from smriti_memcore.models import (
    ConfidenceLevel, DecisionType, Episode, Memory, MemorySource,
    MemoryStatus, Modality, SmritiConfig, SalienceScore,
)
from smriti_memcore.llm_interface import LLMInterface
from smriti_memcore.vector_store import VectorStore
from smriti_memcore.episode_buffer import EpisodeBuffer
from smriti_memcore.palace import SemanticPalace
from smriti_memcore.working_memory import WorkingMemory
from smriti_memcore.attention_gate import AttentionGate
from smriti_memcore.retrieval import RetrievalEngine
from smriti_memcore.consolidation import ConsolidationEngine
from smriti_memcore.meta_memory import MetaMemory
from smriti_memcore.metrics import SmritiMetrics
from smriti_memcore.fts_index import FTSIndex

logger = logging.getLogger(__name__)


class SMRITI:
    """
    Neuro-Inspired EXperience-Unified System.
    
    A novel AI agent memory architecture that transplants the computational
    principles behind human memory champion techniques into AI design.
    
    Usage:
        smriti = SMRITI()
        smriti.encode("User prefers Python for backend development")
        memories = smriti.recall("What language does the user prefer?")
        confidence = smriti.how_well_do_i_know("programming languages")
        smriti.consolidate()
    """

    def __init__(self, config: Optional[SmritiConfig] = None):
        self.config = config or SmritiConfig()

        # Create storage directories
        os.makedirs(self.config.storage_path, exist_ok=True)

        # Metrics / observability
        self._metrics = SmritiMetrics()

        # Initialize components
        self.llm = LLMInterface(
            ollama_base_url=self.config.ollama_base_url,
            default_model=self.config.llm_model,
            openai_api_key=self.config.openai_api_key,
            anthropic_api_key=self.config.anthropic_api_key,
            gemini_api_key=self.config.gemini_api_key,
            metrics=self._metrics,
        )

        self.vector_store = VectorStore(
            model_name=self.config.embedding_model,
            dimension=self.config.embedding_dim,
            storage_path=os.path.join(self.config.storage_path, "vectors"),
        )

        self.episode_buffer = EpisodeBuffer(
            storage_path=os.path.join(self.config.storage_path, "episodes"),
            vector_store=self.vector_store,
        )

        self.palace = SemanticPalace(
            vector_store=self.vector_store,
            storage_path=os.path.join(self.config.storage_path, "palace"),
        )

        # FTS index — expendable derived index, self-heals via rebuild
        self.fts_index = FTSIndex(self.config.storage_path)
        active_memories = [
            m for m in self.palace.memories.values()
            if m.status == MemoryStatus.ACTIVE
        ]
        if self.fts_index.needs_rebuild(len(active_memories)):
            self.fts_index.rebuild(active_memories)

        self.working_memory = WorkingMemory(
            max_slots=self.config.working_memory_slots,
            active_chunks=self.config.active_chunks,
        )

        self.attention_gate = AttentionGate(
            llm=self.llm,
            config=self.config,
        )

        self.retrieval_engine = RetrievalEngine(
            palace=self.palace,
            working_memory=self.working_memory,
            vector_store=self.vector_store,
            config=self.config,
            fts_index=self.fts_index,
        )

        self.consolidation_engine = ConsolidationEngine(
            episode_buffer=self.episode_buffer,
            palace=self.palace,
            vector_store=self.vector_store,
            llm=self.llm,
            config=self.config,
        )

        self.meta_memory = MetaMemory(palace=self.palace)

        # Register cleanup on unexpected exit
        atexit.register(self._atexit_save)
        self._closed = False

        logger.info("SMRITI v2 initialized")

    # ── Core API ─────────────────────────────────────────

    def encode(
        self,
        content: str,
        context: str = "",
        source: MemorySource = MemorySource.DIRECT,
        modality: Modality = Modality.TEXT,
        use_llm: bool = True,
    ) -> Optional[str]:
        """
        Encode new information into SMRITI.
        
        Pipeline: Attention Gate → Episode Buffer → Palace placement
        
        Returns the memory ID if encoded, None if discarded.
        """
        start = time.perf_counter()

        # Input validation
        if not content or not content.strip():
            self._metrics.encode_discarded.inc()
            return None
        if len(content) > self.config.max_content_length:
            content = content[:self.config.max_content_length]
            logger.warning(f"Content truncated to {self.config.max_content_length} chars")

        # 1. Attention Gate: score salience and decide encoding
        episode = self.attention_gate.process(content, context, source, use_llm)

        if episode is None:
            self._metrics.encode_discarded.inc()
            return None  # Discarded

        # 2. Episode Buffer: store the raw experience
        self.episode_buffer.add(episode)

        # 3. Create memory and place in palace
        memory = Memory(
            content=content,
            modality=modality,
            source=source,
            salience=episode.salience,
            embedding=episode.embedding,
            confidence=1.0 if source == MemorySource.USER_STATED else 0.8,
        )

        room = self.palace.place_memory(memory)

        try:
            self.fts_index.add(memory.id, content)
        except Exception as e:
            logger.warning(f"FTS add failed for {memory.id}: {e}")

        # 4. Auto-consolidate if buffer is getting full
        if self.episode_buffer.unconsolidated_count >= self.config.episode_buffer_trigger:
            logger.info("Auto-triggering consolidation (buffer threshold reached)")
            self.consolidate(depth=None)  # Let scheduler decide depth

        # Track metrics
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._metrics.encode_count.inc()
        self._metrics.encode_latency.observe(elapsed_ms)
        self._update_gauges()

        logger.info(
            f"Encoded: '{content[:60]}...' → room '{room.topic}' "
            f"(salience={episode.salience.composite:.2f})"
        )
        return memory.id

    def recall(
        self,
        query: str,
        context: str = "",
        top_k: Optional[int] = None,
    ) -> List[Memory]:
        """
        Recall memories relevant to a query.
        
        Pipeline: Meta-Memory check → Retrieval Engine → Working Memory
        
        Returns list of memories, strongest first.
        Also surfaces any proactive suggestions/warnings.
        """
        start = time.perf_counter()

        # 1. Meta-memory check: do we even know about this?
        decision = self.meta_memory.should_recall_or_ask(query)

        if decision == DecisionType.ADMIT_GAP_AND_ASK:
            self.meta_memory.register_failed_retrieval(query, context)
            logger.info(f"Knowledge gap detected for: {query[:60]}...")
            # Still try retrieval, but the caller should know confidence is low

        # 2. Retrieval Engine: multi-hop search + strengthening
        memories = self.retrieval_engine.retrieve(query, context, top_k)

        if not memories:
            self.meta_memory.register_failed_retrieval(query, context)
            self._metrics.recall_empty.inc()

        # Track metrics
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._metrics.recall_count.inc()
        self._metrics.recall_latency.observe(elapsed_ms)

        return memories

    def how_well_do_i_know(self, topic: str) -> ConfidenceLevel:
        """
        Meta-memory confidence check.
        
        Returns a ConfidenceLevel with coverage, freshness, strength, depth.
        """
        return self.meta_memory.confidence_map(topic)

    # ── Consolidation ────────────────────────────────────

    def consolidate(self, depth=None) -> Dict:
        """
        Run the Consolidation Engine.
        
        If depth is None, the scheduler decides automatically.
        Otherwise, force a specific depth: 'full', 'light', or 'defer'.
        """
        from smriti_memcore.models import ConsolidationDepth

        if isinstance(depth, str):
            depth = ConsolidationDepth(depth)

        result = self.consolidation_engine.consolidate(depth)

        # Track metrics
        self._metrics.consolidation_count.inc()
        if "elapsed_seconds" in result:
            self._metrics.consolidation_latency.observe(result["elapsed_seconds"])
        # Count any process errors
        for proc_result in result.get("processes", {}).values():
            if isinstance(proc_result, dict) and "error" in proc_result:
                self._metrics.consolidation_errors.inc()
        self._update_gauges()

        return result

    def reflect(self) -> Dict:
        """Force a reflection cycle (subset of consolidation)."""
        return self.consolidation_engine._process_reflection()

    def defragment(self) -> Dict:
        """Force palace defragmentation."""
        return self.consolidation_engine._process_defragmentation()

    # ── Memory Management ────────────────────────────────

    def pin(self, memory_id: str):
        """Mark a memory as permanent (never forget)."""
        memory = self.palace.get_memory(memory_id)
        if memory:
            memory.status = MemoryStatus.PINNED
            logger.info(f"Pinned memory: {memory_id}")

    def forget(self, memory_id: str):
        """Explicitly forget a memory."""
        memory = self.palace.get_memory(memory_id)
        if memory:
            memory.status = MemoryStatus.ARCHIVED
            self.fts_index.remove(memory_id)
            logger.info(f"Explicitly forgotten: {memory_id}")

    def resolve_conflict(self, mem_a_id: str, mem_b_id: str, strategy: str = "temporal"):
        """Manually resolve a contradiction between two memories."""
        mem_a = self.palace.get_memory(mem_a_id)
        mem_b = self.palace.get_memory(mem_b_id)
        if mem_a and mem_b:
            self.consolidation_engine._resolve_conflict(
                mem_a, mem_b, {"strategy": strategy}
            )

    # ── Palace ───────────────────────────────────────────

    def create_room(self, topic: str) -> str:
        """Manually create a palace room."""
        room = self.palace.create_room(topic)
        return room.id

    def link_rooms(self, room_a_id: str, room_b_id: str, relationship: str = "semantic"):
        """Manually link two rooms."""
        self.palace.link_rooms(room_a_id, room_b_id, relationship)

    # ── Working Memory ───────────────────────────────────

    def get_context(self) -> str:
        """Get formatted working memory context for LLM injection."""
        return self.working_memory.format_for_llm()

    def get_suggestions(self) -> List[Memory]:
        """Get proactive suggestions from the Ambient Monitor."""
        return self.working_memory.get_suggestions()

    def get_warnings(self) -> List[str]:
        """Get proactive warnings from the Ambient Monitor."""
        return self.working_memory.get_warnings()

    # ── Inspection ───────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Comprehensive system statistics."""
        self._update_gauges()
        return {
            "palace": self.palace.health(),
            "working_memory": self.working_memory.stats(),
            "retrieval": self.retrieval_engine.stats(),
            "consolidation": self.consolidation_engine.stats(),
            "meta_memory": self.meta_memory.stats(),
            "episode_buffer": {
                "total_episodes": self.episode_buffer.count,
                "unconsolidated": self.episode_buffer.unconsolidated_count,
            },
            "vector_store": {
                "total_vectors": self.vector_store.size,
            },
            "metrics": self._metrics.snapshot(),
        }

    def knowledge_gaps(self) -> List[Dict]:
        """What the agent knows it doesn't know."""
        return self.meta_memory.knowledge_gaps()

    def confidence_summary(self) -> str:
        """Human-readable knowledge confidence summary."""
        return self.meta_memory.get_confidence_summary()

    def eviction_history(self, n: int = 10) -> List:
        """What was recently pushed out of working memory."""
        return self.working_memory.get_recent_evictions(n)

    # ── Metrics ──────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics snapshot as a JSON-serializable dict."""
        self._update_gauges()
        return self._metrics.snapshot()

    def get_metrics_prometheus(self) -> str:
        """Get metrics in Prometheus text exposition format."""
        self._update_gauges()
        return self._metrics.prometheus()

    def _update_gauges(self):
        """Refresh gauge values from current system state."""
        self._metrics.memory_count.set(len(self.palace.memories))
        self._metrics.room_count.set(len(self.palace.rooms))
        self._metrics.episode_count.set(self.episode_buffer.count)
        self._metrics.vector_count.set(self.vector_store.size)
        self._metrics.working_memory_occupancy.set(
            len(self.working_memory._slots)
        )

    # ── Persistence ──────────────────────────────────────

    def save(self):
        """Save all state to disk."""
        self.palace.save()
        self.episode_buffer.save()
        self.vector_store.save()
        logger.info("SMRITI state saved")

    def close(self):
        """Save state and release resources. Safe to call multiple times."""
        if self._closed:
            return
        try:
            self.save()
        except Exception as e:
            logger.error(f"Error during save on close: {e}")
        self.episode_buffer.close()
        self.fts_index.close()
        self._closed = True
        # Unregister atexit handler since we cleaned up
        try:
            atexit.unregister(self._atexit_save)
        except Exception:
            pass
        logger.info("SMRITI closed")

    def _atexit_save(self):
        """Best-effort save on unexpected exit."""
        if not self._closed:
            try:
                self.save()
            except Exception:
                pass
            try:
                self.fts_index.close()
            except Exception:
                pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — auto-save and close."""
        self.close()
        return False

    def __repr__(self) -> str:
        h = self.palace.health()
        return (
            f"SMRITI(memories={h.get('memory_count', 0)}, "
            f"rooms={h.get('room_count', 0)}, "
            f"episodes={self.episode_buffer.count})"
        )
