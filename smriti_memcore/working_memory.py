"""
SMRITI v2 — Working Memory.
Capacity-limited priority queue (Miller's Law: 7 slots) with
eviction logging and proactive ambient monitoring.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from smriti_memcore.models import Memory

logger = logging.getLogger(__name__)


@dataclass
class EvictionRecord:
    """Record of a memory evicted from working memory."""
    memory_id: str
    content_preview: str
    evicted_at: datetime
    reason: str  # "capacity", "priority", "explicit"
    was_useful: Optional[bool] = None  # Set later for learning


class WorkingMemory:
    """
    The agent's active cognitive workspace — capacity-limited.
    
    Inspired by Miller's Law (7±2) and Cowan's 4-chunk active limit.
    The capacity constraint is ESSENTIAL — it creates the retrieval
    pressure that makes salience scoring, consolidation, and forgetting
    genuinely necessary. Without it, this is just "the context window."
    """

    def __init__(self, max_slots: int = 7, active_chunks: int = 4):
        self.max_slots = max_slots
        self.active_chunks = active_chunks

        # Priority queue (sorted by priority descending)
        self._slots: List[_SlotEntry] = []
        self.eviction_log: deque = deque(maxlen=500)

        # Proactive suggestions from Ambient Monitor
        self._suggestions: List[Memory] = []
        self._warnings: List[str] = []

    # ── Core Operations ──────────────────────────────────

    def admit(self, memory: Memory, priority: Optional[float] = None) -> Optional[EvictionRecord]:
        """
        Add a memory to working memory. If at capacity, evict the 
        lowest-priority item and log the eviction.
        
        Returns the EvictionRecord if an eviction occurred, None otherwise.
        """
        if priority is None:
            priority = self._compute_priority(memory)

        # Deduplicate — if already present, update priority instead
        if self.contains(memory.id):
            self.update_priority(memory.id, max(priority, self._get_priority(memory.id)))
            return None

        eviction = None

        if len(self._slots) >= self.max_slots:
            # Evict lowest-priority entry
            self._slots.sort(key=lambda s: s.priority)
            evicted_slot = self._slots.pop(0)
            eviction = EvictionRecord(
                memory_id=evicted_slot.memory.id,
                content_preview=evicted_slot.memory.content[:100],
                evicted_at=datetime.now(),
                reason="capacity",
            )
            self.eviction_log.append(eviction)
            logger.debug(
                f"Evicted memory {evicted_slot.memory.id} "
                f"(priority={evicted_slot.priority:.2f})"
            )

        self._slots.append(_SlotEntry(memory=memory, priority=priority))
        self._slots.sort(key=lambda s: s.priority, reverse=True)
        return eviction

    def remove(self, memory_id: str):
        """Explicitly remove a memory from working memory."""
        self._slots = [s for s in self._slots if s.memory.id != memory_id]

    def contains(self, memory_id: str) -> bool:
        """Check if a memory is in working memory."""
        return any(s.memory.id == memory_id for s in self._slots)

    def update_priority(self, memory_id: str, new_priority: float):
        """Update the priority of a memory in working memory."""
        for slot in self._slots:
            if slot.memory.id == memory_id:
                slot.priority = new_priority
                break
        self._slots.sort(key=lambda s: s.priority, reverse=True)

    @property
    def size(self) -> int:
        """Current number of items in working memory."""
        return len(self._slots)

    @property
    def is_full(self) -> bool:
        return len(self._slots) >= self.max_slots

    # ── Context Retrieval ────────────────────────────────

    def get_active_context(self) -> List[Memory]:
        """
        Get the top active chunks — what the LLM actually focuses on.
        These are the highest-priority items (Cowan's 4-chunk limit).
        """
        return [s.memory for s in self._slots[:self.active_chunks]]

    def get_peripheral_context(self) -> List[Memory]:
        """
        Get peripheral items (slots 5-7) — available but not primary.
        """
        return [s.memory for s in self._slots[self.active_chunks:]]

    def get_all(self) -> List[Memory]:
        """Get all memories in working memory, priority-ordered."""
        return [s.memory for s in self._slots]

    def format_for_llm(self) -> str:
        """Format working memory contents for LLM context injection."""
        lines = []

        def _format_mem(mem: Memory) -> str:
            if mem.content_compressed:
                return (
                    f"• {mem.content_compressed}\n"
                    f"  ⟨compressed:{mem.id}⟩ — call smriti_retrieve_original('{mem.id}') for full text"
                )
            return f"• {mem.content}"

        active = self.get_active_context()
        if active:
            lines.append("=== Active Context ===")
            for mem in active:
                lines.append(_format_mem(mem))

        peripheral = self.get_peripheral_context()
        if peripheral:
            lines.append("\n=== Background Context ===")
            for mem in peripheral:
                lines.append(_format_mem(mem))

        if self._suggestions:
            lines.append("\n=== Suggestions ===")
            for mem in self._suggestions:
                # Suggestion previews don't show compression markers, just the first 100 chars
                lines.append(f"💡 Relevant: {mem.content[:100]}")

        if self._warnings:
            lines.append("\n=== Warnings ===")
            for w in self._warnings:
                lines.append(f"⚠️ {w}")

        return "\n".join(lines)

    # ── Ambient Monitor ──────────────────────────────────

    def surface_suggestion(self, memory: Memory):
        """Proactively surface a relevant memory ('this reminds me of...')."""
        if memory.id not in [m.id for m in self._suggestions]:
            self._suggestions.append(memory)
            # Keep only last 3 suggestions
            self._suggestions = self._suggestions[-3:]
            logger.debug(f"Ambient suggestion: {memory.content[:80]}")

    def surface_warning(self, warning: str):
        """Surface a warning ('a similar approach failed before')."""
        self._warnings.append(warning)
        self._warnings = self._warnings[-3:]
        logger.debug(f"Ambient warning: {warning[:80]}")

    def clear_suggestions(self):
        """Clear proactive suggestions (after they've been delivered)."""
        self._suggestions.clear()
        self._warnings.clear()

    def get_suggestions(self) -> List[Memory]:
        """Get current proactive suggestions."""
        return list(self._suggestions)

    def get_warnings(self) -> List[str]:
        """Get current warnings."""
        return list(self._warnings)

    # ── Eviction History ─────────────────────────────────

    def get_recent_evictions(self, n: int = 10) -> List[EvictionRecord]:
        """Get recent eviction history (useful for consolidation decisions)."""
        return self.eviction_log[-n:]

    def mark_eviction_useful(self, memory_id: str, was_useful: bool):
        """Retroactively mark whether an evicted memory was needed."""
        for record in reversed(self.eviction_log):
            if record.memory_id == memory_id:
                record.was_useful = was_useful
                break

    # ── Internal ─────────────────────────────────────────

    def _get_priority(self, memory_id: str) -> float:
        """Get the current priority of a memory in a slot."""
        for slot in self._slots:
            if slot.memory.id == memory_id:
                return slot.priority
        return 0.0

    def _compute_priority(self, memory: Memory) -> float:
        """Compute initial priority for a memory entering working memory."""
        return (
            0.4 * memory.salience.composite +
            0.3 * memory.strength +
            0.2 * memory.confidence +
            0.1 * (1.0 if memory.access_count == 0 else 0.5)  # Novelty bonus
        )

    def stats(self) -> dict:
        """Working memory statistics."""
        return {
            "slots_used": len(self._slots),
            "max_slots": self.max_slots,
            "active_chunks": min(len(self._slots), self.active_chunks),
            "suggestions_pending": len(self._suggestions),
            "warnings_pending": len(self._warnings),
            "total_evictions": len(self.eviction_log),
        }


@dataclass
class _SlotEntry:
    """Internal: a memory slot with its priority score."""
    memory: Memory
    priority: float
    admitted_at: datetime = field(default_factory=datetime.now)
