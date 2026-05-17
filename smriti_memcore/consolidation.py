"""
SMRITI v2 — Consolidation Engine.
Background processing inspired by sleep-based memory consolidation.
8 processes: spaced repetition, chunking, reflection, forgetting,
cross-referencing, skill extraction, conflict resolution, palace defrag.
Event-driven scheduling (not clock-based).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from smriti_memcore.models import (
    ConsolidationDepth, Memory, MemorySource, MemoryStatus,
    MemoryTombstone, SmritiConfig, Skill, SalienceScore, Visibility,
)
from smriti_memcore.episode_buffer import EpisodeBuffer
from smriti_memcore.palace import SemanticPalace, Room
from smriti_memcore.vector_store import VectorStore
from smriti_memcore.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """
    The background maintenance system — like sleep for AI memory.
    
    No existing AI memory system has this. Memories are stored and 
    retrieved, but never reorganized, strengthened, or pruned. 
    This is the biggest gap in current architectures.
    """

    def __init__(
        self,
        episode_buffer: EpisodeBuffer,
        palace: SemanticPalace,
        vector_store: VectorStore,
        llm: LLMInterface,
        config: SmritiConfig,
    ):
        self.buffer = episode_buffer
        self.palace = palace
        self.vector_store = vector_store
        self.llm = llm
        self.config = config

        # State tracking
        self.last_consolidation: Optional[datetime] = None
        self.tombstones: deque = deque(maxlen=500)
        self.skills: Dict[str, Skill] = {}
        self.consolidation_log: deque = deque(maxlen=100)

    # ── Scheduling ───────────────────────────────────────

    def should_consolidate(self) -> ConsolidationDepth:
        """
        Determine if consolidation should run and at what depth.
        Event-driven, not clock-based — AI agents don't have circadian rhythms.
        """
        unconsolidated = self.buffer.unconsolidated_count
        total_memories = len(self.palace.memories)

        # Deep trigger: any highly salient unconsolidated memory waiting
        # Threshold is 0.55 because Mistral/heuristic scoring rarely exceeds 0.6-0.7
        if self.buffer.get_high_salience(min_composite=0.55, limit=1):
            return ConsolidationDepth.FULL

        # Deep trigger: large unconsolidated backlog (check FULL first!)
        if unconsolidated >= self.config.episode_buffer_trigger * 4:
            return ConsolidationDepth.FULL

        # Deep trigger: total memory count exceeded
        if total_memories > self.config.backlog_trigger:
            return ConsolidationDepth.FULL

        # Micro trigger: buffer getting full
        if unconsolidated >= self.config.episode_buffer_trigger:
            return ConsolidationDepth.LIGHT

        return ConsolidationDepth.DEFER

    def consolidate(self, depth: Optional[ConsolidationDepth] = None) -> Dict:
        """
        Run consolidation processes at the specified depth.
        
        FULL: All 8 processes
        LIGHT: Chunking + conflict detection + basic forgetting
        """
        if depth is None:
            depth = self.should_consolidate()

        if depth == ConsolidationDepth.DEFER:
            return {"status": "deferred", "reason": "no consolidation needed"}

        start_time = time.time()
        results = {"depth": depth.value, "processes": {}}

        logger.info(f"Starting {depth.value} consolidation...")

        # Each process is isolated — one failure doesn't crash the rest
        for name, fn in [("chunking", self._process_chunking),
                         ("conflict", self._process_conflict_resolution),
                         ("forgetting", self._process_forgetting)]:
            try:
                results["processes"][name] = fn()
            except Exception as e:
                logger.error(f"Consolidation process '{name}' failed: {e}")
                results["processes"][name] = {"error": str(e)}

        if depth == ConsolidationDepth.FULL:
            for name, fn in [("reflection", self._process_reflection),
                             ("cross_reference", self._process_cross_reference),
                             ("skill_extraction", self._process_skill_extraction),
                             ("spaced_repetition", self._process_spaced_repetition),
                             ("defragmentation", self._process_defragmentation)]:
                try:
                    results["processes"][name] = fn()
                except Exception as e:
                    logger.error(f"Consolidation process '{name}' failed: {e}")
                    results["processes"][name] = {"error": str(e)}

        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        self.last_consolidation = datetime.now()
        self.consolidation_log.append(results)

        logger.info(
            f"Consolidation complete ({depth.value}) in {elapsed:.1f}s"
        )
        return results

    # ── Process 1: Spaced Repetition (Utility-Based) ─────

    def _process_spaced_repetition(self) -> Dict:
        """Review and decay memories based on utility, not just time."""
        reviewed = 0
        decayed = 0
        now = datetime.now()

        for memory in list(self.palace.memories.values()):
            # Skip inactive statuses, but allow PINNED through for review
            if memory.status not in (MemoryStatus.ACTIVE, MemoryStatus.PINNED):
                continue

            # Check if due for review
            if memory.next_review and now < memory.next_review:
                continue

            # Compute utility-based decay
            decay_score = self._compute_utility_decay(memory, now)
            
            if decay_score < 0.3 and memory.status != MemoryStatus.PINNED:
                # Low utility — apply decay (never decay PINNED memories)
                memory.decay(0.9)
                decayed += 1
            else:
                # Still useful (or pinned) — schedule next review
                days = 2 ** memory.consecutive_successful_reviews
                days = min(days, 180)
                memory.next_review = now + timedelta(days=days)

            reviewed += 1

        return {"reviewed": reviewed, "decayed": decayed}

    def _compute_utility_decay(self, memory: Memory, now: datetime) -> float:
        """Utility-based decay: usage + context staleness + mild temporal."""
        # Utility: is it being used?
        days_since_creation = max((now - memory.creation_time).days, 1)  # At least 1 day
        expected_usage = max(1, days_since_creation / 30)  # Expected ~1 access/month
        utility = min(memory.access_count / expected_usage, 1.0)

        # Context staleness (how much has the domain shifted?)
        # Use access recency as proxy
        days_since = (now - memory.last_accessed).total_seconds() / 86400
        context_factor = 0.98 ** days_since

        # Temporal (minor)
        temporal = self.config.decay_rate ** days_since

        return (
            0.5 * utility +
            0.3 * context_factor +
            0.2 * temporal
        )

    # ── Process 2: Chunking ──────────────────────────────

    def _process_chunking(self) -> Dict:
        """Group related unconsolidated episodes into chunks."""
        episodes = self.buffer.get_unconsolidated(limit=50)
        if not episodes:
            return {"chunks_created": 0}

        chunks_created = 0

        # Group by semantic similarity
        groups = self._group_similar_episodes(episodes, threshold=0.6)

        for group in groups:
            if len(group) < 2:
                # Singleton: mark consolidated so it doesn't block the buffer
                self.buffer.mark_consolidated([ep.id for ep in group])
                continue

            # Ask LLM to create chunk summary
            contents = [ep.content for ep in group]
            chunk_result = self.llm.chunk_memories(contents)

            summary = chunk_result.get("summary", " | ".join(contents))

            # Create consolidated memory from chunk
            memory = Memory(
                content=summary,
                source=MemorySource.INFERRED,
                strength=1.5,  # Consolidated memories start stronger
                reflection_level=1,
                metadata={"chunked_from": [ep.id for ep in group]},
            )

            # Place in palace
            self.palace.place_memory(memory)

            # Mark episodes as consolidated
            self.buffer.mark_consolidated([ep.id for ep in group])
            chunks_created += 1

        return {"chunks_created": chunks_created, "episodes_processed": len(episodes)}

    def _group_similar_episodes(self, episodes, threshold: float = 0.6):
        """Group episodes by semantic similarity."""
        if not episodes:
            return []

        # Build embedding matrix
        embeddings = []
        for ep in episodes:
            if ep.embedding:
                embeddings.append(np.array(ep.embedding))
            else:
                emb = self.vector_store.embed(ep.content)
                ep.embedding = emb.tolist()
                embeddings.append(emb)

        embeddings = np.array(embeddings)

        # Simple greedy clustering
        assigned = [False] * len(episodes)
        groups = []

        for i in range(len(episodes)):
            if assigned[i]:
                continue

            group = [episodes[i]]
            assigned[i] = True

            for j in range(i + 1, len(episodes)):
                if assigned[j]:
                    continue

                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    group.append(episodes[j])
                    assigned[j] = True

            groups.append(group)

        return groups

    # ── Process 3: Reflection Synthesizer ─────────────────

    def _process_reflection(self) -> Dict:
        """Generate higher-level abstractions from episodes."""
        reflections_created = 0

        # Get recent high-salience episodes
        episodes = self.buffer.get_high_salience(min_composite=0.5, limit=30)
        if not episodes:
            return {"reflections_created": 0}

        # Group and reflect
        groups = self._group_similar_episodes(episodes, threshold=0.5)

        for group in groups:
            # We allow reflection on highly salient single episodes
            if len(group) < 2 and group[0].salience.composite < 0.7:
                continue

            contents = [ep.content for ep in group]

            # Level 1: Observation
            observation = self.llm.generate_reflection(contents, level=1)

            # Create reflection memory
            memory = Memory(
                content=observation,
                source=MemorySource.INFERRED,
                strength=2.0,  # Reflections are very strong
                confidence=0.8,
                reflection_level=1,
                metadata={"reflected_from": [ep.id for ep in group]},
            )
            self.palace.place_memory(memory)
            reflections_created += 1

            # Level 2: Insight (from larger groups)
            if len(group) >= 5:
                insight = self.llm.generate_reflection(contents, level=2)
                memory_l2 = Memory(
                    content=insight,
                    source=MemorySource.INFERRED,
                    strength=3.0,
                    confidence=0.7,
                    reflection_level=2,
                    metadata={"insight_from": [ep.id for ep in group]},
                )
                self.palace.place_memory(memory_l2)
                reflections_created += 1

            # Mark processed episodes as consolidated so they don't block the buffer
            self.buffer.mark_consolidated([ep.id for ep in group])

        return {"reflections_created": reflections_created}

    # ── Process 4: Forgetting Manager ────────────────────

    def _process_forgetting(self) -> Dict:
        """Managed forgetting based on utility, not just time."""
        removed = 0
        archived = 0
        now = datetime.now()

        for memory in list(self.palace.memories.values()):
            if memory.status in (MemoryStatus.PINNED, MemoryStatus.SUPERSEDED):
                continue
            if memory.source == MemorySource.USER_STATED:
                continue  # Never forget user-stated info

            decay_score = self._compute_utility_decay(memory, now)

            if decay_score < self.config.strength_hard_threshold:
                # Graceful removal with tombstone
                tombstone = MemoryTombstone(
                    original_id=memory.id,
                    summary=memory.content[:100],
                    room_id=memory.room_id,
                    reason=f"utility_decay={decay_score:.3f}",
                    embedding=memory.embedding,
                )
                self.tombstones.append(tombstone)
                memory.status = MemoryStatus.ARCHIVED
                removed += 1

            elif decay_score < self.config.strength_soft_threshold:
                memory.status = MemoryStatus.ARCHIVED
                archived += 1

        return {"removed": removed, "archived": archived}

    # ── Process 5: Cross-Reference Linker ────────────────

    def _process_cross_reference(self) -> Dict:
        """Discover hidden connections between memories in different rooms."""
        links_created = 0
        rooms = list(self.palace.rooms.values())

        if len(rooms) < 2:
            return {"links_created": 0}

        # Compare room centroids for unexpected similarity
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                room_a = rooms[i]
                room_b = rooms[j]

                # Skip if already connected
                neighbors_ids = {n[0].id for n in self.palace.get_neighbors(room_a.id)}
                if room_b.id in neighbors_ids:
                    continue

                # Check centroid similarity
                if room_a.centroid_embedding is not None and room_b.centroid_embedding is not None:
                    sim = float(np.dot(room_a.centroid_embedding, room_b.centroid_embedding))
                    if sim > 0.4:  # Moderate similarity across different rooms
                        self.palace.link_rooms(
                            room_a.id, room_b.id,
                            relationship="semantic",
                            strength=sim,
                        )
                        links_created += 1
                        logger.debug(
                            f"Cross-reference: {room_a.topic} <-> {room_b.topic} "
                            f"(sim={sim:.2f})"
                        )

        return {"links_created": links_created}

    # ── Process 6: Skill Extraction ──────────────────────

    def _process_skill_extraction(self) -> Dict:
        """Detect repeated procedural patterns and extract as skills."""
        skills_extracted = 0

        # Look for code-related memories
        code_memories = [
            m for m in self.palace.memories.values()
            if m.status == MemoryStatus.ACTIVE
            and ("```" in m.content or "def " in m.content or 
                 "function" in m.content.lower())
        ]

        # Group similar code patterns
        groups = {}
        for mem in code_memories:
            # Simple grouping by room
            if mem.room_id not in groups:
                groups[mem.room_id] = []
            groups[mem.room_id].append(mem)

        for room_id, mems in groups.items():
            if len(mems) < 2:
                continue

            # Check if a skill already exists for this room
            existing_skill = any(
                s for s in self.skills.values()
                if s.metadata.get("room_id") == room_id
            )
            if existing_skill:
                continue

            # Create skill from repeated pattern
            contents = [m.content for m in mems[:5]]
            skill = Skill(
                name=f"skill_from_{room_id}",
                description=f"Extracted from {len(mems)} related code patterns",
                code=contents[0],  # Use the strongest as template
                preconditions=[],
                postconditions=[],
            )
            skill.embedding = self.vector_store.embed(skill.description).tolist()
            self.skills[skill.id] = skill
            skills_extracted += 1

        return {"skills_extracted": skills_extracted}

    # ── Process 7: Conflict Resolution ───────────────────

    def _process_conflict_resolution(self) -> Dict:
        """Detect and resolve contradicting memories."""
        conflicts_resolved = 0

        active_memories = [
            m for m in self.palace.memories.values()
            if m.status == MemoryStatus.ACTIVE and m.visibility != Visibility.PRIVATE
        ]

        # Compare recent memories against older ones
        recent = sorted(active_memories, key=lambda m: m.creation_time, reverse=True)[:20]

        for newer in recent:
            if newer.embedding is None:
                continue

            # Find semantically similar older memories
            similar = self.vector_store.search(
                query_vector=np.array(newer.embedding),
                top_k=10,
            )

            for vec_id, sim_score in similar:
                if not vec_id.startswith("mem:"):
                    continue
                mem_id = vec_id[4:]
                if mem_id == newer.id:
                    continue

                older = self.palace.get_memory(mem_id)
                if not older or older.status != MemoryStatus.ACTIVE:
                    continue

                # High similarity but potentially contradicting
                if sim_score > 0.7:
                    contradiction = self.llm.detect_contradiction(
                        newer.content, older.content
                    )
                    if contradiction.get("contradicts", False):
                        self._resolve_conflict(newer, older, contradiction)
                        conflicts_resolved += 1

        return {"conflicts_resolved": conflicts_resolved}

    def _resolve_conflict(self, newer: Memory, older: Memory, details: Dict):
        """Resolve a detected contradiction."""
        # Strategy selection
        if newer.source == MemorySource.USER_STATED:
            strategy = "authority"
        elif older.access_count > 10 and newer.access_count == 0:
            strategy = "temporal_cautious"  # Well-established vs brand new
        else:
            strategy = "temporal"  # Default: newer wins

        if strategy == "authority":
            winner, loser = newer, older
        elif strategy == "temporal":
            winner, loser = newer, older
        else:
            # Both are valid, just mark newer as update
            winner, loser = newer, older

        loser.status = MemoryStatus.SUPERSEDED
        loser.superseded_by = winner.id
        winner.confidence = min(winner.confidence + 0.1, 1.0)  # Conflict resolution bonus (capped)

        logger.info(
            f"Conflict resolved ({strategy}): "
            f"'{winner.content[:50]}' supersedes '{loser.content[:50]}'"
        )

    # ── Process 8: Palace Defragmentation ────────────────

    def _process_defragmentation(self) -> Dict:
        """Prevent graph fragmentation as the palace grows."""
        rooms_merged = 0
        rooms_archived = 0
        bridges_created = 0
        now = datetime.now()

        rooms = list(self.palace.rooms.values())

        # 1. Merge semantically overlapping rooms
        merged_ids = set()
        for i in range(len(rooms)):
            if rooms[i].id in merged_ids:
                continue
            for j in range(i + 1, len(rooms)):
                if rooms[j].id in merged_ids:
                    continue

                emb_a = self.palace._room_embeddings.get(rooms[i].id)
                emb_b = self.palace._room_embeddings.get(rooms[j].id)

                if emb_a is not None and emb_b is not None:
                    sim = float(np.dot(emb_a, emb_b))
                    if sim > self.config.room_merge_threshold:
                        self._merge_rooms(rooms[i], rooms[j])
                        merged_ids.add(rooms[j].id)
                        rooms_merged += 1

        # 2. Archive empty or abandoned rooms
        stale_cutoff = now - timedelta(days=self.config.room_stale_days)
        for room in list(self.palace.rooms.values()):
            active_memories = [
                mid for mid in room.memory_ids
                if mid in self.palace.memories
                and self.palace.memories[mid].status == MemoryStatus.ACTIVE
            ]
            if len(active_memories) == 0 or room.last_visited < stale_cutoff:
                # Don't delete, just mark as stale
                rooms_archived += 1

        # 3. Bridge isolated components
        connected = set(self.palace._adj.keys())

        isolated = [r for r in self.palace.rooms.values() if r.id not in connected]
        if isolated and connected:
            # Find closest connected room for each isolated one
            for iso_room in isolated:
                if iso_room.centroid_embedding is None:
                    continue
                best_match = None
                best_sim = 0
                for rid in connected:
                    r_emb = self.palace._room_embeddings.get(rid)
                    if r_emb is not None:
                        sim = float(np.dot(iso_room.centroid_embedding, r_emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_match = rid
                if best_match and best_sim > 0.2:
                    self.palace.link_rooms(iso_room.id, best_match, "semantic", best_sim)
                    bridges_created += 1

        return {
            "rooms_merged": rooms_merged,
            "rooms_archived": rooms_archived,
            "bridges_created": bridges_created,
            "palace_health": self.palace.health(),
        }

    def _merge_rooms(self, keeper: Room, merged: Room):
        """Merge two overlapping rooms into one."""
        # Move all memories from merged to keeper
        for mid in merged.memory_ids:
            if mid in self.palace.memories:
                self.palace.memories[mid].room_id = keeper.id
                if mid not in keeper.memory_ids:
                    keeper.memory_ids.append(mid)

        # Move edges from merged to keeper
        merged_edges = self.palace._adj.pop(merged.id, [])
        for edge in merged_edges:
            edge.source_room_id = keeper.id
            self.palace._adj.setdefault(keeper.id, []).append(edge)
        # Update reverse edges pointing to merged room
        for room_id, edges in self.palace._adj.items():
            for edge in edges:
                if edge.target_room_id == merged.id:
                    edge.target_room_id = keeper.id

        # Remove merged room
        del self.palace.rooms[merged.id]
        if merged.id in self.palace._room_embeddings:
            del self.palace._room_embeddings[merged.id]

        # Update keeper centroid
        self.palace._update_room_centroid(keeper)
        logger.info(f"Merged room '{merged.topic}' into '{keeper.topic}'")

    # ── Stats ────────────────────────────────────────────

    def stats(self) -> Dict:
        """Consolidation engine statistics."""
        return {
            "last_consolidation": self.last_consolidation.isoformat() if self.last_consolidation else None,
            "total_consolidations": len(self.consolidation_log),
            "tombstones": len(self.tombstones),
            "skills_extracted": len(self.skills),
            "unconsolidated_episodes": self.buffer.unconsolidated_count,
        }
