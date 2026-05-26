"""
SMRITI v2 — Semantic Palace.
Graph-based semantic clustering with contextual priming and multi-hop
associative retrieval. NOT spatial simulation — captures the computational
mechanism (contextual priming + typed associative traversal).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from smriti_memcore.models import Memory, MemoryStatus, Visibility
from smriti_memcore.vector_store import VectorStore

logger = logging.getLogger(__name__)

PALACE_SCHEMA_VERSION = 3


@dataclass
class Room:
    """A semantic cluster in the palace — provides contextual priming."""
    id: str
    topic: str
    centroid_embedding: Optional[np.ndarray] = None
    visit_count: int = 0
    last_visited: datetime = field(default_factory=datetime.now)
    memory_ids: List[str] = field(default_factory=list)
    visibility: Visibility = field(default_factory=lambda: Visibility.SHARED)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "visit_count": self.visit_count,
            "last_visited": self.last_visited.isoformat(),
            "memory_ids": self.memory_ids,
            "memory_count": len(self.memory_ids),
            "visibility": self.visibility.value,
        }


@dataclass
class TypedEdge:
    """An associative bridge between rooms — enables multi-hop reasoning."""
    source_room_id: str
    target_room_id: str
    relationship: str  # "causal", "temporal", "analogical", "compositional"
    strength: float = 1.0
    creation_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "source": self.source_room_id,
            "target": self.target_room_id,
            "relationship": self.relationship,
            "strength": self.strength,
        }


class SemanticPalace:
    """
    Navigable graph of semantic clusters with typed associations.
    
    The value for AI is NOT spatial navigation (AI has no spatial cognition 
    hardware) but rather:
    1. Contextual priming — activating a room pre-loads related concepts
    2. Multi-hop associative retrieval — following typed edges surfaces
       memories that pure vector search would miss
    3. Cluster coherence — related memories mutually reinforce retrieval
    """

    def __init__(self, vector_store: VectorStore, storage_path: Optional[str] = None):
        self.vector_store = vector_store
        self.storage_path = storage_path

        self.rooms: Dict[str, Room] = {}
        self._adj: Dict[str, List[TypedEdge]] = {}   # adjacency list: room_id -> outgoing edges
        self.memories: Dict[str, Memory] = {}
        self.landmarks: List[str] = []  # High-salience memory IDs
        self._lock = threading.Lock()

        # Room embeddings for finding relevant rooms
        self._room_embeddings: Dict[str, np.ndarray] = {}

        if storage_path:
            self._load(storage_path)

    # ── Room Management ──────────────────────────────────

    def create_room(self, topic: str, room_id: Optional[str] = None) -> Room:
        """Create a new semantic cluster."""
        import uuid
        rid = room_id or str(uuid.uuid4())[:8]
        embedding = self.vector_store.embed(topic)

        room = Room(
            id=rid,
            topic=topic,
            centroid_embedding=embedding,
        )
        self.rooms[rid] = room
        self._room_embeddings[rid] = embedding

        # Add room to vector store for room-level search
        self.vector_store.add(
            id=f"room:{rid}",
            vector=embedding,
            metadata={"type": "room", "topic": topic},
        )

        logger.info(f"Created room '{topic}' (id={rid})")
        return room

    def find_rooms(self, query: str, top_k: int = 3) -> List[Room]:
        """Find rooms most relevant to a query (contextual priming)."""
        results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,  # Search wider to filter
        )

        rooms = []
        for vec_id, score in results:
            if vec_id.startswith("room:"):
                rid = vec_id[5:]
                room = self.rooms.get(rid)
                if room:
                    rooms.append(room)
                    if len(rooms) >= top_k:
                        break
        return rooms

    def find_or_create_room(self, content: str, threshold: float = 0.6) -> Room:
        """Find a suitable room for content, or create a new one."""
        if not self.rooms:
            # First room — use the LLM to generate a topic name instead of content
            topic = content[:50].strip()
            return self.create_room(topic)

        # Find most similar room
        embedding = self.vector_store.embed(content)
        best_room = None
        best_score = 0.0

        for rid, room_emb in self._room_embeddings.items():
            score = float(np.dot(embedding, room_emb))
            if score > best_score:
                best_score = score
                best_room = self.rooms.get(rid)

        if best_room and best_score >= threshold:
            return best_room

        # No good match — create new room
        topic = content[:50].strip()
        return self.create_room(topic)

    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by ID."""
        return self.rooms.get(room_id)

    # ── Memory Placement ─────────────────────────────────

    def place_memory(self, memory: Memory, room: Optional[Room] = None) -> Room:
        """Place a memory in the palace (in a specific room or auto-assigned)."""
        with self._lock:
            if room is None:
                room = self.find_or_create_room(memory.content)

            memory.room_id = room.id
            room.memory_ids.append(memory.id)
            self.memories[memory.id] = memory

            # Store embedding in vector store
            if memory.embedding is None:
                memory.embedding = self.vector_store.embed(memory.content).tolist()

            self.vector_store.add(
                id=f"mem:{memory.id}",
                vector=np.array(memory.embedding),
                metadata={
                    "type": "memory",
                    "room_id": room.id,
                    "content": memory.content[:200],
                },
            )

            # Check if this is a landmark (high salience)
            if memory.salience.composite >= 0.8:
                self.landmarks.append(memory.id)
                # Keep landmarks bounded
                if len(self.landmarks) > 200:
                    self.landmarks = self.landmarks[-200:]

            # Update room centroid
            self._update_room_centroid(room)

        return room

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        return self.memories.get(memory_id)

    def get_room_memories(self, room_id: str) -> List[Memory]:
        """Get all active memories in a room."""
        room = self.rooms.get(room_id)
        if not room:
            return []
        return [
            self.memories[mid] for mid in room.memory_ids
            if mid in self.memories and self.memories[mid].status == MemoryStatus.ACTIVE
        ]

    # ── Edge (Hallway) Management ────────────────────────

    def link_rooms(
        self, room_a_id: str, room_b_id: str,
        relationship: str = "semantic", strength: float = 1.0,
    ) -> TypedEdge:
        """Create an associative bridge between two rooms."""
        with self._lock:
            # Check if edge already exists
            for edge in self._adj.get(room_a_id, []):
                if edge.target_room_id == room_b_id and edge.relationship == relationship:
                    edge.strength = max(edge.strength, strength)
                    return edge

            edge = TypedEdge(
                source_room_id=room_a_id,
                target_room_id=room_b_id,
                relationship=relationship,
                strength=strength,
            )
            self._adj.setdefault(room_a_id, []).append(edge)

            # Also create reverse edge (hallways are bidirectional)
            reverse = TypedEdge(
                source_room_id=room_b_id,
                target_room_id=room_a_id,
                relationship=relationship,
                strength=strength,
            )
            self._adj.setdefault(room_b_id, []).append(reverse)

            logger.debug(
                f"Linked rooms {room_a_id} <-> {room_b_id} ({relationship})"
            )
            return edge

    def get_neighbors(self, room_id: str) -> List[Tuple[Room, TypedEdge]]:
        """Get rooms connected to a given room via hallways."""
        neighbors = []
        for edge in self._adj.get(room_id, []):
            neighbor = self.rooms.get(edge.target_room_id)
            if neighbor:
                neighbors.append((neighbor, edge))
        return neighbors

    # ── Multi-Hop Search ─────────────────────────────────

    def search(
        self, query: str, top_k: int = 10, max_hops: int = 1
    ) -> List[Memory]:
        """
        Multi-hop associative search through the palace.
        
        1. Find relevant rooms (entry points)
        2. Search within those rooms
        3. Follow hallways to connected rooms (contextual priming)
        4. Search neighbors too
        """
        entry_rooms = self.find_rooms(query, top_k=3)
        candidates: Dict[str, Tuple[Memory, float, int]] = {}  # id -> (memory, score, hops)

        query_embedding = self.vector_store.embed(query)

        for room in entry_rooms:
            room.visit_count += 1
            room.last_visited = datetime.now()

            # Search within entry room (hop=0)
            for mem in self.get_room_memories(room.id):
                if mem.embedding:
                    score = float(np.dot(query_embedding, np.array(mem.embedding)))
                    if mem.id not in candidates or score > candidates[mem.id][1]:
                        candidates[mem.id] = (mem, score, 0)

            # Follow hallways for multi-hop (hop=1)
            if max_hops >= 1:
                for neighbor, edge in self.get_neighbors(room.id):
                    neighbor.visit_count += 1
                    neighbor.last_visited = datetime.now()

                    for mem in self.get_room_memories(neighbor.id):
                        if mem.embedding:
                            score = float(np.dot(query_embedding, np.array(mem.embedding)))
                            # Discount neighbor scores slightly
                            score *= 0.85 * edge.strength
                            if mem.id not in candidates or score > candidates[mem.id][1]:
                                candidates[mem.id] = (mem, score, 1)

        # Sort by score and take top_k
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

    def search_all_rooms(
        self, query_embedding: np.ndarray, min_strength: float = 0.5
    ) -> List[Memory]:
        """Search across all rooms — used by Ambient Monitor."""
        candidates = []
        for mem in self.memories.values():
            if mem.status != MemoryStatus.ACTIVE or mem.strength < min_strength:
                continue
            if mem.embedding:
                score = float(np.dot(query_embedding, np.array(mem.embedding)))
                if score > 0.4:
                    mem.retrieval_score = score
                    candidates.append(mem)

        candidates.sort(key=lambda m: m.retrieval_score, reverse=True)
        return candidates[:20]

    # ── Palace Health ────────────────────────────────────

    def health(self) -> dict:
        """Palace health metrics for monitoring and defragmentation."""
        if not self.rooms:
            return {"room_count": 0, "memory_count": 0, "edge_count": 0}

        room_sizes = [len(r.memory_ids) for r in self.rooms.values()]
        total_edges = sum(len(edges) for edges in self._adj.values())
        connected_rooms = set(self._adj.keys())

        return {
            "room_count": len(self.rooms),
            "memory_count": len(self.memories),
            "edge_count": total_edges // 2,  # Bidirectional
            "landmark_count": len(self.landmarks),
            "avg_room_size": sum(room_sizes) / len(room_sizes) if room_sizes else 0,
            "max_room_size": max(room_sizes) if room_sizes else 0,
            "connected_rooms": len(connected_rooms),
            "isolated_rooms": len(self.rooms) - len(connected_rooms),
        }

    # ── Internal ─────────────────────────────────────────

    def _update_room_centroid(self, room: Room):
        """Recalculate room centroid from its members."""
        embeddings = []
        for mid in room.memory_ids:
            mem = self.memories.get(mid)
            if mem and mem.embedding:
                embeddings.append(np.array(mem.embedding))
        
        if embeddings:
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            room.centroid_embedding = centroid
            self._room_embeddings[room.id] = centroid
            # Update in vector store
            self.vector_store.add(
                id=f"room:{room.id}",
                vector=centroid,
                metadata={"type": "room", "topic": room.topic},
            )

    def shared_memories(self) -> List[Memory]:
        """Return only ACTIVE + SHARED memories — safe for team consolidation sync."""
        return [
            m for m in self.memories.values()
            if m.status == MemoryStatus.ACTIVE and m.visibility == Visibility.SHARED
        ]

    # ── Persistence ──────────────────────────────────────

    @staticmethod
    def _migrate(state: dict) -> dict:
        """Upgrade palace state dict to PALACE_SCHEMA_VERSION in-place."""
        version = state.get("schema_version", 0)
        if version == PALACE_SCHEMA_VERSION:
            return state
        if version < 1:
            # v0→v1 was never shipped; all pre-versioned files are treated as v0
            # and fall through to the v2 migration below which handles them fully.
            logger.info("Migrating palace from schema v0 → v1 (no-op; handled by v2 migration)")
        if version < 2:
            logger.info("Migrating palace from schema v1 → v2 (adding visibility field)")
            for r in state.get("rooms", {}).values():
                r.setdefault("visibility", "shared")
            for m in state.get("memories", {}).values():
                m.setdefault("visibility", "shared")
        if version < 3:
            logger.info("Migrating palace from schema v2 → v3 (stripping inline embeddings)")
            for m in state.get("memories", {}).values():
                m.pop("embedding", None)
        state["schema_version"] = PALACE_SCHEMA_VERSION
        return state

    def save(self, path: Optional[str] = None):
        """Save palace state to disk."""
        save_path = path or self.storage_path
        if not save_path:
            return
        os.makedirs(save_path, exist_ok=True)

        # Flatten adjacency dict to edge list for serialization
        all_edges = []
        for edges in self._adj.values():
            all_edges.extend(e.to_dict() for e in edges)

        state = {
            "schema_version": PALACE_SCHEMA_VERSION,
            "rooms": {rid: r.to_dict() for rid, r in self.rooms.items()},
            "edges": all_edges,
            "landmarks": self.landmarks,
            "memories": {mid: m.to_dict() for mid, m in self.memories.items()},
        }

        # Atomic write: write to temp file, then rename
        target = os.path.join(save_path, "palace.json")
        tmp_target = target + ".tmp"
        with open(tmp_target, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_target, target)
        
        self.vector_store.save()
        logger.info(f"Saved palace with {len(self.rooms)} rooms, {len(self.memories)} memories")

    def _load(self, path: str):
        """Load palace state from disk."""
        palace_file = os.path.join(path, "palace.json")
        if not os.path.exists(palace_file):
            return
        try:
            with open(palace_file, "r") as f:
                state = json.load(f)

            state = self._migrate(state)

            # Reconstruct rooms
            for rid, rdata in state.get("rooms", {}).items():
                room = Room(
                    id=rid,
                    topic=rdata["topic"],
                    visit_count=rdata.get("visit_count", 0),
                    last_visited=datetime.fromisoformat(rdata.get("last_visited", datetime.now().isoformat())),
                    memory_ids=rdata.get("memory_ids", []),
                    visibility=Visibility(rdata.get("visibility", "shared")),
                )
                self.rooms[rid] = room

            # Reconstruct edges into adjacency dict
            for edata in state.get("edges", []):
                edge = TypedEdge(
                    source_room_id=edata["source"],
                    target_room_id=edata["target"],
                    relationship=edata["relationship"],
                    strength=edata.get("strength", 1.0),
                )
                self._adj.setdefault(edge.source_room_id, []).append(edge)

            # Reconstruct memories
            for mid, mdata in state.get("memories", {}).items():
                from smriti_memcore.models import MemorySource, Modality, SalienceScore
                salience_data = mdata.get("salience", {})
                memory = Memory(
                    id=mid,
                    content=mdata.get("content", ""),
                    embedding=None,  # populated from VectorStore after all memories are loaded (see below)
                    modality=Modality(mdata.get("modality", "text")),
                    source=MemorySource(mdata.get("source", "direct")),
                    status=MemoryStatus(mdata.get("status", "active")),
                    room_id=mdata.get("room_id"),
                    strength=mdata.get("strength", 1.0),
                    confidence=mdata.get("confidence", 1.0),
                    salience=SalienceScore(
                        surprise=salience_data.get("surprise", 0.0),
                        relevance=salience_data.get("relevance", 0.0),
                        emotional=salience_data.get("emotional", 0.0),
                        novelty=salience_data.get("novelty", 0.0),
                        utility=salience_data.get("utility", 0.0),
                    ),
                    creation_time=datetime.fromisoformat(mdata["creation_time"]) if "creation_time" in mdata else datetime.now(),
                    last_accessed=datetime.fromisoformat(mdata["last_accessed"]) if "last_accessed" in mdata else datetime.now(),
                    access_count=mdata.get("access_count", 0),
                    reflection_level=mdata.get("reflection_level", 0),
                    associations=mdata.get("associations", []),
                    metadata=mdata.get("metadata", {}),
                    # Spaced-repetition state — restored from disk
                    next_review=datetime.fromisoformat(mdata["next_review"]) if mdata.get("next_review") else None,
                    consecutive_successful_reviews=mdata.get("consecutive_successful_reviews", 0),
                    # Conflict tracking
                    superseded_by=mdata.get("superseded_by"),
                    # Visibility
                    visibility=Visibility(mdata.get("visibility", "shared")),
                )
                self.memories[mid] = memory

            # Repopulate in-memory embeddings from VectorStore (not persisted in palace.json since v3)
            for mid, memory in self.memories.items():
                if memory.embedding is None:
                    entry = self.vector_store.get(f"mem:{mid}")
                    if entry is not None:
                        memory.embedding = entry.vector.tolist()

            self.landmarks = state.get("landmarks", [])

            # Rebuild room embeddings from loaded memories
            for room in self.rooms.values():
                self._update_room_centroid(room)

            logger.info(f"Loaded palace with {len(self.rooms)} rooms, {len(self.memories)} memories")
        except Exception as e:
            logger.error(f"Failed to load palace: {e}")
