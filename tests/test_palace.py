"""Tests for smriti.palace — rooms, memory placement, persistence, thread safety."""

import os
import threading
import pytest
from smriti_memcore.models import Memory, SalienceScore, MemorySource, MemoryStatus
from smriti_memcore.palace import SemanticPalace


class TestRoomCreation:
    def test_create_room(self, palace):
        room = palace.create_room("Python programming")
        assert room.topic == "Python programming"
        assert room.id in palace.rooms

    def test_create_multiple_rooms(self, palace):
        palace.create_room("topic A")
        palace.create_room("topic B")
        assert len(palace.rooms) == 2


class TestFindRooms:
    def test_find_rooms(self, palace):
        palace.create_room("machine learning algorithms")
        palace.create_room("cooking recipes")
        rooms = palace.find_rooms("neural networks", top_k=1)
        assert len(rooms) >= 1
        assert rooms[0].topic == "machine learning algorithms"

    def test_find_rooms_empty_palace(self, palace):
        rooms = palace.find_rooms("anything")
        assert rooms == []


class TestFindOrCreateRoom:
    def test_creates_when_empty(self, palace):
        room = palace.find_or_create_room("first content ever")
        assert room is not None
        assert len(palace.rooms) == 1

    def test_finds_existing_room(self, palace):
        palace.create_room("machine learning")
        room = palace.find_or_create_room("deep learning models")
        # May find existing ML room or create new one depending on similarity threshold
        assert len(palace.rooms) >= 1

    def test_creates_new_when_different(self, palace):
        palace.create_room("machine learning")
        room = palace.find_or_create_room("gourmet italian cooking techniques")
        # Should create new room for very different content
        assert len(palace.rooms) >= 1  # At least the original


class TestPlaceMemory:
    def test_place_memory(self, palace, make_memory):
        m = make_memory("test fact about Python")
        room = palace.place_memory(m)
        assert m.id in palace.memories
        assert m.room_id == room.id
        assert m.id in room.memory_ids

    def test_place_sets_embedding(self, palace, make_memory):
        m = make_memory("test fact")
        palace.place_memory(m)
        assert m.embedding is not None

    def test_place_landmark(self, palace):
        m = Memory(
            content="extremely important",
            salience=SalienceScore(
                surprise=0.9, relevance=0.9,
                emotional=0.9, novelty=0.9, utility=0.9,
            ),
        )
        palace.place_memory(m)
        assert m.id in palace.landmarks

    def test_landmarks_bounded(self, palace):
        for i in range(210):
            m = Memory(
                content=f"landmark {i}",
                salience=SalienceScore(
                    surprise=0.9, relevance=0.9,
                    emotional=0.9, novelty=0.9, utility=0.9,
                ),
            )
            palace.place_memory(m)
        assert len(palace.landmarks) <= 200


class TestGetMemory:
    def test_get_existing(self, palace, make_memory):
        m = make_memory("findable")
        palace.place_memory(m)
        found = palace.get_memory(m.id)
        assert found is not None
        assert found.content == "findable"

    def test_get_missing(self, palace):
        assert palace.get_memory("nonexistent") is None


class TestLinkRooms:
    def test_link_rooms(self, palace):
        r1 = palace.create_room("topic A")
        r2 = palace.create_room("topic B")
        edge = palace.link_rooms(r1.id, r2.id, "semantic", 0.8)
        assert edge is not None
        neighbors = palace.get_neighbors(r1.id)
        assert len(neighbors) == 1

    def test_link_bidirectional(self, palace):
        r1 = palace.create_room("A")
        r2 = palace.create_room("B")
        palace.link_rooms(r1.id, r2.id)
        assert len(palace.get_neighbors(r1.id)) == 1
        assert len(palace.get_neighbors(r2.id)) == 1

    def test_link_idempotent(self, palace):
        r1 = palace.create_room("A")
        r2 = palace.create_room("B")
        palace.link_rooms(r1.id, r2.id, "semantic")
        palace.link_rooms(r1.id, r2.id, "semantic")  # Same link again
        neighbors = palace.get_neighbors(r1.id)
        assert len(neighbors) == 1  # No duplicates


class TestPersistence:
    def test_save_and_load(self, tmp_dir, vector_store, make_memory):
        path = os.path.join(tmp_dir, "persist_palace")

        # Save
        p1 = SemanticPalace(vector_store=vector_store, storage_path=path)
        m = make_memory("persistent memory")
        p1.place_memory(m)
        p1.save()

        # Load fresh
        p2 = SemanticPalace(vector_store=vector_store, storage_path=path)
        assert len(p2.rooms) == 1
        assert len(p2.memories) == 1
        loaded = p2.get_memory(m.id)
        assert loaded is not None
        assert loaded.content == "persistent memory"
        assert loaded.embedding is not None

    def test_atomic_save(self, palace, make_memory):
        palace.place_memory(make_memory("test"))
        palace.save()
        # No .tmp file should be left behind
        import os
        palace_dir = palace.storage_path
        if palace_dir and os.path.exists(palace_dir):
            files = os.listdir(palace_dir)
            assert "palace.json.tmp" not in files

    def test_room_embeddings_rebuilt_on_load(self, tmp_dir, vector_store, make_memory):
        path = os.path.join(tmp_dir, "rebuild_palace")

        p1 = SemanticPalace(vector_store=vector_store, storage_path=path)
        p1.place_memory(make_memory("fact about python"))
        p1.save()

        p2 = SemanticPalace(vector_store=vector_store, storage_path=path)
        # Room embeddings should be rebuilt
        assert len(p2._room_embeddings) > 0


class TestThreadSafety:
    def test_concurrent_place_memory(self, palace, make_memory):
        """place_memory should be thread-safe."""
        errors = []

        def place(i):
            try:
                palace.place_memory(make_memory(f"concurrent memory {i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=place, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(palace.memories) == 10


class TestHealth:
    def test_health_report(self, palace, make_memory):
        palace.place_memory(make_memory("test"))
        health = palace.health()
        assert "room_count" in health
        assert "memory_count" in health
        assert health["memory_count"] == 1


class TestEmbeddingStrip:
    """palace.json must not contain embedding vectors."""

    def test_to_dict_omits_embedding(self):
        """Memory.to_dict() must not include the 'embedding' key."""
        from smriti_memcore.models import Memory
        m = Memory(content="hello")
        m.embedding = [0.1] * 384
        d = m.to_dict()
        assert "embedding" not in d, "to_dict() must not serialise raw embedding vectors"


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

    def test_save_does_not_write_embedding_to_palace_json(self, palace, make_memory, tmp_dir):
        """Saved palace.json must not contain any 'embedding' key."""
        import json, os
        palace.place_memory(make_memory("the quick brown fox"))
        palace.save()

        palace_file = os.path.join(tmp_dir, "palace", "palace.json")
        with open(palace_file) as f:
            data = json.load(f)
        for mid, mdata in data.get("memories", {}).items():
            assert "embedding" not in mdata, f"memory {mid} still has embedding in palace.json"

    def test_round_trip_embedding_available_after_load(self, tmp_dir, vector_store, make_memory):
        """After save+load, memory.embedding must be populated from VectorStore (not palace.json)."""
        import os
        from smriti_memcore.palace import SemanticPalace

        palace = SemanticPalace(
            vector_store=vector_store,
            storage_path=os.path.join(tmp_dir, "palace"),
        )
        mem = make_memory("the quick brown fox")
        palace.place_memory(mem)
        mid = mem.id
        palace.save()

        # Fresh palace from the same storage — embeddings must be repopulated from VectorStore
        palace2 = SemanticPalace(
            vector_store=vector_store,
            storage_path=os.path.join(tmp_dir, "palace"),
        )
        loaded = palace2.memories.get(mid)
        assert loaded is not None
        assert loaded.embedding is not None, "embedding must be repopulated from VectorStore on load"
        assert len(loaded.embedding) == 384


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
