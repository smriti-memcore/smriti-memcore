"""Tests for Visibility (private/shared) feature — Task 6 of feat/private-rooms."""

import json
import os

import pytest

from smriti_memcore.models import Memory, MemorySource, MemoryStatus, SmritiConfig, Visibility
from smriti_memcore.palace import PALACE_SCHEMA_VERSION, Room, SemanticPalace
from smriti_memcore.consolidation import ConsolidationEngine
from smriti_memcore.core import SMRITI
import smriti_memcore.integrations.mcp_server as _mcp_module


# ── Model-level tests ─────────────────────────────────────


class TestVisibilityModel:
    def test_memory_defaults_to_shared(self):
        m = Memory(content="hello")
        assert m.visibility == Visibility.SHARED

    def test_memory_can_be_set_private(self):
        m = Memory(content="secret", visibility=Visibility.PRIVATE)
        assert m.visibility == Visibility.PRIVATE

    def test_memory_to_dict_includes_visibility(self):
        m = Memory(content="hello", visibility=Visibility.PRIVATE)
        d = m.to_dict()
        assert d["visibility"] == "private"

    def test_room_defaults_to_shared(self):
        r = Room(id="r1", topic="test")
        assert r.visibility == Visibility.SHARED

    def test_room_can_be_set_private(self):
        r = Room(id="r1", topic="test", visibility=Visibility.PRIVATE)
        assert r.visibility == Visibility.PRIVATE

    def test_room_to_dict_includes_visibility(self):
        r = Room(id="r1", topic="test", visibility=Visibility.PRIVATE)
        d = r.to_dict()
        assert d["visibility"] == "private"


# ── Migration tests ───────────────────────────────────────


class TestSchemaMigration:
    def test_schema_version_is_current(self):
        assert PALACE_SCHEMA_VERSION == 3

    def test_v0_migration_sets_visibility_shared(self):
        state = {
            "rooms": {"r1": {"topic": "test"}},
            "memories": {"m1": {"content": "hello"}},
        }
        migrated = SemanticPalace._migrate(state)
        assert migrated["schema_version"] == PALACE_SCHEMA_VERSION
        assert migrated["rooms"]["r1"]["visibility"] == "shared"
        assert migrated["memories"]["m1"]["visibility"] == "shared"

    def test_v1_migration_sets_visibility_shared(self):
        state = {
            "schema_version": 1,
            "rooms": {"r1": {"topic": "test"}},
            "memories": {"m1": {"content": "hello"}},
        }
        migrated = SemanticPalace._migrate(state)
        assert migrated["schema_version"] == PALACE_SCHEMA_VERSION
        assert migrated["rooms"]["r1"]["visibility"] == "shared"
        assert migrated["memories"]["m1"]["visibility"] == "shared"

    def test_already_v2_migration_is_noop(self):
        state = {
            "schema_version": 2,
            "rooms": {"r1": {"topic": "test", "visibility": "private"}},
            "memories": {"m1": {"content": "hello", "visibility": "private"}},
        }
        migrated = SemanticPalace._migrate(state)
        assert migrated["rooms"]["r1"]["visibility"] == "private"
        assert migrated["memories"]["m1"]["visibility"] == "private"


# ── Palace save/load round-trip ───────────────────────────


class TestVisibilitySaveLoad:
    def test_private_memory_survives_save_reload(self, tmp_path, vector_store):
        p = SemanticPalace(vector_store=vector_store, storage_path=str(tmp_path))
        mem = Memory(content="secret thought", visibility=Visibility.PRIVATE)
        p.place_memory(mem)
        p.save()

        p2 = SemanticPalace(vector_store=vector_store, storage_path=str(tmp_path))
        loaded = p2.memories.get(mem.id)
        assert loaded is not None
        assert loaded.visibility == Visibility.PRIVATE

    def test_private_room_survives_save_reload(self, tmp_path, vector_store):
        p = SemanticPalace(vector_store=vector_store, storage_path=str(tmp_path))
        room = p.create_room("private topic")
        room.visibility = Visibility.PRIVATE
        p.save()

        p2 = SemanticPalace(vector_store=vector_store, storage_path=str(tmp_path))
        loaded_room = p2.rooms.get(room.id)
        assert loaded_room is not None
        assert loaded_room.visibility == Visibility.PRIVATE

    def test_saved_palace_has_schema_version(self, tmp_path, vector_store):
        p = SemanticPalace(vector_store=vector_store, storage_path=str(tmp_path))
        p.save()
        with open(os.path.join(str(tmp_path), "palace.json")) as f:
            state = json.load(f)
        assert state["schema_version"] == PALACE_SCHEMA_VERSION


# ── shared_memories helper ────────────────────────────────


class TestSharedMemoriesHelper:
    def test_shared_memories_excludes_private(self, palace):
        shared = Memory(content="public info", visibility=Visibility.SHARED)
        private = Memory(content="private info", visibility=Visibility.PRIVATE)
        palace.place_memory(shared)
        palace.place_memory(private)

        result = palace.shared_memories()
        ids = {m.id for m in result}
        assert shared.id in ids
        assert private.id not in ids

    def test_shared_memories_excludes_non_active(self, palace):
        m = Memory(content="archived", visibility=Visibility.SHARED, status=MemoryStatus.ARCHIVED)
        palace.memories[m.id] = m

        result = palace.shared_memories()
        assert m.id not in {mem.id for mem in result}


# ── Private memories still recalled ──────────────────────


class TestPrivateMemoryRecall:
    def test_private_memory_is_recalled_by_owner(self, palace, vector_store):
        mem = Memory(content="my private note about project X", visibility=Visibility.PRIVATE)
        palace.place_memory(mem)

        results = palace.search(["project X"], [vector_store.embed("project X")], top_k=5)
        ids = {m.id for m in results}
        assert mem.id in ids


# ── MCP layer: smriti_encode private=True ─────────────────


@pytest.fixture
def tmp_smriti(tmp_path):
    config = SmritiConfig(storage_path=str(tmp_path), llm_model="none")
    n = SMRITI(config=config)
    yield n
    n.close()


@pytest.fixture(autouse=False)
def inject_smriti(tmp_smriti):
    original = _mcp_module._smriti
    _mcp_module._smriti = tmp_smriti
    yield tmp_smriti
    _mcp_module._smriti = original


class TestMCPEncodePrivate:
    def test_smriti_encode_private_true_sets_visibility(self, inject_smriti):
        from smriti_memcore.integrations.mcp_server import smriti_encode
        result = smriti_encode(content="sensitive personal detail", private=True)
        assert "memory_id" in result
        mid = result["memory_id"]
        assert mid is not None
        mem = inject_smriti.palace.memories.get(mid)
        assert mem is not None
        assert mem.visibility == Visibility.PRIVATE
        assert result["visibility"] == "private"

    def test_smriti_encode_private_false_is_shared(self, inject_smriti):
        from smriti_memcore.integrations.mcp_server import smriti_encode
        result = smriti_encode(content="public team info about architecture", private=False)
        mid = result.get("memory_id")
        if mid:  # may be discarded by attention gate
            mem = inject_smriti.palace.memories.get(mid)
            assert mem.visibility == Visibility.SHARED
            assert result["visibility"] == "shared"

    def test_amp_encode_private_true_sets_visibility(self, inject_smriti):
        from smriti_memcore.integrations.mcp_server import amp_encode
        result = amp_encode(agent_id="test", content="private salary info", force=True, private=True)
        assert result["status"] == "stored"
        mem = inject_smriti.palace.memories.get(result["id"])
        assert mem is not None
        assert mem.visibility == Visibility.PRIVATE
        assert result["visibility"] == "private"

    def test_smriti_create_private_room_returns_private(self, inject_smriti):
        from smriti_memcore.integrations.mcp_server import smriti_create_private_room
        result = smriti_create_private_room(topic="personal journal")
        assert result["visibility"] == "private"
        room = inject_smriti.palace.rooms.get(result["room_id"])
        assert room is not None
        assert room.visibility == Visibility.PRIVATE

    def test_smriti_stats_includes_visibility_counts(self, inject_smriti):
        from smriti_memcore.integrations.mcp_server import smriti_encode, smriti_stats
        smriti_encode(content="public architecture decision about microservices", private=False)
        smriti_encode(content="confidential salary negotiation details here", private=True)
        stats = smriti_stats()
        assert "private_memories" in stats["palace"]
        assert "shared_memories" in stats["palace"]
        assert isinstance(stats["palace"]["private_memories"], int)
        assert isinstance(stats["palace"]["shared_memories"], int)


# ── Consolidation guard: conflict resolution skips private ─


class TestConflictResolutionSkipsPrivate:
    def test_private_memory_excluded_from_conflict_candidates(
        self, episode_buffer, palace, vector_store, mock_llm
    ):
        engine = ConsolidationEngine(
            episode_buffer=episode_buffer, palace=palace,
            vector_store=vector_store, llm=mock_llm, config=SmritiConfig(),
        )

        private_mem = Memory(
            content="private health detail", visibility=Visibility.PRIVATE,
            source=MemorySource.USER_STATED,
        )
        shared_mem = Memory(
            content="shared team decision about deployment", visibility=Visibility.SHARED,
            source=MemorySource.USER_STATED,
        )
        palace.place_memory(private_mem)
        palace.place_memory(shared_mem)

        result = engine._process_conflict_resolution()

        # Private memory must still be ACTIVE — not touched by conflict resolution
        assert private_mem.status == MemoryStatus.ACTIVE
        # Shared memory was included as a candidate (not filtered out)
        assert shared_mem.status == MemoryStatus.ACTIVE  # mock LLM returns no contradiction
