# Private Rooms — Implementation Plan

**Goal:** Add `visibility` field (`"private"` | `"shared"`) to `Room` and `Memory` so memories can be marked private and excluded from future team consolidation sync. Bump schema to v2. This is open-source infrastructure; team sync enforcement is enterprise.

**Branch:** `feat/private-rooms`  
**Worktree:** `.worktrees/feat-private-rooms`

---

## Task 1 — Add `Visibility` enum and fields to models.py

- [ ] Add `class Visibility(str, Enum): PRIVATE = "private"; SHARED = "shared"` to `smriti_memcore/models.py`
- [ ] Add `visibility: Visibility = Visibility.SHARED` field to `Memory` dataclass
- [ ] Add `visibility: Visibility = Visibility.SHARED` field to `Room` dataclass
- [ ] Add `"visibility": self.visibility.value` to `Memory.to_dict()`
- [ ] Add `"visibility": self.visibility.value` to `Room.to_dict()` (if Room has to_dict; otherwise handle in palace.py)
- [ ] Verify: `python3 -c "from smriti_memcore.models import Memory, Visibility; m = Memory(id='x', content='y'); print(m.visibility)"`

## Task 2 — Update palace.py: schema v2 migration + serialize/deserialize

- [ ] Bump `PALACE_SCHEMA_VERSION = 2` in `smriti_memcore/palace.py`
- [ ] Add v1→v2 migration block in `_migrate()`:
  ```python
  if version < 2:
      logger.info("Migrating palace from schema v1 → v2 (adding visibility field)")
      for r in state.get("rooms", {}).values():
          r.setdefault("visibility", "shared")
      for m in state.get("memories", {}).values():
          m.setdefault("visibility", "shared")
  ```
- [ ] In `_load()` Room reconstruction: add `visibility=Visibility(rdata.get("visibility", "shared"))`
- [ ] In `_load()` Memory reconstruction: add `visibility=Visibility(mdata.get("visibility", "shared"))`
- [ ] In `save()` Room serialization: add `"visibility": r.visibility.value` to room dict
- [ ] Verify save/load round-trip: write a private memory, save, reload, check visibility is preserved

## Task 3 — Add `smriti_encode` `private` parameter to mcp_server.py

- [ ] Add `private: bool = False` parameter to `smriti_encode` tool
- [ ] When `private=True`, set `memory.visibility = Visibility.PRIVATE` on the returned/stored memory after encoding
- [ ] Add `private: bool = False` parameter to `amp_encode` tool similarly
- [ ] Update startup log to mention visibility support

## Task 4 — Add `smriti_create_private_room` tool to mcp_server.py

- [ ] Add new MCP tool `smriti_create_private_room(topic: str) -> Dict`:
  - Creates a Room with `visibility=Visibility.PRIVATE`
  - Returns `{"room_id": ..., "topic": ..., "visibility": "private"}`
- [ ] Add `visibility` field to `smriti_stats` output (count of private vs shared memories)

## Task 5 — Guard consolidation engine against promoting private memories

- [ ] In `consolidation.py` `_process_chunking()`: filter out memories where `m.visibility == Visibility.PRIVATE` before clustering into rooms
- [ ] In `consolidation.py` `_process_conflict_resolution()`: skip private memories (they are personal, not subject to cross-memory contradiction resolution)
- [ ] Add a helper `SemanticPalace.shared_memories()` returning only `ACTIVE` + `SHARED` memories for use by consolidation

## Task 6 — Tests

- [ ] Add `tests/test_visibility.py`:
  - Test: encoding with `private=True` stores memory with `visibility=PRIVATE`
  - Test: private memories excluded from consolidation chunking
  - Test: private memories still recalled by owner (no filter at recall time)
  - Test: schema v1→v2 migration sets all existing memories/rooms to `shared`
  - Test: save/load round-trip preserves `visibility=PRIVATE`

## Task 7 — CHANGELOG

- [ ] Add entry under a new `## [Unreleased]` section (version TBD by Shivam):
  - **Added**: `Visibility` enum (`private`/`shared`) on `Memory` and `Room`
  - **Added**: `private=True` param on `smriti_encode` and `amp_encode`
  - **Added**: `smriti_create_private_room` MCP tool
  - **Changed**: `PALACE_SCHEMA_VERSION` bumped to 2; v1→v2 migration sets all existing data to `shared`
  - **Note**: Team sync enforcement (blocking private memories from team consolidation) is an enterprise feature; the privacy primitives are open source

---

## Verification

After all tasks: `pytest tests/ -q` — all existing 208 tests + new visibility tests pass.
