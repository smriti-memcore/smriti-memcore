"""Tests for smriti_memcore.ui.server — episode feed data layer."""
import json
import sqlite3
try:
    import pytest
except ImportError:
    class _MockPytest:
        @staticmethod
        def approx(val):
            class _Approx:
                def __init__(self, v): self.v = v
                def __eq__(self, o): return abs(self.v - o) < 1e-4
            return _Approx(val)
    pytest = _MockPytest()
from pathlib import Path
from smriti_memcore.ui.server import _read_episodes


def _make_db(tmp_path, rows):
    """Create a minimal episodes.db with given rows."""
    db = tmp_path / "episodes" / "episodes.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            content TEXT,
            timestamp TEXT,
            salience_json TEXT,
            source TEXT,
            trajectory_id TEXT,
            trajectory_step INTEGER,
            reflections_json TEXT,
            consolidated INTEGER DEFAULT 0,
            metadata_json TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return str(tmp_path)


def test_read_episodes_returns_list_of_dicts(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "content one", "2026-04-12T10:00:00", '{"composite": 0.5}',
         "user_stated", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert isinstance(result, list)
    assert len(result) == 1
    ep = result[0]
    assert ep["id"] == "id-1"
    assert ep["content"] == "content one"
    assert ep["timestamp"] == "2026-04-12T10:00:00"
    assert ep["source"] == "user_stated"
    assert ep["salience"] == pytest.approx(0.5)
    assert ep["consolidated"] is True  # must be Python bool, not int 1


def test_read_episodes_consolidated_cast_to_bool(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-0", "pending ep", "2026-04-12T09:00:00", '{"composite":0.1}',
         "direct", None, 0, None, 0, None),
        ("id-1", "done ep",   "2026-04-12T10:00:00", '{"composite":0.9}',
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    by_id = {ep["id"]: ep for ep in result}
    assert by_id["id-0"]["consolidated"] is False
    assert by_id["id-1"]["consolidated"] is True


def test_read_episodes_null_salience_json(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "no sal", "2026-04-12T10:00:00", None,
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert result[0]["salience"] == pytest.approx(0.0)


def test_read_episodes_malformed_salience_json(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "bad sal", "2026-04-12T10:00:00", "not-json",
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert result[0]["salience"] == pytest.approx(0.0)


def test_read_episodes_ordered_newest_first(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "old", "2026-04-10T10:00:00", '{"composite":0.3}', "direct", None, 0, None, 1, None),
        ("id-2", "new", "2026-04-12T10:00:00", '{"composite":0.7}', "direct", None, 0, None, 1, None),
        ("id-3", "mid", "2026-04-11T10:00:00", '{"composite":0.5}', "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert [ep["id"] for ep in result] == ["id-2", "id-3", "id-1"]


def test_read_episodes_missing_db_returns_empty(tmp_path):
    result = _read_episodes(str(tmp_path))  # no episodes/ subdir
    assert result == []


import io
from smriti_memcore.ui.server import _Handler


def _simulate_get(storage_path: str, path: str):
    """Simulate a GET request through _Handler without requiring socket bindings."""
    class SimulatedHandler(_Handler):
        def __init__(self, p, s):
            self.path = p
            self.storage_path = s
            self.wfile = io.BytesIO()
            self.status = None
            self.headers = {}

        def send_response(self, code):
            self.status = code

        def send_header(self, k, v):
            self.headers[k] = v

        def end_headers(self):
            pass

    h = SimulatedHandler(path, storage_path)
    h.do_GET()
    return h.status, h.headers, h.wfile.getvalue()


def test_api_episodes_endpoint_returns_json(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "test content", "2026-04-12T10:00:00", '{"composite":0.4}',
         "user_stated", None, 0, None, 1, None),
    ])
    status, headers, body = _simulate_get(storage, "/api/episodes")
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    data = json.loads(body)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "id-1"
    assert data[0]["consolidated"] is True


def test_api_episodes_empty_when_no_db(tmp_path):
    status, headers, body = _simulate_get(str(tmp_path), "/api/episodes")
    assert status == 200
    data = json.loads(body)
    assert data == []


def test_read_palace_nodes_and_edges(tmp_path):
    from smriti_memcore.ui.server import _read_palace
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir(parents=True)
    palace_json = {
        "rooms": {
            "r1": {"id": "r1", "topic": "Quantum", "visibility": "shared", "memory_ids": ["m1"], "visit_count": 3}
        },
        "memories": {
            "m1": {"id": "m1", "content": "Superposition state", "room_id": "r1", "strength": 0.88, "visibility": "shared", "reflection_level": 1}
        },
        "edges": []
    }
    with open(palace_dir / "palace.json", "w") as f:
        json.dump(palace_json, f)

    res = _read_palace(str(tmp_path))
    assert res["room_count"] == 1
    assert len(res["nodes"]) == 2  # 1 room + 1 memory
    assert len(res["edges"]) == 1  # 1 link to room
    room_node = next(n for n in res["nodes"] if n["type"] == "room")
    mem_node = next(n for n in res["nodes"] if n["type"] == "memory")
    assert room_node["id"] == "room:r1"
    assert mem_node["id"] == "mem:m1"
    assert mem_node["color"] == "#06b6d4"  # shared color


def test_api_graph_endpoint_returns_json(tmp_path):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir(parents=True)
    palace_json = {
        "rooms": {
            "r1": {"id": "r1", "topic": "Secret", "visibility": "private", "memory_ids": ["m1"], "visit_count": 1}
        },
        "memories": {
            "m1": {"id": "m1", "content": "Private key", "room_id": "r1", "strength": 1.0, "visibility": "private", "reflection_level": 0}
        },
        "edges": []
    }
    with open(palace_dir / "palace.json", "w") as f:
        json.dump(palace_json, f)

    status, headers, body = _simulate_get(str(tmp_path), "/api/graph")
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    data = json.loads(body)
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    mem_node = next(n for n in data["nodes"] if n["type"] == "memory")
    assert mem_node["color"] == "#f59e0b"  # private color


if __name__ == "__main__":
    import tempfile
    test_funcs = [
        test_read_episodes_returns_list_of_dicts,
        test_read_episodes_consolidated_cast_to_bool,
        test_read_episodes_null_salience_json,
        test_read_episodes_malformed_salience_json,
        test_read_episodes_ordered_newest_first,
        test_read_episodes_missing_db_returns_empty,
        test_read_palace_nodes_and_edges,
        test_api_episodes_endpoint_returns_json,
        test_api_episodes_empty_when_no_db,
        test_api_graph_endpoint_returns_json,
    ]
    passed = 0
    for fn in test_funcs:
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
            passed += 1
            print(f"✓ {fn.__name__}")

    print(f"\nAll {passed} tests completed successfully.")

