"""
SMRITI v2 — FTS Index.
SQLite FTS5 wrapper for keyword candidate generation.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import List, Tuple

from smriti_memcore.models import Memory

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize = 'porter ascii'
);
"""


class FTSIndex:
    def __init__(self, storage_path: str):
        if storage_path == ":memory:":
            fts_db_path = ":memory:"
        else:
            fts_db_path = os.path.join(storage_path, "fts.db")
        self._path = fts_db_path
        self._conn = self._open(fts_db_path)

    def _open(self, path: str) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.commit()
            return conn
        except sqlite3.DatabaseError:
            if path != ":memory:":
                logger.warning(f"Corrupt fts.db at {path} — deleting and rebuilding")
                try:
                    os.remove(path)
                except OSError:
                    pass
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.commit()
            return conn

    def needs_rebuild(self, active_count: int) -> bool:
        row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] != active_count

    def add(self, memory_id: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO memories(memory_id, content) VALUES (?, ?)",
            (memory_id, content),
        )
        self._conn.commit()

    def remove(self, memory_id: str) -> None:
        self._conn.execute(
            "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
        )
        self._conn.commit()

    def rebuild(self, memories: List[Memory]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM memories")
            self._conn.executemany(
                "INSERT INTO memories(memory_id, content) VALUES (?, ?)",
                [(m.id, m.content) for m in memories],
            )

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not clean_query:
            return []
        rows = self._conn.execute(
            "SELECT memory_id, bm25(memories) FROM memories "
            "WHERE memories MATCH ? ORDER BY bm25(memories) LIMIT ?",
            (clean_query, top_k),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
