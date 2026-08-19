"""
SMRITI v2 — Episode Buffer.
Time-ordered event log with salience scores, trajectory tracking, 
and reflection annotations. SQLite-backed for persistence.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from smriti_memcore.models import Episode, MemorySource, SalienceScore
from smriti_memcore.vector_store import VectorStore

logger = logging.getLogger(__name__)


class EpisodeBuffer:
    """
    The agent's diary — raw experience log with salience scores.
    
    Stores episodes chronologically with:
    - 5-dimensional salience annotations
    - Trajectory tracking (MIRA-style sequences)
    - Reflection annotations
    - Source attribution
    - Semantic search via VectorStore
    """

    def __init__(self, storage_path: str, vector_store: VectorStore):
        self.storage_path = storage_path
        self.vector_store = vector_store
        self._episodes: Dict[str, Episode] = {}
        self._total_count: int = 0  # Total episodes in DB (including consolidated)
        self._lock = threading.Lock()
        self._closed = False

        # Persistent SQLite connection
        os.makedirs(storage_path, exist_ok=True)
        self._db_path = os.path.join(storage_path, "episodes.db")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                salience_json TEXT,
                source TEXT DEFAULT 'direct',
                trajectory_id TEXT,
                trajectory_step INTEGER DEFAULT 0,
                reflections_json TEXT DEFAULT '[]',
                consolidated INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}'
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trajectory ON episodes(trajectory_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_consolidated ON episodes(consolidated)
        """)
        self._conn.commit()

        # Load existing episodes into memory
        self._load_from_db()

    # ── Core Operations ──────────────────────────────────

    def add(self, episode: Episode) -> str:
        """Add an episode to the buffer."""
        with self._lock:
            self._episodes[episode.id] = episode
            self._total_count += 1

        # Store embedding in vector store
        if episode.embedding is not None:
            self.vector_store.add(
                id=f"ep:{episode.id}",
                vector=episode.embedding,
                metadata={"type": "episode", "content": episode.content[:200]},
            )
        elif episode.content:
            embedding = self.vector_store.embed(episode.content)
            episode.embedding = embedding.tolist()
            self.vector_store.add(
                id=f"ep:{episode.id}",
                vector=embedding,
                metadata={"type": "episode", "content": episode.content[:200]},
            )

        # Persist to SQLite
        self._save_episode(episode)

        logger.debug(f"Added episode {episode.id}: {episode.content[:80]}...")
        return episode.id

    def get(self, episode_id: str) -> Optional[Episode]:
        """Get a specific episode by ID. Falls back to DB for consolidated episodes."""
        ep = self._episodes.get(episode_id)
        if ep is not None:
            return ep

        # Fallback: query SQLite for consolidated episodes not in RAM
        try:
            cursor = self._conn.execute(
                "SELECT * FROM episodes WHERE id = ?", (episode_id,)
            )
            row = cursor.fetchone()
            if row:
                salience_data = json.loads(row[3]) if row[3] else {}
                return Episode(
                    id=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    salience=SalienceScore(
                        surprise=salience_data.get("surprise", 0),
                        relevance=salience_data.get("relevance", 0),
                        emotional=salience_data.get("emotional", 0),
                        novelty=salience_data.get("novelty", 0),
                        utility=salience_data.get("utility", 0),
                    ),
                    source=MemorySource(row[4]) if row[4] else MemorySource.DIRECT,
                    trajectory_id=row[5],
                    trajectory_step=row[6] or 0,
                    reflections=json.loads(row[7]) if row[7] else [],
                    consolidated=bool(row[8]),
                    metadata=json.loads(row[9]) if row[9] else {},
                )
        except Exception as e:
            logger.error(f"Failed to fetch episode {episode_id} from DB: {e}")
        return None

    def remove(self, episode_id: str):
        """Remove an episode from the buffer."""
        with self._lock:
            if episode_id in self._episodes:
                del self._episodes[episode_id]
                self.vector_store.remove(f"ep:{episode_id}")
                self._conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
                self._conn.commit()

    @property
    def count(self) -> int:
        """Total number of episodes (including consolidated in DB)."""
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM episodes")
            self._total_count = cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get total episode count: {e}")
        return self._total_count

    @property
    def unconsolidated_count(self) -> int:
        """Number of episodes not yet consolidated."""
        try:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE consolidated = 0"
            )
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get unconsolidated count: {e}")
            return sum(1 for ep in self._episodes.values() if not ep.consolidated)

    # ── Search & Query ───────────────────────────────────

    def search_semantic(self, query: str, top_k: int = 10) -> List[Episode]:
        """Search episodes by semantic similarity."""
        results = self.vector_store.search(query=query, top_k=top_k * 10)
        episodes = []
        for vec_id, score in results:
            if vec_id.startswith("ep:"):
                ep_id = vec_id[3:]
                # Fix: Must use self.get() rather than self._episodes.get() so it falls back to SQLite for consolidated episodes.
                ep = self.get(ep_id)
                if ep:
                    episodes.append(ep)
                    if len(episodes) >= top_k:
                        break
        return episodes

    def get_recent(self, n: int = 20) -> List[Episode]:
        """Get the N most recent episodes."""
        try:
            cursor = self._conn.execute(
                "SELECT id, content, timestamp, salience_json, source, trajectory_id, trajectory_step, reflections_json, consolidated, metadata_json FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,)
            )
            recent = []
            for row in cursor:
                salience_data = json.loads(row[3]) if row[3] else {}
                ep = Episode(
                    id=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    salience=SalienceScore(
                        surprise=salience_data.get("surprise", 0),
                        relevance=salience_data.get("relevance", 0),
                        emotional=salience_data.get("emotional", 0),
                        novelty=salience_data.get("novelty", 0),
                        utility=salience_data.get("utility", 0),
                    ),
                    source=MemorySource(row[4]) if row[4] else MemorySource.DIRECT,
                    trajectory_id=row[5],
                    trajectory_step=row[6] or 0,
                    reflections=json.loads(row[7]) if row[7] else [],
                    consolidated=bool(row[8]),
                    metadata=json.loads(row[9]) if row[9] else {},
                )
                recent.append(ep)
            return recent
        except Exception as e:
            logger.error(f"Failed to query recent episodes: {e}")
            
        sorted_eps = sorted(
            self._episodes.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )
        return sorted_eps[:n]

    def get_unconsolidated(self, limit: int = 100) -> List[Episode]:
        """Get unconsolidated episodes for the Consolidation Engine."""
        try:
            cursor = self._conn.execute(
                "SELECT id, content, timestamp, salience_json, source, trajectory_id, trajectory_step, reflections_json, consolidated, metadata_json FROM episodes WHERE consolidated = 0 ORDER BY timestamp"
            )
            unconsolidated = []
            for row in cursor:
                salience_data = json.loads(row[3]) if row[3] else {}
                ep = Episode(
                    id=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    salience=SalienceScore(
                        surprise=salience_data.get("surprise", 0),
                        relevance=salience_data.get("relevance", 0),
                        emotional=salience_data.get("emotional", 0),
                        novelty=salience_data.get("novelty", 0),
                        utility=salience_data.get("utility", 0),
                    ),
                    source=MemorySource(row[4]) if row[4] else MemorySource.DIRECT,
                    trajectory_id=row[5],
                    trajectory_step=row[6] or 0,
                    reflections=json.loads(row[7]) if row[7] else [],
                    consolidated=bool(row[8]),
                    metadata=json.loads(row[9]) if row[9] else {},
                )
                unconsolidated.append(ep)
            
            # Sync in-memory cache
            with self._lock:
                current_unconsolidated_ids = {ep.id for ep in unconsolidated}
                for ep_id in list(self._episodes.keys()):
                    if ep_id not in current_unconsolidated_ids:
                        del self._episodes[ep_id]
                for ep in unconsolidated:
                    self._episodes[ep.id] = ep
                    
            unconsolidated.sort(key=lambda e: e.timestamp)
            return unconsolidated[:limit]
        except Exception as e:
            logger.error(f"Failed to query unconsolidated episodes: {e}")
            
        unconsolidated = [
            ep for ep in self._episodes.values() if not ep.consolidated
        ]
        unconsolidated.sort(key=lambda e: e.timestamp)
        return unconsolidated[:limit]

    def get_by_trajectory(self, trajectory_id: str) -> List[Episode]:
        """Get all episodes in a trajectory sequence."""
        try:
            cursor = self._conn.execute(
                "SELECT id, content, timestamp, salience_json, source, trajectory_id, trajectory_step, reflections_json, consolidated, metadata_json FROM episodes WHERE trajectory_id = ? ORDER BY trajectory_step", (trajectory_id,)
            )
            trajectory = []
            for row in cursor:
                salience_data = json.loads(row[3]) if row[3] else {}
                ep = Episode(
                    id=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    salience=SalienceScore(
                        surprise=salience_data.get("surprise", 0),
                        relevance=salience_data.get("relevance", 0),
                        emotional=salience_data.get("emotional", 0),
                        novelty=salience_data.get("novelty", 0),
                        utility=salience_data.get("utility", 0),
                    ),
                    source=MemorySource(row[4]) if row[4] else MemorySource.DIRECT,
                    trajectory_id=row[5],
                    trajectory_step=row[6] or 0,
                    reflections=json.loads(row[7]) if row[7] else [],
                    consolidated=bool(row[8]),
                    metadata=json.loads(row[9]) if row[9] else {},
                )
                trajectory.append(ep)
            return trajectory
        except Exception as e:
            logger.error(f"Failed to query trajectory episodes: {e}")
            
        trajectory = [
            ep for ep in self._episodes.values()
            if ep.trajectory_id == trajectory_id
        ]
        trajectory.sort(key=lambda e: e.trajectory_step)
        return trajectory

    def search_trajectories(self, query: str, top_k: int = 5) -> List[List[Episode]]:
        """Find trajectory sequences similar to a query."""
        similar_eps = self.search_semantic(query, top_k=top_k * 2)
        trajectory_ids = set()
        for ep in similar_eps:
            if ep.trajectory_id:
                trajectory_ids.add(ep.trajectory_id)

        trajectories = []
        for tid in list(trajectory_ids)[:top_k]:
            traj = self.get_by_trajectory(tid)
            if traj:
                trajectories.append(traj)
        return trajectories

    def get_high_salience(self, min_composite: float = 0.7, limit: int = 50) -> List[Episode]:
        """Get high-salience episodes."""
        try:
            cursor = self._conn.execute(
                "SELECT id, content, timestamp, salience_json, source, trajectory_id, trajectory_step, reflections_json, consolidated, metadata_json FROM episodes"
            )
            high_sal = []
            for row in cursor:
                try:
                    salience_data = json.loads(row[3] or "{}")
                    composite = salience_data.get("composite", 0.0)
                except Exception:
                    composite = 0.0
                if composite >= min_composite:
                    ep = Episode(
                        id=row[0],
                        content=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        salience=SalienceScore(
                            surprise=salience_data.get("surprise", 0),
                            relevance=salience_data.get("relevance", 0),
                            emotional=salience_data.get("emotional", 0),
                            novelty=salience_data.get("novelty", 0),
                            utility=salience_data.get("utility", 0),
                        ),
                        source=MemorySource(row[4]) if row[4] else MemorySource.DIRECT,
                        trajectory_id=row[5],
                        trajectory_step=row[6] or 0,
                        reflections=json.loads(row[7]) if row[7] else [],
                        consolidated=bool(row[8]),
                        metadata=json.loads(row[9]) if row[9] else {},
                    )
                    high_sal.append(ep)
            high_sal.sort(key=lambda e: e.salience.composite, reverse=True)
            return high_sal[:limit]
        except Exception as e:
            logger.error(f"Failed to query high salience episodes: {e}")
            
        high_sal = [
            ep for ep in self._episodes.values()
            if ep.salience.composite >= min_composite
        ]
        high_sal.sort(key=lambda e: e.salience.composite, reverse=True)
        return high_sal[:limit]

    # ── Modification ─────────────────────────────────────

    def mark_consolidated(self, episode_ids: List[str]):
        """Mark episodes as consolidated and remove from in-memory cache."""
        with self._lock:
            if self._closed:
                return
            for ep_id in episode_ids:
                if ep_id in self._episodes:
                    self._episodes[ep_id].consolidated = True
                    self._conn.execute(
                        "UPDATE episodes SET consolidated = 1 WHERE id = ?",
                        (ep_id,),
                    )
                    # Remove from RAM (only unconsolidated stay in memory)
                    del self._episodes[ep_id]
            self._conn.commit()

    def add_reflection(self, episode_id: str, reflection: str):
        """Add a reflection annotation to an episode."""
        ep = self._episodes.get(episode_id)
        if ep:
            ep.reflections.append(reflection)
            self._save_episode(ep)

    # ── Persistence ──────────────────────────────────────

    def _save_episode(self, ep: Episode):
        """Save/update an episode in SQLite."""
        with self._lock:
            if self._closed:
                return
            self._conn.execute("""
                INSERT OR REPLACE INTO episodes 
                (id, content, timestamp, salience_json, source, trajectory_id, 
                 trajectory_step, reflections_json, consolidated, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ep.id,
                ep.content,
                ep.timestamp.isoformat(),
                json.dumps(ep.salience.to_dict()),
                ep.source.value,
                ep.trajectory_id,
                ep.trajectory_step,
                json.dumps(ep.reflections),
                1 if ep.consolidated else 0,
                json.dumps(ep.metadata),
            ))
            self._conn.commit()

    def _load_from_db(self):
        """Load unconsolidated episodes from SQLite into memory.
        
        Only active (unconsolidated) episodes are kept in RAM.
        Consolidated episodes remain in SQLite and are queried on demand.
        """
        try:
            cursor = self._conn.execute(
                "SELECT * FROM episodes WHERE consolidated = 0 ORDER BY timestamp"
            )
            for row in cursor:
                salience_data = json.loads(row[3]) if row[3] else {}
                ep = Episode(
                    id=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    salience=SalienceScore(
                        surprise=salience_data.get("surprise", 0),
                        relevance=salience_data.get("relevance", 0),
                        emotional=salience_data.get("emotional", 0),
                        novelty=salience_data.get("novelty", 0),
                        utility=salience_data.get("utility", 0),
                    ),
                    source=MemorySource(row[4]) if row[4] else MemorySource.DIRECT,
                    trajectory_id=row[5],
                    trajectory_step=row[6] or 0,
                    reflections=json.loads(row[7]) if row[7] else [],
                    consolidated=bool(row[8]),
                    metadata=json.loads(row[9]) if row[9] else {},
                )
                self._episodes[ep.id] = ep

            # Get total count from DB for accurate reporting
            self._total_count = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            logger.info(f"Loaded {len(self._episodes)} unconsolidated episodes ({self._total_count} total in DB)")
        except Exception as e:
            logger.error(f"Failed to load episodes: {e}")

    def save(self):
        """Explicitly save all episodes (vector store too)."""
        for ep in self._episodes.values():
            self._save_episode(ep)
        self.vector_store.save()

    def close(self):
        """Close the persistent database connection. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
