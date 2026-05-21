"""
SMRITI v2 — Shared data models.
All core dataclasses, enums, and configuration used across the system.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────

class MemorySource(Enum):
    """Where a memory came from — affects trust and conflict resolution."""
    DIRECT = "direct"           # Agent's own observation
    USER_STATED = "user_stated" # Explicitly told by user (highest authority)
    INFERRED = "inferred"       # Generated through reflection/consolidation
    EXTERNAL = "external"       # Shared by another agent (lower trust)


class Modality(Enum):
    """Type of content stored in a memory."""
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    STRUCTURED = "structured"   # JSON, tables, etc.


class MemoryStatus(Enum):
    """Lifecycle status of a memory."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"   # Contradicted by newer info
    DECAYING = "decaying"       # Below strength threshold, pending removal
    ARCHIVED = "archived"       # Cold storage (not actively indexed)
    PINNED = "pinned"           # User-marked as permanent


class ConsolidationDepth(Enum):
    """How thorough a consolidation cycle should be."""
    FULL = "full"       # All 8 processes
    LIGHT = "light"     # Chunking + conflict detection only
    DEFER = "defer"     # Too busy, schedule for later


class Visibility(str, Enum):
    """Whether a memory or room can be shared in team consolidation."""
    PRIVATE = "private"   # Never leaves this user's palace
    SHARED = "shared"     # Eligible for team-level consolidation sync


class DecisionType(Enum):
    """Meta-memory decision on how to handle a query."""
    RECALL_CONFIDENTLY = "recall_confidently"
    RECALL_BUT_VERIFY = "recall_but_verify"
    ADMIT_GAP_AND_ASK = "admit_gap_and_ask"


# ──────────────────────────────────────────────────────────
# Core Data Models
# ──────────────────────────────────────────────────────────

@dataclass
class SalienceScore:
    """5-dimensional salience scoring inspired by the amygdala's role."""
    surprise: float = 0.0       # Deviation from predictions
    relevance: float = 0.0      # Relevance to current goals
    emotional: float = 0.0      # Outcome intensity (positive or negative)
    novelty: float = 0.0        # How different from existing knowledge
    utility: float = 0.0        # How practically useful

    # Learned weights (defaults, updated by SalienceWeightLearner)
    _weights: Dict[str, float] = field(default_factory=lambda: {
        "surprise": 0.15,
        "relevance": 0.30,
        "emotional": 0.15,
        "novelty": 0.10,
        "utility": 0.30,
    })

    @property
    def composite(self) -> float:
        """Weighted composite salience score."""
        return (
            self._weights["surprise"] * self.surprise +
            self._weights["relevance"] * self.relevance +
            self._weights["emotional"] * self.emotional +
            self._weights["novelty"] * self.novelty +
            self._weights["utility"] * self.utility
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "surprise": self.surprise,
            "relevance": self.relevance,
            "emotional": self.emotional,
            "novelty": self.novelty,
            "utility": self.utility,
            "composite": self.composite,
        }


@dataclass
class Memory:
    """A single memory unit — the fundamental atom of the SMRITI system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    modality: Modality = Modality.TEXT
    source: MemorySource = MemorySource.DIRECT
    status: MemoryStatus = MemoryStatus.ACTIVE

    # Palace placement
    room_id: Optional[str] = None
    associations: List[str] = field(default_factory=list)  # IDs of linked memories

    # Strength & decay
    strength: float = 1.0
    confidence: float = 1.0
    salience: SalienceScore = field(default_factory=SalienceScore)

    # Temporal tracking
    creation_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Spaced repetition
    next_review: Optional[datetime] = None
    consecutive_successful_reviews: int = 0

    # Conflict tracking
    superseded_by: Optional[str] = None

    # Visibility — controls team consolidation sync eligibility
    visibility: Visibility = field(default_factory=lambda: Visibility.SHARED)

    # Retrieval metadata (transient, not persisted)
    retrieval_score: float = 0.0
    hops: int = 0

    # Adjacency-lifted relevance score from palace.search (transient; spec §6.1)
    relevance_score: float = 0.0
    # Snippet — transient, populated by SnippetExtractor on long memories
    snippet: Optional[str] = None

    # Reflection level (0=raw, 1=observation, 2=insight, 3=principle)
    reflection_level: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reinforce(self, factor: float = 1.1):
        """Strengthen this memory (testing effect)."""
        self.strength = min(self.strength * factor, 10.0)
        self.last_accessed = datetime.now()
        self.access_count += 1

    def decay(self, factor: float = 0.95):
        """Weaken this memory."""
        self.strength *= factor
        if self.strength < 0.01:
            self.status = MemoryStatus.DECAYING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            # "embedding" intentionally omitted — persisted in vectors.npy only (palace.json schema v3+)
            "modality": self.modality.value,
            "source": self.source.value,
            "status": self.status.value,
            "room_id": self.room_id,
            "associations": self.associations,
            "strength": self.strength,
            "confidence": self.confidence,
            "salience": self.salience.to_dict(),
            "creation_time": self.creation_time.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "reflection_level": self.reflection_level,
            "metadata": self.metadata,
            # Spaced-repetition state — must survive save/reload
            "next_review": self.next_review.isoformat() if self.next_review else None,
            "consecutive_successful_reviews": self.consecutive_successful_reviews,
            # Conflict tracking
            "superseded_by": self.superseded_by,
            # Visibility
            "visibility": self.visibility.value,
        }


@dataclass
class MemoryTombstone:
    """Marker left when a memory is gracefully forgotten."""
    original_id: str
    summary: str
    room_id: Optional[str] = None
    removed_at: datetime = field(default_factory=datetime.now)
    reason: str = ""
    embedding: Optional[List[float]] = None  # Can still be found if specifically sought


@dataclass 
class Episode:
    """A timestamped event in the Episode Buffer."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    salience: SalienceScore = field(default_factory=SalienceScore)
    source: MemorySource = MemorySource.DIRECT
    embedding: Optional[List[float]] = None
    
    # Trajectory tracking (MIRA-style)
    trajectory_id: Optional[str] = None
    trajectory_step: int = 0
    
    # Reflection annotations
    reflections: List[str] = field(default_factory=list)
    
    # Whether this episode has been consolidated
    consolidated: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """An executable procedural memory in the Skill Vault."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    code: str = ""
    embedding: Optional[List[float]] = None
    
    # Preconditions and postconditions
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    
    # Usage tracking
    usage_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    
    # Composition
    sub_skills: List[str] = field(default_factory=list)  # IDs of component skills

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceLevel:
    """Meta-memory confidence assessment for a topic."""
    coverage: float = 0.0      # How much of the topic is covered (0-1)
    freshness: float = 0.0     # How recent the knowledge is (0-1)
    strength: float = 0.0      # Average memory strength (0-1)
    depth: int = 0             # Max reflection level (0-3)

    @property
    def overall(self) -> float:
        w = {"coverage": 0.35, "freshness": 0.25, "strength": 0.25, "depth": 0.15}
        depth_norm = min(self.depth / 3.0, 1.0)
        return (
            w["coverage"] * self.coverage +
            w["freshness"] * self.freshness +
            w["strength"] * self.strength +
            w["depth"] * depth_norm
        )

    @property
    def is_unknown(self) -> bool:
        return self.coverage < 0.05


# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

@dataclass
class SmritiConfig:
    """Configuration for a SMRITI instance."""
    # Working Memory
    working_memory_slots: int = 7
    active_chunks: int = 4

    # Retrieval
    retrieval_top_k: int = 10
    reinforcement_factor: float = 1.1
    effort_threshold: float = 1.5
    difficulty_bonus: float = 1.2

    # Scoring weights
    recency_weight: float = 0.2
    relevance_weight: float = 0.4
    strength_weight: float = 0.2
    salience_weight: float = 0.2

    # Consolidation triggers
    episode_buffer_trigger: int = 50        # Consolidate when buffer reaches this
    idle_trigger_seconds: float = 300.0     # 5 minutes idle
    backlog_trigger: int = 1000             # Deep consolidation threshold

    # Forgetting
    strength_hard_threshold: float = 0.05   # Below this → remove
    strength_soft_threshold: float = 0.15   # Below this → archive
    decay_rate: float = 0.99                # Per-day temporal decay

    # Conflict resolution
    conflict_threshold: float = 0.7

    # Palace
    room_merge_threshold: float = 0.85      # Similarity above this → merge rooms
    room_stale_days: int = 90               # Archive rooms not visited in 90 days

    # LLM
    llm_model: str = "mistral"              # Ollama model for reasoning
    code_model: str = "codellama"           # Ollama model for code tasks
    judge_model: str = "gemini-flash"       # Judge model for evaluation
    ollama_base_url: str = "http://localhost:11434"

    # API Keys — resolved from env vars if not explicitly set
    openai_api_key: Optional[str] = None    # Fallback: OPENAI_API_KEY env var
    anthropic_api_key: Optional[str] = None # Fallback: ANTHROPIC_API_KEY env var
    gemini_api_key: Optional[str] = None    # Fallback: GEMINI_API_KEY env var

    # Vector store
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Smarter recall (2026-05-20 design)
    rewrite_mode_default: str = "auto"           # "auto" | "llm" | "none"
    snippet_mode_default: str = "auto"
    snippet_min_chars: int = 300                 # ≤ this → return content as-is
    snippet_max_sentences: int = 2
    llm_rewrite_cache_size: int = 100
    llm_rewrite_prompt_version: str = "v1"       # cache-key component
    adjacency_alpha: float = 0.3                 # lift coefficient
    adjacency_lift_max: float = 1.0              # cap on weighted-average lift
    entry_rooms_top_k: int = 5                   # widened from hardcoded 3

    # Storage
    storage_path: str = "./smriti_data"

    # Safety
    max_content_length: int = 50_000        # Max chars per memory content

    def __post_init__(self):
        import os

        # Resolve API keys from environment if not explicitly set
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.gemini_api_key is None:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY")

        # Validate numeric constraints
        if self.working_memory_slots < 1:
            raise ValueError(f"working_memory_slots must be >= 1, got {self.working_memory_slots}")
        if self.active_chunks < 1 or self.active_chunks > self.working_memory_slots:
            raise ValueError(f"active_chunks must be 1..{self.working_memory_slots}, got {self.active_chunks}")
        if not (0 < self.decay_rate <= 1.0):
            raise ValueError(f"decay_rate must be in (0, 1], got {self.decay_rate}")
        if self.retrieval_top_k < 1:
            raise ValueError(f"retrieval_top_k must be >= 1, got {self.retrieval_top_k}")
        if self.max_content_length < 100:
            raise ValueError(f"max_content_length must be >= 100, got {self.max_content_length}")

        # Validate scoring weights sum (warn, don't error)
        weight_sum = self.recency_weight + self.relevance_weight + self.strength_weight + self.salience_weight
        if abs(weight_sum - 1.0) > 0.01:
            import warnings
            warnings.warn(f"Scoring weights sum to {weight_sum:.2f} instead of 1.0")

        # Smarter recall validation
        _valid_modes = {"auto", "llm", "none"}
        if self.rewrite_mode_default not in _valid_modes:
            raise ValueError(
                f"rewrite_mode_default must be one of {_valid_modes}, got {self.rewrite_mode_default!r}"
            )
        if self.snippet_mode_default not in _valid_modes:
            raise ValueError(
                f"snippet_mode_default must be one of {_valid_modes}, got {self.snippet_mode_default!r}"
            )
        if not (0.0 <= self.adjacency_alpha <= 1.0):
            raise ValueError(
                f"adjacency_alpha must be in [0, 1], got {self.adjacency_alpha}"
            )
        if self.entry_rooms_top_k < 1:
            raise ValueError(
                f"entry_rooms_top_k must be >= 1, got {self.entry_rooms_top_k}"
            )
        if self.snippet_min_chars < 0:
            raise ValueError(
                f"snippet_min_chars must be >= 0, got {self.snippet_min_chars}"
            )
        if self.snippet_max_sentences < 1:
            raise ValueError(
                f"snippet_max_sentences must be >= 1, got {self.snippet_max_sentences}"
            )

