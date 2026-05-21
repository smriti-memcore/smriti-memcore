"""Tests for smriti.models — data models, enums, config validation."""

import os
import pytest
from datetime import datetime, timedelta

from smriti_memcore.models import (
    SmritiConfig, Memory, Episode, SalienceScore, MemorySource,
    MemoryStatus, Modality, ConfidenceLevel, DecisionType,
    ConsolidationDepth, MemoryTombstone, Skill,
)


class TestSalienceScore:
    """Tests for SalienceScore dataclass."""

    def test_composite_weighted_average(self):
        s = SalienceScore(surprise=1.0, relevance=1.0, emotional=1.0, novelty=1.0, utility=1.0)
        assert s.composite == pytest.approx(1.0)

    def test_composite_zeros(self):
        s = SalienceScore()
        assert s.composite == pytest.approx(0.0)

    def test_composite_mixed(self):
        s = SalienceScore(surprise=0.5, relevance=0.8, emotional=0.2, novelty=0.3, utility=0.9)
        # Weighted: 0.15*0.5 + 0.30*0.8 + 0.15*0.2 + 0.10*0.3 + 0.30*0.9
        expected = 0.075 + 0.24 + 0.03 + 0.03 + 0.27
        assert s.composite == pytest.approx(expected, rel=1e-3)

    def test_to_dict(self):
        s = SalienceScore(surprise=0.1, relevance=0.2)
        d = s.to_dict()
        assert d["surprise"] == 0.1
        assert d["relevance"] == 0.2
        assert "composite" in d


class TestMemory:
    """Tests for Memory dataclass."""

    def test_creation_defaults(self):
        m = Memory(content="test")
        assert m.id is not None
        assert m.content == "test"
        assert m.strength == 1.0
        assert m.confidence == 1.0
        assert m.status == MemoryStatus.ACTIVE
        assert m.embedding is None

    def test_to_dict_roundtrip(self):
        m = Memory(
            content="roundtrip test",
            embedding=[0.1, 0.2, 0.3],
            associations=["mem_a", "mem_b"],
            metadata={"key": "value"},
            source=MemorySource.USER_STATED,
        )
        d = m.to_dict()
        assert d["content"] == "roundtrip test"
        # embedding is no longer serialized to palace.json (schema v3+) — stored in VectorStore
        assert "embedding" not in d
        assert d["associations"] == ["mem_a", "mem_b"]
        assert d["metadata"] == {"key": "value"}
        assert d["source"] == "user_stated"

    def test_decay(self):
        m = Memory(content="test", strength=2.0)
        m.decay(0.9)
        assert m.strength == pytest.approx(1.8)

    def test_reinforce(self):
        m = Memory(content="test", strength=1.0)
        m.reinforce(1.5)
        # reinforce multiplies: 1.0 * 1.5 = 1.5
        assert m.strength == pytest.approx(1.5)
        assert m.access_count == 1

    def test_reinforce_capped(self):
        m = Memory(content="test", strength=9.5)
        m.reinforce(1.1)
        # 9.5 * 1.1 = 10.45 → capped at 10.0
        assert m.strength == 10.0

    def test_to_dict_includes_all_fields(self):
        m = Memory(content="test")
        d = m.to_dict()
        required_fields = [
            "id", "content", "modality", "source", "status",
            "room_id", "associations", "strength", "confidence", "salience",
            "creation_time", "last_accessed", "access_count", "reflection_level",
            "metadata",
        ]
        for field in required_fields:
            assert field in d, f"Missing field: {field}"
        # embedding is intentionally excluded since schema v3 (stored in VectorStore)
        assert "embedding" not in d


class TestSmritiConfig:
    """Tests for SmritiConfig validation."""

    def test_default_config(self):
        c = SmritiConfig()
        assert c.decay_rate > 0
        assert c.working_memory_slots > 0

    def test_decay_rate_zero_rejected(self):
        with pytest.raises(ValueError, match="decay_rate"):
            SmritiConfig(decay_rate=0)

    def test_decay_rate_negative_rejected(self):
        with pytest.raises(ValueError, match="decay_rate"):
            SmritiConfig(decay_rate=-0.5)

    def test_decay_rate_above_one_rejected(self):
        with pytest.raises(ValueError, match="decay_rate"):
            SmritiConfig(decay_rate=1.5)

    def test_working_memory_slots_zero_rejected(self):
        with pytest.raises(ValueError, match="working_memory_slots"):
            SmritiConfig(working_memory_slots=0)

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        c = SmritiConfig()
        assert c.openai_api_key == "test-key-123"

    def test_explicit_api_key_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        c = SmritiConfig(openai_api_key="explicit-key")
        assert c.openai_api_key == "explicit-key"


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_creation_defaults(self):
        e = Episode(content="test episode")
        assert e.id is not None
        assert e.consolidated is False

    def test_salience_attached(self):
        s = SalienceScore(relevance=0.9)
        e = Episode(content="test", salience=s)
        assert e.salience.relevance == 0.9


class TestConfidenceLevel:
    """Tests for ConfidenceLevel."""

    def test_unknown_default(self):
        c = ConfidenceLevel()
        assert c.is_unknown is True
        assert c.overall == pytest.approx(0.0)

    def test_known_state(self):
        c = ConfidenceLevel(coverage=0.8, freshness=0.9, strength=0.7, depth=2)
        assert c.is_unknown is False
        assert c.overall > 0
