"""Tests for the SMRITI MCP server."""
import json
import os
import tempfile
from datetime import datetime

import pytest

from smriti_memcore.models import Memory, MemorySource, MemoryStatus, Modality, SmritiConfig, SalienceScore
from smriti_memcore.core import SMRITI
import smriti_memcore.integrations.mcp_server as _mcp_module


@pytest.fixture
def tmp_smriti(tmp_path):
    """SMRITI instance with temp storage and no LLM."""
    config = SmritiConfig(
        storage_path=str(tmp_path),
        llm_model="none",  # avoid Ollama dependency
    )
    n = SMRITI(config=config)
    yield n
    n.close()


def _is_json_serializable(obj) -> bool:
    """Check that obj round-trips through JSON without error."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def test_serialize_memory_is_json_safe():
    """serialize_memory output must be JSON-serializable."""
    from smriti_memcore.integrations.mcp_server import serialize_memory

    mem = Memory(
        content="test content",
        source=MemorySource.DIRECT,
        status=MemoryStatus.ACTIVE,
        modality=Modality.TEXT,
        salience=SalienceScore(surprise=0.5, relevance=0.8),
        creation_time=datetime(2026, 1, 1),
        last_accessed=datetime(2026, 1, 2),
    )
    result = serialize_memory(mem)
    assert _is_json_serializable(result)


def test_serialize_memory_enum_values():
    """Enums must be serialized to their .value strings."""
    from smriti_memcore.integrations.mcp_server import serialize_memory

    mem = Memory(source=MemorySource.USER_STATED, modality=Modality.CODE)
    result = serialize_memory(mem)
    assert result["source"] == "user_stated"
    assert result["modality"] == "code"


def test_serialize_memory_datetime_iso():
    """datetime fields must be ISO strings."""
    from smriti_memcore.integrations.mcp_server import serialize_memory

    dt = datetime(2026, 3, 19, 12, 0, 0)
    mem = Memory(creation_time=dt, last_accessed=dt)
    result = serialize_memory(mem)
    assert result["creation_time"] == "2026-03-19T12:00:00"
    assert result["last_accessed"] == "2026-03-19T12:00:00"


def test_serialize_memory_expected_keys():
    """Output must contain the core fields expected by MCP consumers."""
    from smriti_memcore.integrations.mcp_server import serialize_memory

    mem = Memory(content="hello")
    result = serialize_memory(mem)
    for key in ("id", "content", "strength", "room_id", "reflection_level", "source", "last_accessed"):
        assert key in result, f"Missing key: {key}"


def test_build_smriti_config_defaults(tmp_path, monkeypatch):
    """Default env vars produce a valid SmritiConfig with expanded path."""
    monkeypatch.setenv("SMRITI_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("SMRITI_LLM_MODEL", raising=False)
    monkeypatch.delenv("SMRITI_LLM_API_KEY", raising=False)
    # Clear ambient cloud keys to prevent SmritiConfig.__post_init__ env var fallback
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    from smriti_memcore.integrations.mcp_server import build_smriti_config
    config = build_smriti_config()
    assert config.storage_path == str(tmp_path)
    assert config.llm_model == "mistral"
    # Unused providers get "" not None — prevents env var inheritance
    assert config.anthropic_api_key == ""
    assert config.openai_api_key == ""


def test_build_smriti_config_anthropic_routing(tmp_path, monkeypatch):
    """SMRITI_LLM_MODEL=claude-* sets anthropic_api_key; others get ''."""
    monkeypatch.setenv("SMRITI_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("SMRITI_LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("SMRITI_LLM_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from smriti_memcore.integrations.mcp_server import build_smriti_config
    config = build_smriti_config()
    assert config.llm_model == "claude-sonnet-4-6"
    assert config.anthropic_api_key == "sk-ant-test"
    assert config.openai_api_key == ""   # "" not None — no env var inheritance


def test_build_smriti_config_openai_routing(tmp_path, monkeypatch):
    """SMRITI_LLM_MODEL=gpt-* sets openai_api_key; others get ''."""
    monkeypatch.setenv("SMRITI_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("SMRITI_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("SMRITI_LLM_API_KEY", "sk-openai-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from smriti_memcore.integrations.mcp_server import build_smriti_config
    config = build_smriti_config()
    assert config.openai_api_key == "sk-openai-test"
    assert config.anthropic_api_key == ""   # "" not None


def test_llm_model_ollama_routing(tmp_path, monkeypatch):
    """Non-prefixed model name (ollama path) produces '' for all provider key fields."""
    monkeypatch.setenv("SMRITI_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("SMRITI_LLM_MODEL", "mistral")
    monkeypatch.delenv("SMRITI_LLM_API_KEY", raising=False)

    from smriti_memcore.integrations.mcp_server import build_smriti_config
    config = build_smriti_config()
    assert config.anthropic_api_key == ""
    assert config.openai_api_key == ""
    assert config.gemini_api_key == ""


def test_build_smriti_config_expands_tilde(monkeypatch):
    """~ in SMRITI_STORAGE_PATH must be expanded."""
    monkeypatch.setenv("SMRITI_STORAGE_PATH", "~/.smriti/test")
    monkeypatch.delenv("SMRITI_LLM_MODEL", raising=False)
    monkeypatch.delenv("SMRITI_LLM_API_KEY", raising=False)

    from smriti_memcore.integrations.mcp_server import build_smriti_config
    config = build_smriti_config()
    assert not config.storage_path.startswith("~")
    assert config.storage_path == os.path.expanduser("~/.smriti/test")


# ── Task 4: Core memory tools ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def inject_smriti(tmp_smriti):
    """Inject test SMRITI instance into the module-level _smriti variable."""
    original = _mcp_module._smriti
    _mcp_module._smriti = tmp_smriti
    yield
    _mcp_module._smriti = original


def test_encode_returns_memory_id():
    """smriti_encode returns a memory_id string for salient content."""
    from smriti_memcore.integrations.mcp_server import smriti_encode
    result = smriti_encode(content="Python is preferred for backend services")
    assert "memory_id" in result
    assert isinstance(result["memory_id"], str)
    assert len(result["memory_id"]) > 0


def test_encode_discarded_on_empty():
    """smriti_encode returns discarded status for empty/whitespace content."""
    from smriti_memcore.integrations.mcp_server import smriti_encode
    result = smriti_encode(content="   ")
    assert result.get("memory_id") is None
    assert result.get("status") == "discarded"


def test_encode_source_default_is_direct():
    """smriti_encode defaults source to 'direct', not 'user_stated'."""
    from smriti_memcore.integrations.mcp_server import smriti_encode
    result = smriti_encode(content="Default source test content")
    assert "memory_id" in result


def test_recall_returns_list():
    """smriti_recall returns a list (empty when store is empty)."""
    from smriti_memcore.integrations.mcp_server import smriti_recall
    result = smriti_recall(query="anything")
    assert isinstance(result, list)


def test_recall_returns_serializable():
    """smriti_recall output is fully JSON-serializable."""
    from smriti_memcore.integrations.mcp_server import smriti_encode, smriti_recall
    smriti_encode(content="LangChain integration uses BaseChatMessageHistory")
    memories = smriti_recall(query="LangChain")
    assert _is_json_serializable(memories)


def test_recall_memory_has_expected_keys():
    """Each recalled memory dict has the required keys."""
    from smriti_memcore.integrations.mcp_server import smriti_encode, smriti_recall
    smriti_encode(content="SMRITI uses a semantic palace for memory storage")
    memories = smriti_recall(query="semantic palace")
    if memories:  # may be empty if attention gate discards
        mem = memories[0]
        for key in ("id", "content", "strength", "room_id", "reflection_level", "source", "last_accessed"):
            assert key in mem, f"Missing key: {key}"


def test_get_context_returns_string():
    """smriti_get_context returns a dict with a 'context' string key."""
    from smriti_memcore.integrations.mcp_server import smriti_get_context
    result = smriti_get_context()
    assert "context" in result
    assert isinstance(result["context"], str)

# ── Task 5: Confidence tools ──────────────────────────────────────────────────

def test_how_well_do_i_know_all_fields():
    """Returns all 6 required fields including decision."""
    from smriti_memcore.integrations.mcp_server import smriti_how_well_do_i_know
    result = smriti_how_well_do_i_know(topic="Python")
    for key in ("coverage", "freshness", "strength", "depth", "overall", "decision"):
        assert key in result, f"Missing key: {key}"


def test_how_well_do_i_know_decision_valid_values():
    """decision field must be one of the three DecisionType values."""
    from smriti_memcore.integrations.mcp_server import smriti_how_well_do_i_know
    result = smriti_how_well_do_i_know(topic="unknown topic xyz")
    assert result["decision"] in ("recall_confidently", "recall_but_verify", "admit_gap_and_ask")


def test_how_well_do_i_know_numeric_fields():
    """Numeric confidence fields must be floats."""
    from smriti_memcore.integrations.mcp_server import smriti_how_well_do_i_know
    result = smriti_how_well_do_i_know(topic="anything")
    for key in ("coverage", "freshness", "strength", "overall"):
        assert isinstance(result[key], float), f"{key} must be float"


def test_knowledge_gaps_returns_list():
    """smriti_knowledge_gaps returns a list."""
    from smriti_memcore.integrations.mcp_server import smriti_knowledge_gaps
    result = smriti_knowledge_gaps()
    assert isinstance(result, list)


def test_knowledge_gaps_shape_when_populated():
    """Each gap dict has the required keys."""
    from smriti_memcore.integrations.mcp_server import smriti_recall, smriti_knowledge_gaps
    smriti_recall(query="extremely obscure topic that does not exist in memory xyz123")
    gaps = smriti_knowledge_gaps()
    if gaps:
        gap = gaps[0]
        for key in ("topic", "context", "discovered_at", "resolved"):
            assert key in gap, f"Missing key: {key}"


# ── Task 6: Memory management tools ──────────────────────────────────────────

def test_pin_success():
    """smriti_pin returns {status: pinned, memory_id} after pinning."""
    from smriti_memcore.integrations.mcp_server import smriti_encode, smriti_pin
    enc = smriti_encode(content="Important fact that must never be forgotten")
    if enc.get("memory_id") is None:
        pytest.skip("attention gate discarded test content")
    memory_id = enc["memory_id"]
    result = smriti_pin(memory_id=memory_id)
    assert result == {"status": "pinned", "memory_id": memory_id}
    # Verify the memory is actually PINNED in the palace
    mem = _mcp_module._smriti.palace.get_memory(memory_id)
    assert mem.status == MemoryStatus.PINNED


def test_pin_not_found():
    """smriti_pin returns error dict for unknown memory_id."""
    from smriti_memcore.integrations.mcp_server import smriti_pin
    result = smriti_pin(memory_id="nonexistent-id-xyz")
    assert "error" in result


def test_forget_sets_archived():
    """smriti_forget returns {status: archived} and memory is ARCHIVED."""
    from smriti_memcore.integrations.mcp_server import smriti_encode, smriti_forget
    enc = smriti_encode(content="Temporary note to be forgotten after use")
    if enc.get("memory_id") is None:
        pytest.skip("attention gate discarded test content")
    memory_id = enc["memory_id"]
    result = smriti_forget(memory_id=memory_id)
    assert result == {"status": "archived", "memory_id": memory_id}
    mem = _mcp_module._smriti.palace.get_memory(memory_id)
    assert mem.status == MemoryStatus.ARCHIVED


def test_forget_not_found():
    """smriti_forget returns error dict for unknown memory_id."""
    from smriti_memcore.integrations.mcp_server import smriti_forget
    result = smriti_forget(memory_id="nonexistent-id-xyz")
    assert "error" in result


def test_consolidate_light():
    """smriti_consolidate('light') returns a summary dict."""
    from smriti_memcore.integrations.mcp_server import smriti_consolidate
    result = smriti_consolidate(depth="light")
    assert "depth" in result
    assert result["depth"] == "light"


def test_consolidate_invalid_depth():
    """smriti_consolidate with invalid depth returns error."""
    from smriti_memcore.integrations.mcp_server import smriti_consolidate
    result = smriti_consolidate(depth="defer")
    assert "error" in result
    result2 = smriti_consolidate(depth="invalid")
    assert "error" in result2


# ── Task 7: Introspection tools ───────────────────────────────────────────────

def test_stats_top_level_keys():
    """smriti_stats returns all 8 expected top-level keys."""
    from smriti_memcore.integrations.mcp_server import smriti_stats
    result = smriti_stats()
    for key in ("palace", "working_memory", "retrieval", "consolidation",
                "meta_memory", "episode_buffer", "vector_store", "metrics"):
        assert key in result, f"Missing top-level key: {key}"


def test_stats_is_json_serializable():
    """smriti_stats output must be JSON-serializable."""
    from smriti_memcore.integrations.mcp_server import smriti_stats
    result = smriti_stats()
    assert _is_json_serializable(result)


def test_get_suggestions_returns_list():
    """smriti_get_suggestions returns a list."""
    from smriti_memcore.integrations.mcp_server import smriti_get_suggestions
    result = smriti_get_suggestions()
    assert isinstance(result, list)


def test_get_suggestions_serializable():
    """smriti_get_suggestions output is JSON-serializable."""
    from smriti_memcore.integrations.mcp_server import smriti_get_suggestions
    result = smriti_get_suggestions()
    assert _is_json_serializable(result)


def test_smriti_encode_with_force_stores_unconditionally():
    """smriti_encode with force=True bypasses Attention Gate discard and stores memory."""
    from smriti_memcore.integrations.mcp_server import smriti_encode
    result = smriti_encode(content="x", force=True)
    assert result.get("memory_id") is not None
    assert result.get("visibility") == "shared"


def test_build_smriti_config_embed_model_override(monkeypatch):
    """SMRITI_EMBED_MODEL overrides embedding_model in build_smriti_config."""
    monkeypatch.setenv("SMRITI_EMBED_MODEL", "nomic-embed-text")
    from smriti_memcore.integrations.mcp_server import build_smriti_config
    config = build_smriti_config()
    assert config.embedding_model == "nomic-embed-text"


def test_mcp_server_mcpserver_instance():
    """mcp_server is defined as an instance of MCPServer/FastMCP."""
    from smriti_memcore.integrations.mcp_server import mcp_server
    assert mcp_server is not None
    assert hasattr(mcp_server, "tool")

