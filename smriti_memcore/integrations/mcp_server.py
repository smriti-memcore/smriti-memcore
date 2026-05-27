"""
SMRITI MCP Server.
Exposes the SMRITI memory system as a Claude Code MCP server via stdio transport.

Usage:
    python -m smriti_memcore.integrations.mcp_server

Environment variables:
    SMRITI_STORAGE_PATH   Where to persist data (default: ~/.smriti/global)
    SMRITI_LLM_MODEL      LLM model name — provider inferred from prefix (default: mistral)
    SMRITI_LLM_API_KEY    API key for cloud providers; empty for Ollama
"""
from __future__ import annotations

import atexit
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

try:
    import mcp
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "To use the SMRITI MCP server, install the mcp extra:\n"
        "pip install smriti-memcore[mcp]"
    )

from smriti_memcore.core import SMRITI
from smriti_memcore.models import (
    ConsolidationDepth,
    Memory,
    MemorySource,
    MemoryStatus,
    Modality,
    SmritiConfig,
    Visibility,
)

logger = logging.getLogger(__name__)

# ── Startup Config ────────────────────────────────────────────────────────────

def build_smriti_config() -> SmritiConfig:
    """
    Build SmritiConfig from environment variables.

    SMRITI_STORAGE_PATH  — storage dir, ~ expanded (default: ~/.smriti/global)
    SMRITI_LLM_MODEL     — model name, provider inferred by prefix (default: mistral)
    SMRITI_LLM_API_KEY   — API key for cloud providers (default: "")
    """
    storage_path = os.path.expanduser(
        os.environ.get("SMRITI_STORAGE_PATH", "~/.smriti/global")
    )
    llm_model = os.environ.get("SMRITI_LLM_MODEL", "mistral")
    api_key = os.environ.get("SMRITI_LLM_API_KEY", "")

    # Infer provider from model name prefix — matches LLMInterface routing in llm_interface.py:61-68
    # IMPORTANT: Pass "" (empty string, not None) for unused provider keys.
    # SmritiConfig.__post_init__ falls back to reading ANTHROPIC_API_KEY/OPENAI_API_KEY/GEMINI_API_KEY
    # env vars only when the field is None. Passing "" prevents that silent inheritance.
    anthropic_key = api_key if llm_model.startswith("claude") else ""
    openai_key = api_key if llm_model.startswith("gpt-") else ""
    gemini_key = api_key if llm_model.startswith("gemini") else ""

    return SmritiConfig(
        storage_path=storage_path,
        llm_model=llm_model,
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
        gemini_api_key=gemini_key,
    )


# Module-level SMRITI instance — initialized at startup, shared across tool calls.
# Tests replace this: `import smriti_memcore.integrations.mcp_server as s; s._smriti = test_instance`
_smriti: Optional[SMRITI] = None

mcp_server = FastMCP(
    "smriti-memory",
    instructions=(
        "SMRITI memory system — AMP Full-conformant (amp_version: 1.0). "
        "Exposes 14 native smriti_* tools and 6 AMP alias tools (amp.encode … amp.stats). "
        "Single-tenant: agent_id is accepted on AMP verbs but ignored; "
        "isolation is at the storage-path level."
        "\n\n"
        "Encoding discipline: when calling smriti_encode or amp.encode, label "
        "unverified claims as hypotheses ('hypothesis:', 'likely', 'unverified') "
        "until they are verified against code, output, or commit — and cite the "
        "evidence (file:line, sha, tool-output id) for any factual claim. A "
        "confidently-worded but wrong memory becomes a liability for future "
        "sessions."
    ),
)


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_memory(memory: Memory, smriti: Optional[SMRITI] = None) -> Dict[str, Any]:
    """Convert a Memory dataclass to a JSON-serializable dict.

    When `memory.snippet` is set, the `content` field returns the snippet (not the
    full content); `expandable=true` signals that the caller can use
    `smriti_get_memory(memory_id)` to fetch the full text (added in Task 13).

    `metadata.rewrite_fallback` and `metadata.snippet_fallback` reflect the most
    recent recall's degradation flags (from `retrieval_engine.retrieval_log[-1]`).
    """
    # Pull last-recall metadata from the smriti instance if provided
    rewrite_fb = False
    snippet_fb = False
    if smriti is not None:
        log = smriti.retrieval_engine.retrieval_log
        if log:
            rewrite_fb = bool(log[-1].get("rewrite_fallback", False))
            snippet_fb = bool(log[-1].get("snippet_fallback", False))

    has_snippet = memory.snippet is not None
    return {
        "id": memory.id,
        "memory_id": memory.id,  # alias for snippet-aware callers (spec §8.2)
        "content": memory.snippet if has_snippet else memory.content,
        "expandable": has_snippet,
        "metadata": {
            "rewrite_fallback": rewrite_fb,
            "snippet_fallback": snippet_fb,
        },
        "strength": memory.strength,
        "confidence": memory.confidence,
        "room_id": memory.room_id,
        "reflection_level": memory.reflection_level,
        "source": memory.source.value,
        "modality": memory.modality.value,
        "status": memory.status.value,
        "visibility": memory.visibility.value,
        "creation_time": memory.creation_time.isoformat(),
        "last_accessed": memory.last_accessed.isoformat(),
        "access_count": memory.access_count,
        "salience": memory.salience.to_dict(),
        "hops": memory.hops,
        "retrieval_score": memory.retrieval_score,
    }
# ── Core Memory Tools ─────────────────────────────────────────────────────────

@mcp_server.tool()
def smriti_encode(
    content: str,
    source: str = "direct",
    modality: str = "text",
    private: bool = False,
) -> Dict[str, Any]:
    """
    Encode information into SMRITI long-term memory.

    Returns the memory_id if stored, or {"memory_id": null, "status": "discarded"}
    if the Attention Gate determined the content has insufficient salience.

    source: "direct" (default), "user_stated" (highest trust, confidence=1.0),
            "inferred", or "external"
    modality: "text" (default), "code", "image", "structured"
    private: if True, memory is marked private and excluded from team consolidation sync

    IMPORTANT — encoding discipline. Label unverified diagnoses as hypotheses,
    not facts. Cite the evidence (file:line, sha, tool output) for any factual
    claim. A fix that made one test pass is a hypothesis with a passing test,
    not a confirmed root cause. Re-label or remove memories that turn out wrong
    rather than leaving confident-but-stale claims in the store — recalls treat
    them as truth.
    """
    try:
        mem_source = MemorySource(source)
    except ValueError:
        return {"error": f"Invalid source '{source}'. Use: direct, user_stated, inferred, external"}
    try:
        mem_modality = Modality(modality)
    except ValueError:
        return {"error": f"Invalid modality '{modality}'. Use: text, code, image, structured"}

    try:
        memory_id = _smriti.encode(content, source=mem_source, modality=mem_modality)
        if memory_id is None:
            return {"memory_id": None, "status": "discarded"}
        mem = _smriti.palace.memories.get(memory_id)
        if private and mem:
            mem.visibility = Visibility.PRIVATE
        _smriti.save()
        return {"memory_id": memory_id, "visibility": mem.visibility.value if mem else "shared"}
    except Exception as e:
        logger.error(f"smriti_encode failed: {e}")
        return {"error": str(e)}


@mcp_server.tool()
def smriti_recall(
    query: str,
    top_k: int = 10,
    rewrite: Optional[Literal["auto", "llm", "none"]] = None,
    snippet: Optional[Literal["auto", "llm", "none"]] = None,
) -> List[Dict[str, Any]]:
    """
    Recall memories relevant to a query.

    rewrite: "auto" (lexical variants, fast, default) | "llm" (LLM paraphrases, 1-3s,
        better for hard queries) | "none" (pass query through unchanged).
        Omit to use server config default.
    snippet: "auto" (top-2 sentence-match, fast, default) | "llm" (LLM-extracted
        sentences, slower, noisy memories) | "none" (return full content).
        Omit to use server config default.

    Returns a list of memory dicts (one per result). Each dict's `content` field is the
    snippet when one was extracted; `expandable=true` and `metadata.snippet_fallback` /
    `metadata.rewrite_fallback` flags surface internal mode degradation.
    """
    try:
        memories = _smriti.recall(query, top_k=top_k, rewrite=rewrite, snippet=snippet)
        return [serialize_memory(m, smriti=_smriti) for m in memories]
    except Exception as e:
        logger.error(f"smriti_recall failed: {e}")
        return [{"error": str(e)}]


@mcp_server.tool()
def smriti_get_memory(memory_id: str) -> Dict[str, Any]:
    """
    Fetch the full content of a memory by id.

    Use this when smriti_recall returned a snippet (expandable=true) and you need the
    complete memory. Returns the same shape as smriti_recall's per-memory dict, with
    `content` always set to the full memory text and `expandable=false`.
    """
    try:
        mem = _smriti.palace.get_memory(memory_id)
        if mem is None:
            return {"error": f"memory {memory_id} not found"}
        # Clear any stale snippet so serialize_memory returns full content
        mem.snippet = None
        return serialize_memory(mem, smriti=_smriti)
    except Exception as e:
        logger.error(f"smriti_get_memory failed: {e}")
        return {"error": str(e)}


@mcp_server.tool()
def smriti_get_context() -> Dict[str, str]:
    """
    Get formatted working memory context for injection into a prompt.

    Returns the current capacity-bounded working memory (7±2 slots) as a
    formatted string ready to prepend to a system prompt or user message.
    """
    try:
        return {"context": _smriti.get_context()}
    except Exception as e:
        logger.error(f"smriti_get_context failed: {e}")
        return {"error": str(e)}


# ── Confidence & Gap Tools ────────────────────────────────────────────────────

@mcp_server.tool()
def smriti_how_well_do_i_know(topic: str) -> Dict[str, Any]:
    """
    Assess confidence about a topic.

    Returns 5 confidence dimensions (coverage, freshness, strength, depth, overall)
    and a decision: "recall_confidently", "recall_but_verify", or "admit_gap_and_ask".

    Uses two internal calls: confidence_map() for dimensions, should_recall_or_ask()
    for the decision — these are separate MetaMemory methods.
    """
    try:
        # Call decision first — it internally calls confidence_map() once
        decision = _smriti.meta_memory.should_recall_or_ask(topic)
        # Then call confidence_map() once more for the dimension breakdown
        conf = _smriti.meta_memory.confidence_map(topic)
        return {
            "coverage": conf.coverage,
            "freshness": conf.freshness,
            "strength": conf.strength,
            "depth": conf.depth,
            "overall": conf.overall,
            "decision": decision.value,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def smriti_knowledge_gaps() -> List[Dict[str, Any]]:
    """
    List topics SMRITI knows it doesn't know.

    Returns gap dicts with keys: topic, context, discovered_at (ISO string), resolved (bool).
    Gaps are registered when recall returns empty or confidence is below threshold.
    """
    try:
        return _smriti.knowledge_gaps()
    except Exception as e:
        return [{"error": str(e)}]


# ── Memory Management Tools ───────────────────────────────────────────────────

@mcp_server.tool()
def smriti_pin(memory_id: str) -> Dict[str, Any]:
    """
    Mark a memory as permanent — it will never be decayed or forgotten.

    Returns {"status": "pinned", "memory_id": ...} on success,
    or {"error": ...} if the memory_id is not found.
    """
    try:
        mem = _smriti.palace.get_memory(memory_id)
        if mem is None:
            return {"error": f"Memory not found: {memory_id}"}
        _smriti.pin(memory_id)
        # Verify the pin actually took effect
        mem = _smriti.palace.get_memory(memory_id)
        if mem is None or mem.status != MemoryStatus.PINNED:
            return {"error": f"Failed to pin memory: {memory_id}"}
        return {"status": "pinned", "memory_id": memory_id}
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def smriti_forget(memory_id: str) -> Dict[str, Any]:
    """
    Gracefully forget a memory by archiving it.

    Sets memory status to ARCHIVED (not deleted — a record remains).
    Returns {"status": "archived", "memory_id": ...} on success,
    or {"error": ...} if the memory_id is not found.
    """
    try:
        mem = _smriti.palace.get_memory(memory_id)
        if mem is None:
            return {"error": f"Memory not found: {memory_id}"}
        _smriti.forget(memory_id)
        return {"status": "archived", "memory_id": memory_id}
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def smriti_create_private_room(topic: str) -> Dict[str, Any]:
    """
    Create a private semantic room in the palace.

    The room itself is marked private. Use this to organise a topic that should
    never be promoted to team-level consolidation sync. To store a private memory,
    encode it with private=True — room visibility is not automatically inherited
    by memories at encode time.

    Returns {"room_id": ..., "topic": ..., "visibility": "private"}.
    """
    try:
        room = _smriti.palace.create_room(topic)
        room.visibility = Visibility.PRIVATE
        _smriti.save()
        return {"room_id": room.id, "topic": room.topic, "visibility": "private"}
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def smriti_consolidate(depth: str = "light") -> Dict[str, Any]:
    """
    Run a consolidation cycle to organize and strengthen memories.

    depth="light": chunking + conflict detection only (fast, safe to call often)
    depth="full": all 8 consolidation processes (thorough, use periodically)

    Note: "defer" is intentionally excluded — it means "let the scheduler decide"
    and is not useful as an explicit call.
    """
    if depth not in ("light", "full"):
        return {"error": "depth must be 'light' or 'full'"}
    try:
        result = _smriti.consolidate(depth=depth)

        # Handle deferred (scheduler decided no consolidation needed)
        if result.get("status") == "deferred":
            return {
                "depth": depth,
                "status": "deferred",
                "processed": 0,
                "summary": result.get("reason", "no consolidation needed"),
                "elapsed_seconds": 0,
            }

        # Normal consolidation result — count successful processes
        processes = result.get("processes", {})
        processed_count = sum(
            1 for p in processes.values()
            if isinstance(p, dict) and "error" not in p
        )
        return {
            "depth": result.get("depth", depth),
            "status": "completed",
            "processed": processed_count,
            "processes": list(processes.keys()),
            "summary": f"{result.get('depth', depth)} consolidation: {processed_count} processes ran",
            "elapsed_seconds": result.get("elapsed_seconds", 0),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Introspection Tools ───────────────────────────────────────────────────────

@mcp_server.tool()
def smriti_stats() -> Dict[str, Any]:
    """
    Get comprehensive SMRITI system statistics.

    Returns a nested dict with 8 top-level keys:
    palace, working_memory, retrieval, consolidation, meta_memory,
    episode_buffer, vector_store, metrics.
    Also includes visibility counts (private_memories, shared_memories) in palace.
    """
    try:
        result = _smriti.stats()
        memories = _smriti.palace.memories.values()
        result["palace"]["private_memories"] = sum(
            1 for m in memories
            if m.visibility == Visibility.PRIVATE and m.status == MemoryStatus.ACTIVE
        )
        result["palace"]["shared_memories"] = sum(
            1 for m in memories
            if m.visibility == Visibility.SHARED and m.status == MemoryStatus.ACTIVE
        )
        return result
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def smriti_get_suggestions() -> List[Dict[str, Any]]:
    """
    Get proactive suggestions from SMRITI's ambient monitor.

    Returns a list of memory dicts — patterns and insights surfaced from
    background consolidation that may be relevant to the current context.
    """
    try:
        suggestions = _smriti.get_suggestions()
        return [serialize_memory(s) for s in suggestions]
    except Exception as e:
        return [{"error": str(e)}]


@mcp_server.tool()
def smriti_open_ui(port: int = 7799) -> Dict[str, Any]:
    """
    Launch the interactive Memory Browser UI in the user's default web browser.

    Use this when the user asks to see, visualize, or browse their memories.
    The UI runs locally and provides a visual graph of the Semantic Palace.
    """
    try:
        from smriti_memcore.ui.server import launch
        # Launch non-blocking (daemon thread runs parallel to MCP server)
        # It automatically opens the browser window
        launch(storage_path=_smriti.config.storage_path, port=port, open_browser=True, blocking=False)
        return {"status": "success", "message": f"Memory Browser UI launching at http://127.0.0.1:{port}"}
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def smriti_sync_obsidian(vault_path: str = "") -> Dict[str, Any]:
    """
    Export the current Semantic Palace to an Obsidian vault.

    Writes one .md per room + _index.md to vault_path.
    Safe to re-run — overwrites Palace/ cleanly.
    Call this after smriti_consolidate to keep the Obsidian vault in sync.

    vault_path: output directory inside the Obsidian vault.
                If omitted, falls back to the SMRITI_OBSIDIAN_PATH env var.
                Returns an error if neither is set.
    Returns: { status, rooms_written, files_written, vault_path }
    """
    try:
        import json
        from pathlib import Path

        from smriti_memcore.palace_to_obsidian import (
            build_room_slug_map,
            render_index,
            render_room_note,
        )

        resolved = vault_path or os.environ.get("SMRITI_OBSIDIAN_PATH", "")
        if not resolved:
            return {
                "error": (
                    "No vault path provided. Either pass vault_path or set the "
                    "SMRITI_OBSIDIAN_PATH environment variable in your MCP server config."
                )
            }

        # Persist in-memory state first — consolidate doesn't call save()
        _smriti.save()

        palace_file = Path(_smriti.config.storage_path) / "palace" / "palace.json"
        vault_dir = Path(resolved).expanduser()

        with open(palace_file) as f:
            palace = json.load(f)

        rooms: dict = palace.get("rooms", {})
        memories: dict = palace.get("memories", {})
        slug_map = build_room_slug_map(rooms, memories)

        vault_dir.mkdir(parents=True, exist_ok=True)

        files_written = 0
        for room_id, room in rooms.items():
            slug = slug_map[room_id]
            room_mems = [m for m in memories.values() if m.get("room_id") == room_id]
            content = render_room_note(room_id, room, room_mems, slug_map)
            (vault_dir / f"{slug}.md").write_text(content, encoding="utf-8")
            files_written += 1

        index_content = render_index(rooms, slug_map, memories)
        (vault_dir / "_index.md").write_text(index_content, encoding="utf-8")
        files_written += 1

        return {
            "status": "success",
            "rooms_written": len(rooms),
            "files_written": files_written,
            "vault_path": str(vault_dir),
        }
    except Exception as e:
        logger.error(f"smriti_sync_obsidian failed: {e}")
        return {"error": str(e)}


# ── AMP Conformance Aliases ───────────────────────────────────────────────────
# Six AMP verbs (amp.encode … amp.stats) exposed alongside the native smriti_*
# tools for full AMP v1.0 compliance. agent_id is accepted but ignored —
# smriti-memcore is single-tenant; isolation is handled at the storage-path level.

@mcp_server.tool(name="amp.encode")
def amp_encode(
    agent_id: str,
    content: str,
    force: bool = False,
    source: str = "direct",
    private: bool = False,
) -> Dict[str, Any]:
    """
    Store a new memory for an agent. (AMP Core verb)

    force=True bypasses the salience gate and always stores.
    private=True marks the memory as private (excluded from team consolidation sync).
    Returns {status: "stored", id: "..."} or {status: "below_threshold"}.

    IMPORTANT — encoding discipline. Label unverified diagnoses as hypotheses,
    not facts. Cite the evidence (file:line, sha, tool output) for any factual
    claim. A fix that made one test pass is a hypothesis with a passing test,
    not a confirmed root cause. Re-label or remove memories that turn out wrong
    rather than leaving confident-but-stale claims in the store — recalls treat
    them as truth.
    """
    if not content or not content.strip():
        return {"error": "content must not be empty", "amp_error_code": "invalid_request"}

    mem_source = MemorySource.USER_STATED if force else MemorySource(source) if source in {m.value for m in MemorySource} else None
    if mem_source is None:
        return {"error": f"Invalid source '{source}'", "amp_error_code": "invalid_request"}

    try:
        memory_id = _smriti.encode(content, source=mem_source, use_llm=not force)
        if memory_id is None:
            return {"status": "below_threshold"}
        mem = _smriti.palace.memories.get(memory_id)
        if private and mem:
            mem.visibility = Visibility.PRIVATE
        _smriti.save()
        return {"status": "stored", "id": memory_id, "visibility": mem.visibility.value if mem else "shared"}
    except Exception as e:
        logger.error(f"amp.encode failed: {e}")
        return {"error": str(e), "amp_error_code": "backend_error"}


@mcp_server.tool(name="amp.recall")
def amp_recall(
    agent_id: str,
    query: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Retrieve memories relevant to a query. (AMP Core verb)

    Returns {results: [{id, content, score, timestamp, status}, ...]}.
    Archived memories are excluded by default.

    Note: passes snippet="none" so memory.content is the full text (AMP contract)
    AND so memory.snippet doesn't get populated as a side effect (which would leak
    into subsequent serialize_memory() calls — see code-review issue Important #1).
    """
    try:
        memories = _smriti.recall(query, top_k=top_k, snippet="none")
        results = [
            {
                "id": m.id,
                "content": m.content,
                "score": m.retrieval_score,
                "timestamp": m.creation_time.isoformat(),
                "status": m.status.value,
            }
            for m in memories
            if m.status.value != "archived"
        ]
        return {"results": results}
    except Exception as e:
        logger.error(f"amp.recall failed: {e}")
        return {"error": str(e), "amp_error_code": "backend_error"}


@mcp_server.tool(name="amp.forget")
def amp_forget(
    agent_id: str,
    id: str,
) -> Dict[str, Any]:
    """
    Permanently forget a memory. (AMP Core verb)

    Returns {status: "forgotten"} or {status: "not_found"}.
    """
    try:
        mem = _smriti.palace.get_memory(id)
        if mem is None:
            return {"status": "not_found"}
        _smriti.forget(id)
        _smriti.save()
        return {"status": "forgotten"}
    except Exception as e:
        logger.error(f"amp.forget failed: {e}")
        return {"error": str(e), "amp_error_code": "backend_error"}


@mcp_server.tool(name="amp.stats")
def amp_stats(agent_id: str) -> Dict[str, Any]:
    """
    Return backend statistics. (AMP Core verb)

    Always includes memory_count (int). May include episode_buffer and retrieval stats.
    """
    try:
        s = _smriti.stats()
        return {
            "memory_count": s["palace"].get("memory_count", 0),
            "episode_buffer": s["episode_buffer"].get("total_episodes", 0),
            "retrieval": s.get("retrieval", {}),
        }
    except Exception as e:
        logger.error(f"amp.stats failed: {e}")
        return {"error": str(e), "amp_error_code": "backend_error"}


@mcp_server.tool(name="amp.pin")
def amp_pin(
    agent_id: str,
    id: str,
) -> Dict[str, Any]:
    """
    Mark a memory as permanent — it will never be decayed or archived. (AMP Full verb)

    Returns {status: "pinned"} or {status: "not_found"}.
    """
    try:
        mem = _smriti.palace.get_memory(id)
        if mem is None:
            return {"status": "not_found"}
        _smriti.pin(id)
        _smriti.save()
        return {"status": "pinned"}
    except Exception as e:
        logger.error(f"amp.pin failed: {e}")
        return {"error": str(e), "amp_error_code": "backend_error"}


@mcp_server.tool(name="amp.consolidate")
def amp_consolidate(
    agent_id: str,
    depth: str = "light",
) -> Dict[str, Any]:
    """
    Trigger backend consolidation. (AMP Full verb)

    depth: "light" (fast) or "full" (thorough).
    Returns {status: "ok", memories_processed: int}.
    """
    if depth not in ("light", "full"):
        return {"error": "depth must be 'light' or 'full'", "amp_error_code": "invalid_request"}
    try:
        result = _smriti.consolidate(depth=depth)

        if result.get("status") == "deferred":
            return {"status": "ok", "memories_processed": 0}

        processes = result.get("processes", {})
        memories_processed = sum(
            1 for p in processes.values()
            if isinstance(p, dict) and "error" not in p
        )
        return {"status": "ok", "memories_processed": memories_processed}
    except Exception as e:
        logger.error(f"amp.consolidate failed: {e}")
        return {"error": str(e), "amp_error_code": "backend_error"}


# ── Startup ───────────────────────────────────────────────────────────────────

def _startup():
    """Initialize the module-level SMRITI instance from env vars."""
    global _smriti
    config = build_smriti_config()
    logger.info(f"Starting SMRITI MCP server (storage: {config.storage_path}, model: {config.llm_model})")
    _smriti = SMRITI(config=config)
    atexit.register(_smriti.save)
    logger.info("SMRITI MCP server ready — 14 smriti_* tools + 6 AMP aliases registered")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _startup()
    mcp_server.run(transport="stdio")
