# Architectural Integration: SMRITI Memcore + Headroom

This document analyzes the design of [Headroom](https://github.com/chopratejas/headroom) and outlines a concrete proposal for integrating its core concepts into SMRITI to make the cognitive memory architecture faster, cheaper, and more robust.

---

## 1. Executive Summary

| Feature / Goal | SMRITI Memcore (Now) | Integrated with Headroom | Benefit |
| :--- | :--- | :--- | :--- |
| **Injected Context Size** | Injecting full raw memory texts into Working Memory. | **Reversible Content Compression & Retrieval (CCR)**. | 60–90% token reduction per prompt; allows larger memories without hitting context limits. |
| **Prompt Cache Hits** | Dynamic, turn-by-turn memory changes invalidate KV cache. | **CacheAligner (Prefix Stabilization & Padding)**. | Maximizes Anthropic/OpenAI prompt caching; dramatically reduces recurrent costs. |
| **Consolidation Cost** | Processing raw logs and code blocks in background via LLM. | **AST / JSON Local Pre-compression** before LLM consolidation. | ~70% lower token cost for background System 2 consolidation cycles. |
| **Multi-Agent Memory** | Isolated local storage path per workspace. | **Cross-agent memory with provenance tracking**. | Safe memory sharing across different terminal sessions and agents. |

---

## 2. Deep Dive: Key Integration Points

### A. Reversible Content Compression & Retrieval (CCR)
**The Problem:** 
When SMRITI retrieves memories (e.g. past code snippets, JSON logs, terminal outputs) and loads them into Working Memory, injecting these verbose blocks into the active LLM context consumes a large number of tokens and risks overwhelming the prompt.

**The Headroom Approach:**
Headroom's **CCR** compresses data using local heuristics:
* **SmartCrusher:** Shrinks JSON by removing boilerplate, nested arrays, and redundant schemas while preserving key-value pairs.
* **CodeCompressor:** An AST-aware code compressor that strips comments, whitespace, imports, and non-essential helper methods, leaving only structural interfaces and signatures.
* **Kompress-base:** A tiny local ML model trained on agentic traces to prune prose to its semantic core.
The original raw content is cached locally, and the LLM is given the compressed version alongside a retrieval tool to fetch the original if details are needed.

**How SMRITI Integrates CCR:**
```text
                         [Memory Storage Phase]
  [raw code/json] ──▶ ContentRouter ──▶ CodeCompressor / SmartCrusher
                                                  │
                                 ┌────────────────┴───────────────┐
                                 ▼ (compressed summary)           ▼ (original text)
                          [Semantic Palace]             [Episode Buffer / SQLite]
                             (Searchable)                 (Keyed by Memory ID)
```

1. **Storage:** When `smriti_encode` is called with long code snippets, log files, or raw JSON, SMRITI passes the payload through a local `ContentRouter` to select the right compression algorithm.
2. **Splitting:** The *compressed* summary is stored in `Memory.content_compressed`. The *original* verbose content stays in `Memory.content`. Embeddings are **always computed from the original content** to preserve semantic search quality.
3. **Retrieval:** When recalled, only `content_compressed` is injected into the LLM context (via `format_for_llm()`). A compression marker is appended: `⟨compressed:memory_id⟩ — call smriti_retrieve_original(id) for full text`.
4. **Tool Hook:** SMRITI exposes a tool `smriti_retrieve_original(memory_id)`. If the agent needs to see the uncompressed code or JSON details, it invokes the tool.

**ContentRouter Logic:**
The router uses SMRITI's existing `Modality` enum as the primary routing signal — no duplicate content-type detection:

| Modality | Compressor | Fallback |
|:---|:---|:---|
| `CODE` | `CodeCompressor` (AST-based) | Raw passthrough if AST parsing fails |
| `STRUCTURED` | `SmartCrusher` (JSON schema-aware) | Raw passthrough if JSON is malformed |
| `TEXT` | Passthrough (no compression in Phase 1) | — |
| `IMAGE` | Passthrough | — |

**Minimum compression threshold:** If the compression ratio is < 20% (i.e., the compressed text is > 80% of the original), skip compression and set `content_compressed = None`. The overhead of storing two copies isn't worth it.

**Schema Change — `Memory` model (palace.json schema v4):**
```python
# New field on Memory dataclass
content_compressed: Optional[str] = None  # Compressed version for LLM injection
```
- `content`: Always the original, uncompressed text. Used for embeddings, FTS, and `smriti_retrieve_original`.
- `content_compressed`: The compressed version injected into Working Memory context. `None` if compression was skipped or not applicable.
- **Migration**: Existing memories (schema v3) have `content_compressed = None`. On load, treat `None` as "use `content` as-is" — no backfill needed. New memories get compressed at encode time.

---

### B. Cache Aligner for Injected Memory Blocks
**The Problem:**
Prompt caching (like Anthropic's Prompt Caching or OpenAI's KV Cache) requires the starting prefix of the prompt to remain stable. Since SMRITI retrieves and injects fresh memories dynamically on every turn, the injected memory blocks constantly change, breaking cache hits and increasing prompt token costs.

**The Headroom Approach:**
`CacheAligner` pads, groups, and sorts injected contexts into stable blocks to ensure that the prefix segments align with chunk boundaries, maximizing cache hits.

**How SMRITI Integrates CacheAligner:**

> **Scope clarification:** SMRITI is an MCP tool server — it does **not** control the host's prompt layout. The following optimizations apply only to what SMRITI controls: the content and ordering of the string returned by `smriti_get_context()` / `format_for_llm()`.

* **Stable Block Ordering:** Instead of ordering retrieved memories by retrieval score (which fluctuates turn-by-turn), sort them deterministically (e.g., alphabetically by `memory.id`) before injecting them. This ensures the prefix is stable across turns if the same memories are recalled.
* **Context Padding & Boundaries:** Pad the injected memory block to the nearest 1024-token boundary (estimated at ~4 chars/token). This ensures that changes inside the memory context do not shift the positions of subsequent prompt structures, preventing downstream cache invalidation.
* **No "anchor block injection"** — that requires host cooperation and is outside SMRITI's control. We document the recommended prompt layout for hosts that want to maximize cache hits.

---

### C. System 2 Consolidation Optimizer
**The Problem:**
Background consolidation (System 2) reads raw episodic sequences from the Episode Buffer and runs an LLM to extract facts, conflict-resolve, and create semantic rooms. Running background consolidation on massive raw terminal outputs or log files is expensive and can exceed local model (Ollama) context windows.

**How SMRITI Integrates Local Pre-compression:**
* Before sending raw logs or code outputs to the consolidation LLM, SMRITI can run them through local AST parsers and JSON crushers.
* For example, a 50KB JSON API response stored in the episode buffer is crushed to 3KB of structured metadata before the consolidation LLM reviews it.
* This makes background consolidation much faster and highly compatible with fast, local 3B/7B models.

**Coverage:** Pre-compression applies to all three LLM-consuming consolidation processes:
1. `_process_chunking()` — compresses `ep.content` before `llm.chunk_memories()`
2. `_process_reflection()` — compresses before `llm.generate_reflection()`
3. `_process_conflict_resolution()` — compresses before `llm.detect_contradiction()`

**Important:** Pre-compression is a **transient transformation** — a temporary compressed copy is created for the LLM call, but the original episode content is never mutated. The Episode Buffer remains the source of truth.

---

## 3. Dependency Strategy

**Decision: Reimplement, don't depend.**

Headroom is a research repository without stable releases or a PyPI package. The core compressor algorithms (`SmartCrusher` for JSON, `CodeCompressor` for AST) are conceptually simple and well-documented. We will:

1. **Reimplement** the algorithms as clean, dependency-free modules inside `smriti_memcore/compressors/`.
2. **Cite Headroom** as the inspiration in module docstrings and README.
3. **Keep SMRITI's install lightweight** — no new runtime dependencies beyond the Python stdlib `ast` and `json` modules.

---

## 4. Proposed Implementation Plan

### Phase 1: Local Compressor Integration
1. Add a `smriti_memcore/compressors/` package containing:
   * `__init__.py`: Exports `ContentRouter` and individual compressors.
   * `router.py`: `ContentRouter` class — routes by `Modality` enum.
   * `json_crusher.py`: JSON schema-aware pruner (removes redundant arrays, deep nesting, boilerplate keys).
   * `code_crusher.py`: AST-based pruner for Python (strips comments, docstrings, whitespace, import blocks). JavaScript support deferred.
2. Integrate `ContentRouter` into `SMRITI.encode()` to auto-compress verbose payloads.
3. Add `content_compressed: Optional[str]` to the `Memory` dataclass. Bump palace.json to schema v4.
4. Add metrics: `compression_ratio` histogram, `compression_errors` counter.
5. Add unit tests for each compressor: round-trip correctness, edge cases (empty input, malformed JSON, syntax errors in code, non-Python code fallback).

### Phase 2: CCR Tool and Retrieval Integration
1. Update `WorkingMemory.format_for_llm()` to prefer `content_compressed` over `content` when available.
2. Append compression marker to injected text: `⟨compressed:mem_id⟩`.
3. Add MCP tool `smriti_retrieve_original(memory_id: str)` — returns uncompressed `content` from the palace.
4. Add AMP alias `amp.retrieve_original(agent_id, id)`.
5. Add metric: `original_retrieval_count` counter.
6. Integration test: encode → recall → verify compressed content → retrieve_original → verify original.

### Phase 3: Prefix Alignment
1. Update `WorkingMemory.format_for_llm()` to sort memories deterministically by `memory.id` instead of by priority.
2. Implement block-padding (round output to nearest 4096-char boundary with whitespace).
3. Document recommended host prompt layout for maximum cache hit rate.

### Phase 4: Consolidation Pre-compression
1. Add a `compress_for_llm(content, modality)` helper in the compressors package.
2. Integrate into `_process_chunking()`, `_process_reflection()`, and `_process_conflict_resolution()`.
3. Benchmark: measure actual token savings on real SMRITI episode data vs. baseline.

---

## 5. Verification Plan

### Automated Tests
- Unit tests for `json_crusher.py`: valid JSON, nested objects, arrays, empty input, malformed JSON, large payloads.
- Unit tests for `code_crusher.py`: Python functions, classes, decorators, type hints, syntax errors, non-Python code.
- Unit tests for `ContentRouter`: correct routing by modality, compression ratio threshold, fallback behavior.
- Integration test for CCR round-trip: encode(code) → recall → verify compressed → retrieve_original → verify matches input.
- Metric assertions: compression_ratio within expected bounds.

### Manual Verification
- Run SMRITI MCP server with compression enabled, encode real code/JSON from a project, verify compressed context in `smriti_get_context()`.
- Measure actual token reduction on a representative episode buffer.

### Benchmarks
- Compare consolidation LLM token usage with and without pre-compression on 50+ episodes.
- Measure encode latency overhead from compression (target: < 50ms per encode for typical payloads).
