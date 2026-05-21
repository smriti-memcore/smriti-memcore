# Smarter Recall — Design Spec

**Date:** 2026-05-20
**Status:** Draft
**Scope:** Cluster 1 of the smriti retrospective feedback — recall-side quality improvements.

---

## 1. Goal

Improve smriti's recall pipeline along three dimensions identified in the 2026-05-20 retrospective:

1. **Query rewriting** — today's recall over-indexes on exact lexical match. Synonyms, paraphrases, and stop-word noise cause false negatives (the FAISS-correction memory that didn't surface on "smriti recall query weakness").
2. **Cross-room graph hits** — today neighbor rooms get a 0.85 discount; the feedback wants weakly-hit rooms adjacent to strong-hit rooms *surfaced*, not penalised.
3. **Snippet-level recall** — today recall returns whole-paragraph memories (200-500 words each) even when only one sentence is relevant; the token spend is paid for the whole paragraph.

Session-open brief (item #5 from the retrospective) is **explicitly out of scope** for this spec — see §11.

Target outcome: ~40-50% reduction in recall token-spend and measurable hit-rate@10 improvement on a benchmark harness (see §8).

## 2. Non-Goals

- No new MCP tool. The existing `smriti_recall` tool gains parameters; no `smriti_session_brief`, no `smriti_recall_by_id` rewrite, no new endpoints.
- No persistence-schema change. `Memory.snippet` is transient, like `retrieval_score`.
- No change to the FTS5 / RRF hybrid layer. Query rewriting feeds into both vector and FTS sides but doesn't replace either.
- No centroid-freshness audit. Codex flagged that `_update_room_centroid()` may not run on all mutation paths (deletes, consolidation merges); that audit is a separate ticket — this spec assumes centroids are fresh on read.
- No multi-process / multi-worker considerations. The LLM rewrite cache is per-process; shared cache is a future concern.

## 3. Architecture & Data Flow

Three new pieces slot around the existing hybrid pipeline without replacing it.

```
Caller
  │
  ▼
SMRITI.recall(query, rewrite="auto", snippet="auto", top_k=10)
  │
  ▼
RetrievalEngine.retrieve()
  │
  ├──[NEW] QueryRewriter.expand(query, mode)
  │         → List[str] of query variants (always includes the raw query)
  │
  ├──[MODIFIED] palace.search(variants, top_k_rooms=5)
  │         ─ Embed each variant once; cache the embeddings
  │         ─ Entry rooms widened from top-3 → top-5 by max-over-variants
  │           similarity to room centroid
  │         ─ Per-memory adjacency lift replaces the 0.85 discount
  │
  ├── fts_index.search(joined_variants)     (unchanged interface)
  ├── RRF merge                             (unchanged)
  ├── multi-factor scoring                  (unchanged)
  ├── reinforce + spaced-rep + WM admit     (unchanged)
  │
  └──[NEW] SnippetExtractor.extract(memory, query_variants, query_embedding, mode)
            ─ memory.snippet populated in-place (or left None below threshold)
  │
  ▼
List[Memory]   (memory.content untouched; memory.snippet populated where applicable)
```

**New files:**

| File | Responsibility | LOC budget |
|---|---|---|
| `smriti_memcore/query_rewriter.py` | Lexical + LLM query variant generation, LRU cache | ~100 |
| `smriti_memcore/snippet.py` | Sentence splitting + scoring + snippet assembly | ~80 |

**Modified files:**

| File | Change |
|---|---|
| `smriti_memcore/models.py` | Add `Memory.snippet: Optional[str] = None`; add `SmritiConfig` fields (§7) |
| `smriti_memcore/palace.py` | Replace neighbor discount with per-memory adjacency lift; widen entry rooms to top-5 |
| `smriti_memcore/retrieval.py` | Wire `QueryRewriter` and `SnippetExtractor` into `retrieve()`; pass variants to palace/FTS |
| `smriti_memcore/core.py` | Plumb `rewrite` and `snippet` parameters through `SMRITI.recall()` |
| `smriti_memcore/integrations/mcp_server.py` | Expose `rewrite` and `snippet` enums on the `smriti_recall` tool; serialize `snippet`, `expandable`, `metadata.rewrite_fallback` |

## 4. Query Rewriter

### 4.1 Interface

```python
class QueryRewriter:
    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        cache_size: int = 100,
        prompt_version: str = "v1",
    ): ...

    def expand(self, query: str, mode: str = "auto") -> ExpandResult:
        """Return query variants plus metadata."""

@dataclass
class ExpandResult:
    variants: List[str]               # ≥ 1 entry; index 0 is always the raw query
    used_mode: str                    # what actually ran ("auto" / "llm" / "none")
    fallback: bool = False            # True iff requested mode failed and we fell back
```

### 4.2 `mode="auto"` — lexical variants

Generate up to 3 variants by simple transforms; dedupe identical strings:

1. **Raw** — `query` as-is (always variant[0]).
2. **Stop-stripped** — drop tokens in the shared `_STOP_WORDS` set (imported from `fts_index.py`). Single source of truth.
3. **Content-words** — additionally drop modal/aux verbs and short tokens (len ≤ 2).

Caller embeds each variant once.

### 4.3 `mode="llm"` — paraphrase via LLM

Single `generate_json` call to the configured LLM:

```
Given this user query, generate exactly 3 paraphrased variants that preserve
meaning but use different wording. Return as a JSON list of strings.

Query: {query}
Variants:
```

Returned variants are concatenated to the raw query (so we always have ≥ 4 variants).

**Failure handling:** if the LLM call raises or returns malformed JSON, log a warning, set `fallback=True`, and fall back to `mode="auto"`.

### 4.4 LLM cache

LRU dict on the `QueryRewriter` instance, bounded at `cache_size` (default 100). Used **only** in the LLM path — `mode="auto"` is microseconds and never cached.

**Cache key:** `(query, llm.model_name, prompt_version)` — composite so a model swap or prompt-template bump cleanly invalidates entries.

No TTL; LRU eviction is sufficient.

### 4.5 `mode="none"` — passthrough

Returns `ExpandResult(variants=[query], used_mode="none", fallback=False)`.

## 5. Snippet Extractor

### 5.1 Interface

```python
class SnippetExtractor:
    def __init__(
        self,
        min_chars: int = 300,
        max_sentences: int = 2,
        llm: Optional[LLMInterface] = None,
    ): ...

    def extract(
        self,
        memory: Memory,
        query_variants: List[str],
        query_embedding: np.ndarray,
        mode: str = "auto",
    ) -> None:
        """Mutates memory.snippet in place. Never touches memory.content."""
```

### 5.2 Required pre-extraction step (state-leak guard)

```python
memory.snippet = None   # always clear before deciding whether to populate
```

This must run unconditionally at the top of `extract()`, including the short-circuit and `mode="none"` paths. Codex flagged that `Memory` objects live in `palace.memories` and are returned across recall calls — leftover snippets from a prior call would otherwise leak.

### 5.3 `mode="none"` — skip extraction

Clears `memory.snippet` (§5.2) and returns. Used by library callers who want raw content.

### 5.4 `mode="auto"` — lexical sentence-match

1. **Threshold short-circuit:** if `len(memory.content) ≤ min_chars` (default 300), return after clearing snippet — content is already atomic.
2. **Sentence split:** `re.split(r'(?<=[.!?])\s+', content)`. Limitations on code/markdown/abbreviations are accepted; the cosine-floor fallback (§5.5) handles the worst cases.
3. **Score each sentence:** for each sentence, sum query-token overlap counts across all variants. Stop words filtered using the shared `_STOP_WORDS` set.
4. **Pick top N by score** (default `max_sentences=2`); ties broken by original-order position.
5. **Reorder picks** to original document order; join with `" … "` between non-adjacent picks.
6. Set `memory.snippet = assembled`.

### 5.5 Zero-overlap fallback — cosine floor

If every sentence scores 0 (lexical overlap is empty, e.g., the recall hit was purely on vector similarity), pick the sentence with the highest cosine similarity to `query_embedding`:

```python
sentence_embeddings = [vector_store.embed(s) for s in sentences]
scores = [float(np.dot(query_embedding, se)) for se in sentence_embeddings]
top_idx = int(np.argmax(scores))
memory.snippet = sentences[top_idx]
```

Cheap (rare path; one embedding per sentence; small N). Decisively better than "first sentence" — Codex was right that first-sentence biases toward intros.

### 5.6 `mode="llm"` — LLM-extracted sentences

Single call per memory:

```
Given this query and memory content, extract the 1-2 sentences most relevant
to the query. Return only the extracted text, nothing else.

Query: {query}
Content: {content}
```

If the LLM raises or returns empty, fall back to `mode="auto"`. Same metadata-flag pattern as §4.3 — set a per-memory flag captured by `RetrievalEngine` and surfaced in the MCP layer's `metadata.snippet_fallback`.

## 6. Per-Memory Adjacency Lift

### 6.1 What changes in `palace.search()`

Replace the legacy block:

```python
# Old (palace.py:298-308)
for mem in self.get_room_memories(neighbor.id):
    score = float(np.dot(query_embedding, np.array(mem.embedding)))
    score *= 0.85 * edge.strength      # legacy discount — removed
```

With:

```python
# 1. Score every room by max similarity over all query variants.
room_scores: Dict[str, float] = {}
for rid in self.rooms:
    centroid = self._room_embeddings.get(rid)
    if centroid is None:
        continue
    s = max(float(np.dot(v, centroid)) for v in query_variant_embeddings)
    room_scores[rid] = max(0.0, s)     # clamp ≥ 0 — negative cosines must not subtract

# 2. Score every candidate memory.
alpha = self.config.adjacency_alpha
lift_max = self.config.adjacency_lift_max

for mem in candidate_memories:
    base = max(float(np.dot(v, np.array(mem.embedding))) for v in query_variant_embeddings)
    base = max(0.0, base)              # clamp ≥ 0

    # Weighted-average lift across the room's 1-hop neighbors.
    num = 0.0
    den = 0.0
    for neighbor, edge in self.get_neighbors(mem.room_id):
        w = max(0.0, min(1.0, edge.strength))   # defensive clamp; see §6.3
        num += room_scores.get(neighbor.id, 0.0) * w
        den += w
    lift = (num / den) if den > 0 else 0.0
    lift = min(lift, lift_max)

    mem.retrieval_score = base * (1.0 + alpha * lift)
```

### 6.2 Entry-room widening

Today `palace.search()` picks the top-3 rooms by centroid similarity for candidate generation. Codex flagged that the lift changes *ranking* but not *recall* — a memory in an off-top-3 room with no edge to any top-3 still never enters the pool.

**Change:** widen to `entry_rooms_top_k=5` (new config field, §7). Cheap, directly addresses the recall ceiling.

### 6.3 Edge-strength bound

`edge.strength` is used in the formula. Audit during implementation: confirm it is bounded `[0, 1]` everywhere it is written. If unbounded or if the audit is inconclusive, the defensive `max(0.0, min(1.0, edge.strength))` clamp in §6.1 is the safety net.

### 6.4 Why this is correct

- A memory in room R with a strong direct hit (`base` high) stays high regardless of graph structure — the lift is multiplicative and bounded.
- A memory in room R with a weak direct hit but whose room is graph-adjacent to a strong-hit room gets surfaced — `(1 + α · lift)` can push a 0.3 base score up by ~30% with default α and a saturated lift.
- The weighted-average normalization prevents hub-room saturation: a room with 50 weak edges and a room with 2 strong edges get comparable lifts.
- `α` is a single tunable knob, defaulting to a conservative 0.3.

## 7. Configuration

New fields on `SmritiConfig`:

```python
# Smarter recall
rewrite_mode_default: str = "auto"           # "auto" | "llm" | "none"
snippet_mode_default: str = "auto"
snippet_min_chars: int = 300                 # below this, return content as-is
snippet_max_sentences: int = 2
llm_rewrite_cache_size: int = 100
llm_rewrite_prompt_version: str = "v1"       # cache-key component
adjacency_alpha: float = 0.3                 # lift coefficient
adjacency_lift_max: float = 1.0              # cap on weighted-average lift
entry_rooms_top_k: int = 5                   # widened from hardcoded 3
```

Validation in `SmritiConfig.__post_init__`:

- `rewrite_mode_default` and `snippet_mode_default` ∈ `{"auto", "llm", "none"}`
- `0 ≤ adjacency_alpha ≤ 1`
- `entry_rooms_top_k ≥ 1`
- `snippet_min_chars ≥ 0` and `snippet_max_sentences ≥ 1`

## 8. Public API & MCP Contract

### 8.1 Python library

```python
SMRITI.recall(
    query: str,
    context: str = "",
    top_k: Optional[int] = None,
    rewrite: str = "auto",          # NEW
    snippet: str = "auto",          # NEW
) -> List[Memory]
```

Same two parameters propagate through `RetrievalEngine.retrieve()`. Memory objects gain a transient `snippet: Optional[str]` field.

**Backward compatibility:** callers who don't pass `rewrite` / `snippet` get the new defaults (`"auto"`). For strict regression-guard tests, pass `rewrite="none", snippet="none"` to disable both new features.

### 8.2 MCP `smriti_recall` tool

JSON schema gains:

```json
{
  "rewrite": {
    "type": "string",
    "enum": ["auto", "llm", "none"],
    "default": "auto",
    "description": "auto = lexical variants (fast, no LLM); llm = LLM paraphrases (1-3s, better for hard queries); none = pass query through unchanged"
  },
  "snippet": {
    "type": "string",
    "enum": ["auto", "llm", "none"],
    "default": "auto",
    "description": "auto = top-2 sentence-match (fast); llm = LLM-extracted sentences (slower, noisy memories); none = return full content"
  }
}
```

JSON response per memory gains:

```json
{
  "memory_id": "...",
  "content": "<snippet if extraction populated it, else full content>",
  "expandable": true,                       // present always; true iff a snippet was used
  "metadata": {
    "rewrite_fallback": false,              // true iff LLM rewrite failed → auto
    "snippet_fallback": false               // true iff LLM snippet failed → auto
  }
}
```

Always-present fields keep the agent's schema understanding stable.

## 9. Testing Strategy (TDD)

All new behavior follows red-green-refactor. Each task in the implementation plan begins with a failing test that pins the contract.

| Test file | Coverage |
|---|---|
| `tests/test_query_rewriter.py` (new) | `mode="auto"` yields ≥ 1 and ≤ 3 deduped variants; `mode="none"` returns `[query]`; `mode="llm"` calls LLM exactly once per unique query; LLM failure sets `fallback=True` and falls back to auto; LRU cache hit on repeat; cache key changes when `model_name` or `prompt_version` changes; `ExpandResult.variants[0] == query` invariant |
| `tests/test_snippet.py` (new) | Below `min_chars` → snippet stays None; above threshold → top-N sentences in document order; zero-overlap path uses cosine floor (mocked embeddings); `mode="none"` clears snippet; `extract()` always clears prior snippet on entry; LLM failure falls back |
| `tests/test_palace.py` (extend) | Adjacency lift surfaces a graph-adjacent weak-hit memory above a non-adjacent weak-hit (mocked embeddings); negative cosine clamped to 0; lift saturates at cap; weighted-average normalization keeps a 50-weak-edge hub bounded; `entry_rooms_top_k=5` returns memories from off-top-3 rooms |
| `tests/test_retrieval.py` (extend) | `recall(rewrite="none", snippet="none")` matches v1.1.x behavior (regression guard); end-to-end `rewrite="auto" snippet="auto"` happy path; `memory.snippet` not persisted across save/reload (transient field check); `rewrite_fallback` flag set when LLM raises |
| `tests/test_mcp_server.py` (extend) | Tool schema exposes both new enums with correct values; JSON response includes `expandable` and `metadata.*` keys; serializer prefers `snippet` over `content` when present |

## 10. Benchmark Harness

`scripts/bench_recall.py` — one-shot, not in CI. Seeds a palace with N=500 synthetic memories across 20 rooms, runs 50 paraphrased queries, reports:

- Hit-rate@10 (target memory in top-10 result set)
- Avg tokens returned per query (proxy for snippet effectiveness)
- p95 latency for `rewrite="auto"` vs `rewrite="llm"`

Run once before the change (baseline on `main`) and once after (on the feature branch). Numbers go in the PR description, not in tests.

## 11. Known Limitations

These were raised by Codex review and accepted as out-of-scope for this spec:

- **Centroid staleness:** `_update_room_centroid()` is not called on consolidation merges or deletes (audit suggests; not confirmed). Tracked as a separate cleanup ticket.
- **Per-instance LLM cache:** behavior is process-local. Multi-worker deployments would see uneven cache hit rates. Acceptable for single-process smriti; revisit if/when a worker pool is introduced.
- **Sentence-split limitations:** code blocks, markdown, abbreviations, and decimals will sometimes be over- or under-split by the stdlib regex. The cosine-floor fallback (§5.5) handles the worst case where over-splitting fragments meaning away from query tokens. NLTK is not added for v1.
- **Max-over-variants gate-widening:** one variant can select the room while another scores the memory. Intentional — this is the semantic of multi-query rewriting; it does mean false-positive rates can rise slightly. Mitigated by the existing multi-factor scoring (recency, strength, salience) acting on the candidate pool.
- **Session-open brief (retrospective item #5):** deferred. If `recall()` quality improves as projected, the agent calls `recall()` on the user's first message anyway; a separate brief tool is redundant for Claude Code's lifecycle. Revisit only if recall improvements don't close the bootstrap gap.

## 12. Open Questions Resolved by This Spec

| Question | Resolution |
|---|---|
| Library vs MCP wrapper split? | Core library (smriti is the service; MCP is a thin adapter). |
| Rewriting mechanism? | Hybrid: lexical multi-query (default), LLM opt-in. |
| Snippet shape? | Hybrid threshold; lexical sentence-match with cosine fallback. |
| Brief mechanism? | Out of scope. |
| Adjacency? | Per-memory weighted-average lift; entry rooms widened to top-5. |
| Edge.strength bounds? | Audit + defensive clamp `[0, 1]`. |
| LLM cache key? | `(query, model_name, prompt_version)`. |
| Snippet state leak? | `extract()` clears `memory.snippet` unconditionally on entry. |
| First-sentence vs cosine-floor fallback? | Cosine floor — Codex was right. |

## 13. Acceptance Criteria

This work is done when:

1. All new and modified tests pass under TDD discipline (red → green → commit per task).
2. The existing 233-test suite continues to pass.
3. `scripts/bench_recall.py` shows ≥ 30% reduction in avg tokens/query AND no regression (≤ -2 percentage points) in hit-rate@10.
4. `recall(rewrite="none", snippet="none")` is byte-for-byte equivalent in result ordering to pre-change behavior (regression guard).
5. The MCP tool schema correctly exposes both new enums; an agent can choose between modes.
