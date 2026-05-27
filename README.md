# SMRITI Memory

**A neuro-inspired long-term memory architecture for AI agents.**

SMRITI combines a capacity-bounded Working Memory, a graph-based Semantic Palace, and asynchronous background consolidation to give LLM agents persistent, scalable memory — without blocking real-time interactions.

> 📄 **Paper:** *SMRITI: A Scalable, Neuro-Inspired Architecture for Long-Term Event Memory in LLM Agents* — Shivam Tyagi, 2025 — [DOI: 10.13140/RG.2.2.25477.82407](https://doi.org/10.13140/RG.2.2.25477.82407)

[![PyPI](https://img.shields.io/pypi/v/smriti-memcore.svg)](https://pypi.org/project/smriti-memcore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

```text
                           ┌─────────────────────────────────┐
                           │    Asynchronous Consolidation   │
                           │      (8 Background Processes)   │
                           │  • Chunking      • Cross-Ref.   │
                           │  • Conflict Res. • Skill Ext.   │
                           │  • Forgetting    • Spaced Rep.  │
                           │  • Reflection    • Defragment.  │
                           └────────────────┬────────────────┘
                                            │ background
  ┌──────────┐   ┌──────────┐   ┌───────────▼─────────┐   ┌──────────┐
  │  Input   │──▶│ Attention │──▶│   Episode Buffer    │──▶│ Semantic │
  │  Text    │   │   Gate    │   │  (append-only log)  │   │  Palace  │
  └──────────┘   │ (salience │   └─────────────────────┘   │  Graph   │
                 │  filter)  │                              │ G=(V,E)  │
                 └──────────┘                              └────┬─────┘
                                                                │
  ┌──────────┐   ┌──────────┐   ┌───────────────────┐           │
  │  Query   │──▶│ Retrieval│──▶│  Working Memory   │◀──────────┘
  │          │   │  Engine  │   │   (7 ± 2 slots)   │
  └──────────┘   │ Q(v) =   │   └───────────────────┘
                 │ β₁cos +  │
                 │ β₂decay+ │   ┌───────────────────┐
                 │ β₃freq + │──▶│    Meta-Memory    │
                 │ β₄sal    │   │ (confidence map)  │
                 └──────────┘   └───────────────────┘
```

**Core idea:** Inspired by human Dual-Process Theory (Daniel Kahneman's *Thinking, Fast and Slow*), SMRITI decouples memory operations into two pathways:
- **System 1 (Fast & Heuristic):** Real-time ingestion. Routes interactions to the short-term Episode Buffer in milliseconds without blocking the agent.
- **System 2 (Slow & Analytical):** Background consolidation. Uses LLM reasoning to chunk, organize, and abstract semantic knowledge asynchronously while the agent is idle.
---

## Quick Start — Claude Code (MCP)

The fastest way to use SMRITI is as a persistent memory layer for [Claude Code](https://claude.ai/code). One command, and your AI remembers you across every session.

**Run the install script:**

```bash
bash <(curl -s https://raw.githubusercontent.com/smriti-memcore/smriti-memcore/main/install_smriti_mcp.sh)
```

The script will:
- Create a dedicated venv at `~/.smriti/venv`
- Install `smriti-memcore[mcp]` into it
- Prompt for your LLM choice and API key
- Register the MCP server in `~/.claude.json`
- Optionally configure automatic memory hooks

**Then restart Claude Code.** Verify with `/mcp` — `smriti` should appear as connected.

**Available tools (20: 14 native + 6 AMP v1.0 aliases):**

| Tool | Description |
|---|---|
| `smriti_encode` | Store information in long-term memory (`private=True` keeps it out of team sync) |
| `smriti_recall` | Retrieve memories by natural-language query; optional `rewrite` and `snippet` enums (`"auto"`/`"llm"`/`"none"`) for query rewriting and content snippet extraction |
| `smriti_get_memory` | Fetch the full content of a single memory by ID — useful after a snippet recall returns `expandable=true` |
| `smriti_get_context` | Inject working memory into the current prompt |
| `smriti_how_well_do_i_know` | Confidence check on a topic |
| `smriti_knowledge_gaps` | List topics SMRITI knows it doesn't know |
| `smriti_pin` | Mark a memory as permanent (never decayed) |
| `smriti_forget` | Archive a memory |
| `smriti_consolidate` | Run a consolidation cycle |
| `smriti_stats` | System-wide statistics (includes private/shared memory counts) |
| `smriti_get_suggestions` | Proactive insights from background consolidation |
| `smriti_create_private_room` | Create a private semantic room — memories in it are excluded from team consolidation sync |
| `smriti_open_ui` | Launch the visual Memory Browser in the default web browser |
| `smriti_sync_obsidian` | Export the Semantic Palace to an Obsidian vault |

**AMP v1.0 aliases** (interoperable with any AMP-conformant agent framework):

| AMP Tool | Maps to |
|---|---|
| `amp.encode` | `smriti_encode` — with `agent_id` + `force` + `private` params, AMP response schema |
| `amp.recall` | `smriti_recall` — returns `{results: [{id, content, score, timestamp, status}]}` |
| `amp.forget` | `smriti_forget` — returns `{status: "forgotten" \| "not_found"}` |
| `amp.stats` | `smriti_stats` — returns `{memory_count, ...}` |
| `amp.pin` | `smriti_pin` — returns `{status: "pinned" \| "not_found"}` |
| `amp.consolidate` | `smriti_consolidate` — returns `{status: "ok", memories_processed: int}` |

> smriti-memcore is single-tenant — `agent_id` is accepted on all AMP verbs but ignored. Isolation is at the storage-path level.

**LLM options** — set during install or via environment variables:

| Model | Provider | Requires |
|---|---|---|
| `mistral` (default) | Local Ollama | `ollama pull mistral` |
| `claude-*` | Anthropic | `SMRITI_LLM_API_KEY` |
| `gpt-*` | OpenAI | `SMRITI_LLM_API_KEY` |
| `gemini*` | Google | `SMRITI_LLM_API_KEY` |

---

## Installation (Python Library)

```bash
pip install smriti-memcore
```

With optional **FAISS** accelerated vector search:

```bash
pip install smriti-memcore[faiss]
```

Or install from source:

```bash
git clone https://github.com/smriti-memcore/smriti-memcore.git
cd smriti-memcore
pip install -e .
```

### Prerequisites

SMRITI uses an LLM for reasoning tasks (consolidation, reflection, skill extraction). By default it connects to a local [Ollama](https://ollama.ai) instance:

```bash
ollama pull mistral
```

Alternatively, you can use **OpenAI**, **Anthropic**, or **Google Gemini** — see [Using Cloud LLM Providers](#using-cloud-llm-providers) below.

---

## Using Cloud LLM Providers

SMRITI is **provider-agnostic**. Just change the `llm_model` and pass your API key:

```python
from smriti import SMRITI, SmritiConfig

# ── OpenAI ──────────────────────────────────────────────
config = SmritiConfig(
    llm_model="gpt-4o",
    openai_api_key="sk-...",
)

# ── Anthropic ───────────────────────────────────────────
config = SmritiConfig(
    llm_model="claude-3-5-sonnet-20241022",
    anthropic_api_key="sk-ant-...",
)

# ── Google Gemini ───────────────────────────────────────
config = SmritiConfig(
    llm_model="gemini-1.5-flash",
    gemini_api_key="AIza...",
)

# ── Local Ollama (default) ──────────────────────────────
config = SmritiConfig(
    llm_model="mistral",  # or llama3, codellama, phi3, etc.
)

memory = SMRITI(config=config)
```

Routing is automatic based on the model name prefix: `gpt-*` → OpenAI, `claude*` → Anthropic, `gemini*` → Gemini, everything else → Ollama.

---

## Quick Start

```python
from smriti import SMRITI, SmritiConfig

# Initialize
config = SmritiConfig(
    storage_path="./my_agent_memory",
    llm_model="mistral",
)
memory = SMRITI(config=config)

# Encode information
memory.encode("User prefers Python for backend development.")
memory.encode("User is allergic to shellfish.", context="medical")

# Recall by natural-language query
results = memory.recall("What language does the user prefer?")
for mem in results:
    print(f"  [{mem.strength:.2f}] {mem.content}")

# Check what you know (and don't know)
confidence = memory.how_well_do_i_know("programming languages")
print(f"Confidence: {confidence.overall:.0%}")

# Run background consolidation
memory.consolidate()

# Persist to disk
memory.save()
```

### Framework Integrations
SMRITI can be used natively inside standard agent frameworks. 

#### LangChain
Use `SmritiLangChainMemory` to replace `ConversationBufferMemory`. This gives your agent the cost-savings of a capacity-bounded Working Memory while asynchronously archiving the conversation into the Semantic Palace.

```python
from langchain.chains import ConversationChain
from smriti.integrations.langchain_memory import SmritiLangChainMemory
from smriti import SMRITI

# 1. Initialize SMRITI
smriti_engine = SMRITI(storage_path="./langchain_smriti_db")

# 2. Wrap it for LangChain
smriti_memory = SmritiLangChainMemory(smriti_client=smriti_engine, top_k=3)

# 3. Plug it into standard chains
conversation = ConversationChain(
    llm=my_llm,
    memory=smriti_memory,
)

conversation.predict(input="I prefer using PyTorch.")
```

See [`examples/langchain_agent.py`](examples/langchain_agent.py) or [`examples/quickstart.py`](examples/quickstart.py) for complete working code.

#### Claude Code (MCP Server)

See [Quick Start — Claude Code (MCP)](#quick-start--claude-code-mcp) above for one-command setup.

### Memory Browser UI

SMRITI ships with a native, zero-dependency visualizer for traversing the Semantic Palace graph.

```bash
smriti_ui --storage ~/.smriti/global --port 7799
```

**Features:**
- **Zero dependencies:** Built entirely with Python's standard `http.server` and D3.js — no Node.js/NPM needed.
- **Backwards Compatible:** Instantly works with your existing `palace.json` created by older versions of SMRITI. Just point `--storage` to your existing directory.
- **Interactive Graph:** Navigate the Semantic Palace using a force-directed network view or clustered room topology.
- **Searchable Dashboard:** Instantly filter your stored knowledge by content, room, and system state.
- **Real-time Statistics:** Track average memory strength, composite salience, and architectural distribution.

*(If using without pip installation, run `python -m smriti_memcore.ui` from the source root).*

### Obsidian Vault Integration

Export the Semantic Palace to an [Obsidian](https://obsidian.md/) vault so its graph view mirrors your memory graph.

**How it maps:**

| Semantic Palace | Obsidian |
|---|---|
| Room | `Palace/<topic-slug>.md` note |
| Memory | Section inside room note (with strength/salience metadata) |
| Room ↔ Room edge | `[[wikilink]]` between room notes |
| `Palace/_index.md` | Overview table of all rooms and connections |

**Via MCP tool (Claude Code):** After setting `SMRITI_OBSIDIAN_PATH` in your MCP server config, call the tool directly — no Bash needed:

```
smriti_sync_obsidian()
# or with an explicit path:
smriti_sync_obsidian(vault_path="~/path/to/your-vault/Palace")
```

Add to your MCP server env in `~/.claude.json`:
```json
"SMRITI_OBSIDIAN_PATH": "~/path/to/your-vault/Palace"
```

**Via CLI (non-MCP / scripting):**

```bash
smriti_palace_to_obsidian --vault ~/path/to/your-vault/Palace
```

**Workflow:** Re-run after each `smriti_consolidate` call to keep the vault in sync with updated rooms and connections. The `Palace/` folder is fully regenerated each run — do not edit those files manually.

*(If using without pip installation, run `python -m smriti_memcore.palace_to_obsidian` from the source root).*

---

## Key API

| Method | Description |
|---|---|
| `encode(content, context, source)` | Ingest new information through the Attention Gate |
| `recall(query, top_k, rewrite, snippet)` | Retrieve relevant memories via graph traversal; `rewrite`/`snippet` accept `"auto"`/`"llm"`/`"none"` to opt into query rewriting and snippet extraction |
| `how_well_do_i_know(topic)` | Meta-memory confidence check |
| `consolidate(depth)` | Run background consolidation (`"full"`, `"light"`, `"defer"`) |
| `save()` | Persist all state to disk |
| `pin(memory_id)` | Mark a memory as permanent |
| `forget(memory_id)` | Gracefully forget a memory (leaves a tombstone) |
| `stats()` | System-wide statistics |

---

## Configuration

All parameters are optional and have sensible defaults:

```python
from smriti import SmritiConfig

config = SmritiConfig(
    # Working Memory
    working_memory_slots=7,          # Miller's Law: 7 ± 2

    # Retrieval scoring weights
    recency_weight=0.2,
    relevance_weight=0.4,
    strength_weight=0.2,
    salience_weight=0.2,

    # Forgetting
    decay_rate=0.99,                 # per-day temporal decay
    strength_hard_threshold=0.05,    # below this → forget

    # Palace graph
    room_merge_threshold=0.85,       # similarity to auto-merge rooms

    # Smart recall (all default off via "auto" mode that no-ops without flags)
    rewrite_mode_default="auto",     # "auto" | "llm" | "none" — query rewriting
    snippet_mode_default="auto",     # "auto" | "llm" | "none" — snippet extraction
    snippet_min_chars=300,           # content ≤ this is returned as-is
    snippet_max_sentences=2,         # max sentences in a snippet
    llm_rewrite_cache_size=100,      # LRU cache for LLM rewrites
    llm_rewrite_prompt_version="v1", # cache-key component for prompt changes
    adjacency_alpha=0.3,             # per-memory adjacency-lift coefficient
    adjacency_lift_max=1.0,          # cap on weighted-average adjacency lift

    # LLM provider (pick one)
    llm_model="mistral",                     # Ollama (default)
    # llm_model="gpt-4o",                    # OpenAI
    # llm_model="claude-3-5-sonnet-20241022",# Anthropic
    # llm_model="gemini-1.5-flash",          # Google
    ollama_base_url="http://localhost:11434",

    # Storage
    storage_path="./smriti_data",
)
```

---

## What's New in v1.4.0

- **Query rewriting** — `smriti_recall(rewrite="auto"|"llm"|"none")` widens recall by generating paraphrase variants. `auto` uses lexical variants; `llm` calls the configured LLM with an LRU cache. Default off.
- **Snippet extraction** — `smriti_recall(snippet="auto"|"llm"|"none")` returns a short relevant excerpt instead of full content. `auto` uses lexical sentence-match with a cosine-floor fallback for zero-overlap queries; `llm` falls back to `auto` on empty/error response. State-leak guarded. Default off.
- **Per-memory adjacency lift** — replaces the prior flat 0.85 cross-room discount with a weighted lift driven by `adjacency_alpha` (capped by `adjacency_lift_max`), plus entry-room widening.
- **New `smriti_get_memory` MCP tool** — fetch full content of a single memory after a snippet recall (`expandable=true` in the snippet response indicates truncation).
- **`SmritiConfig` smart-recall fields** — `rewrite_mode_default`, `snippet_mode_default`, `snippet_min_chars`, `snippet_max_sentences`, `llm_rewrite_cache_size`, `llm_rewrite_prompt_version`, `adjacency_alpha`, `adjacency_lift_max`.
- **Bench harness `scripts/bench_recall.py`** — hit-rate@10 + tokens/query + p95 latency, with `--baseline <sha>` for cross-branch ID-order regression check. Smoke run shows ~74.5% token reduction with features enabled.
- **No breaking changes** — all features default off; existing callers see identical behavior.

## What's New in v1.3.0

- **Private rooms** — `smriti_create_private_room(topic)` creates a semantic room whose memories are excluded from team consolidation sync
- **`private=True` on encode** — `smriti_encode` and `amp.encode` now accept `private=True`; Claude uses this when you say *"remember this privately"*
- **`Visibility` field on memories and rooms** — `"private"` | `"shared"`; default is `"shared"`. Private memories are still recalled by the owner — privacy only controls team sync eligibility
- **AMP spec updated** — `visibility` field added to `MemoryResult`, `private` param added to `amp.encode`, `visibility` filter added to `amp.recall` filters schema
- **palace.json schema v2** — automatic migration on first load; all existing memories and rooms default to `"shared"`

## What's New in v1.2.0

- **AMP v1.0 Full conformance** — MCP server now exposes all 6 AMP verbs (`amp.encode`, `amp.recall`, `amp.forget`, `amp.stats`, `amp.pin`, `amp.consolidate`) alongside the existing `smriti_*` tools. Passes all 25 AMP compliance tests (Core + Full).
- **Zero breaking changes** — all existing `smriti_*` tool calls continue to work unchanged. AMP tools are additive aliases.

## What's New in v1.0.0

- **Consolidation robustness overhaul** — fixed a critical bug where singleton episodes leaked in the buffer indefinitely, causing consolidation to report "no significant memories" even when important facts were present
- **Smarter salience scoring** — the heuristic scorer now differentiates content types (personal facts, knowledge updates, instructions) instead of scoring everything the same
- **Better contradiction detection** — Mistral no longer incorrectly discards memories that agree with existing ones
- **Validated across 4 models** — benchmarked with gpt-4o-mini, Mistral 7B, CodeLlama 7B, and Llama 3.2 3B

See [CHANGELOG.md](CHANGELOG.md) for full details.

---

## Benchmarks

### LoCoMo (Multi-System Comparison)

SMRITI was benchmarked against four baseline architectures on the [LoCoMo](https://github.com/snap-research/locomo) long-sequence dataset (28 dialog turns, 15 evaluation questions, consolidation enabled):

| System | F1 Score | Latency | Tokens/Query | Consolidation |
|---|---|---|---|---|
| FullContext | **0.345** | 1147ms | 550 | — |
| MemGPT-style | 0.334 | 1397ms | 478 | — |
| NaiveRAG | 0.312 | 1387ms | 145 | — |
| **SMRITI v2** | 0.279 | 1317ms | **146** | 41.2s (async) |
| Mem0-style | 0.235 | 1088ms | 106 | — |

*Results with GPT-4o-mini. SMRITI consolidation runs asynchronously and does not block queries.*

### Local Model Comparison (v1.0.0)

All runs use the fixed consolidation pipeline with heuristic scoring:

| Model | F1 Score | Exact Match | Latency | Best Category |
|---|---|---|---|---|
| **CodeLlama 7B** | **0.317** | **0.200** | 5634ms | Temporal (0.682) |
| Mistral 7B | 0.284 | 0.067 | 3181ms | Knowledge Update (0.516) |
| gpt-4o-mini | 0.262 | 0.000 | 1271ms | Single-hop (0.350) |
| Llama 3.2 3B | 0.184 | 0.067 | 1446ms | Multi-hop (0.134) |

**Key finding:** CodeLlama 7B outperforms all models on temporal reasoning (F1=0.682) and achieves the highest exact-match rate (20%). Mistral 7B remains the best all-rounder with strong knowledge-update handling.

### LongMemEval (Long-Term Interactive Memory)

SMRITI integrates an evaluation harness for the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark to test retrieval over 50+ chat sessions:

| System Configuration | Exact Match Accuracy | Average Query Latency |
|---|---|---|
| **Baseline (Full Context)** | 100.0% | 11.98s |
| **SMRITI Dual-Process** | **80.0%** | **0.98s** |

*SMRITI restricts the LLM context to the 5 most relevant memories, resulting in a **>12× latency reduction** compared to context-stuffing.*

### Vector Search Backend

SMRITI supports two vector search backends. FAISS is auto-detected when installed:

| Backend | 1K vectors | 10K vectors | 100K vectors | Memory (100K) |
|---|---|---|---|---|
| NumPy | 22 µs | 179 µs | 2.75 ms | 146.5 MB |
| **FAISS** | 28 µs | 200 µs | **2.24 ms** | **979 B** |

At scale, FAISS is **1.2× faster** with **150,000× less memory**.

### Reproducing Benchmarks

```bash
pip install -e ".[benchmarks]"

# Multi-system comparison (requires API key)
python benchmarks/run_benchmark.py --model gpt-4o-mini --systems smriti --consolidate --dataset locomo

# Local model comparison (requires Ollama)
python benchmarks/run_benchmark.py --model mistral --systems smriti --consolidate --dataset locomo
python benchmarks/run_benchmark.py --model codellama --systems smriti --consolidate --dataset locomo

# Vector backend comparison
python benchmarks/vector_benchmark.py
```

---

## Project Structure

```
smriti-memcore/
├── smriti_memcore/        # Core library
│   ├── __init__.py
│   ├── core.py            # SMRITI orchestrator
│   ├── models.py          # Data models & SmritiConfig
│   ├── palace.py          # Semantic Palace graph
│   ├── episode_buffer.py  # Append-only temporal log
│   ├── working_memory.py  # Capacity-bounded priority queue
│   ├── attention_gate.py  # Salience filter
│   ├── retrieval.py       # Multi-factor retrieval engine
│   ├── query_rewriter.py  # Query paraphrase variants (auto/llm)
│   ├── snippet.py         # Content snippet extraction (auto/llm)
│   ├── fts_index.py       # SQLite FTS5 lexical index (hybrid search)
│   ├── consolidation.py   # Async background processes
│   ├── meta_memory.py     # Confidence mapping
│   ├── vector_store.py    # Vector persistence
│   ├── llm_interface.py   # Multi-provider LLM connector (Ollama/OpenAI/Anthropic/Gemini)
│   ├── metrics.py         # Observability: counters, gauges, histograms, Prometheus export
│   └── integrations/      # Framework adapters
│       ├── langchain_memory.py  # LangChain BaseMemory component
│       └── mcp_server.py        # Claude Code MCP server (20 tools: 14 smriti_* + 6 AMP aliases)
├── install_smriti_mcp.sh   # One-command Claude Code setup
├── scripts/               # Utility scripts (bench_recall.py, etc.)
├── tests/                 # 298 tests across 18 files
├── baselines/             # Baseline implementations for comparison
├── benchmarks/            # Benchmark harness & scripts
├── examples/              # Usage examples
├── paper/                 # IEEE research paper (LaTeX + Markdown)
│   └── figures/           # Benchmark charts and UI diagrams
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Citation

If you use SMRITI in your research, please cite:

```bibtex
@article{tyagi2025smriti,
  title={SMRITI: A Scalable, Neuro-Inspired Architecture for Long-Term Event Memory in LLM Agents},
  author={Tyagi, Shivam},
  year={2025},
  doi={10.13140/RG.2.2.25477.82407}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
