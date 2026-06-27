# SMRITI Memory

> **Enterprise-grade, privacy-first Long-Term Memory (LTM) engine for AI agents and multi-agent systems.**

[![PyPI](https://img.shields.io/pypi/v/smriti-memcore.svg)](https://pypi.org/project/smriti-memcore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!-- mcp-name: io.github.shivamtyagi18/smriti-memory -->

---

SMRITI is a high-performance, neuro-inspired long-term memory architecture designed to give LLM agents persistent, structured recall without blocking real-time loops. By separating memory operations into System 1 (fast ingestion) and System 2 (asynchronous analytical consolidation), SMRITI provides a highly responsive, resource-efficient memory layer.

## 🚀 Key Features

*   ⚡ **Dual-Process Architecture**: Decouples fast System 1 ingestion (append-only short-term buffer) from slow System 2 background consolidation (LLM-driven knowledge abstraction, relation extraction, and defragmentation).
*   🔒 **Privacy-First (Private Rooms)**: Support for in-memory private semantic rooms and visibility tags (`private` vs. `shared`) to keep sensitive user information excluded from team-wide memory synchronization.
*   🔌 **Out-of-the-Box MCP Server**: Natively compliant with the Model Context Protocol (MCP), supporting seamless integration with Claude Code, Claude Desktop, Gemini Antigravity, and Codex (Antigravity-IDE).
*   📦 **Agent Memory Protocol (AMP v1.0) Support**: Standardized API endpoints (`amp.*` aliases) ensuring compatibility with any AMP-compliant agent framework.
*   📊 **Visual Observability**: Zero-dependency interactive web interface (D3.js-based memory graph visualizer) and built-in Prometheus metrics.
*   📂 **Obsidian Vault Syncing**: Automatically syncs the agent's Semantic Palace memory graph into an Obsidian vault for human curation and knowledge tracking.
*   🧩 **Framework Integrations**: Simple wrappers and adapters for LangChain, LlamaIndex, and other popular Python agent frameworks.

---

## 🧠 Core Architecture

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
  │  └───────┘   │ (salience │   └─────────────────────┘   │  Graph   │
  │              │  filter)  │                              │ G=(V,E)  │
  └──────────┘   └──────────┘                              └────┬─────┘
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

*   **System 1 (Heuristic Pathway)**: Routes incoming raw interactions immediately to the short-term Episode Buffer in milliseconds.
*   **System 2 (Analytical Pathway)**: Runs asynchronously in the background. It structures raw logs into the **Semantic Palace Graph**, handles contradiction resolution, extracts skills, and decays weak memories over time.

---

## 🏁 Quick Start

### 1. Unified MCP Server (Claude Code, Gemini, Codex)

SMRITI can be used as a global, persistent memory layer across all your MCP-enabled developer clients.

#### Method A: One-Line Installer (Recommended)
Run the setup script directly in your terminal:
```bash
bash <(curl -s https://raw.githubusercontent.com/smriti-memcore/smriti-memcore/main/install_smriti_mcp.sh)
```

#### Method B: Via PyPI
Install the package and run the setup CLI:
```bash
pip3 install smriti-memcore
smriti_install
```

*The installer configures a Python virtual environment, prompts for your preferred LLM provider, and registers the server in Claude, Gemini, and Codex config files.*

---

### 2. Python SDK

For application developers building custom agent loops.

```bash
pip install smriti-memcore[faiss] # FAISS is recommended for accelerated vector search
```

```python
from smriti import SMRITI, SmritiConfig

# Initialize memory engine with OpenAI
config = SmritiConfig(
    storage_path="./my_agent_memory",
    llm_model="gpt-4o",
    openai_api_key="your-api-key-here"
)
memory = SMRITI(config=config)

# Ingest observations
memory.encode("User prefers using PyTorch for neural networks.")
memory.encode("User is allergic to shellfish.", context="medical")

# Recall relevant context using multi-factor retrieval
results = memory.recall("What framework does the user prefer?")
for mem in results:
    print(f"[{mem.strength:.2f}] {mem.content}")

# Manually trigger System 2 background consolidation
memory.consolidate()
memory.save()
```

---

## 🛠️ MCP Tool Reference

SMRITI exposes **19 tools** (13 native + 6 AMP aliases) for clients:

### Core Tools

| Tool Name | Description |
|---|---|
| `smriti_encode` | Ingests a new memory. Accept `private=True` to exclude from team syncs. |
| `smriti_recall` | Retrieves memories using semantic and graph-based traversal. |
| `smriti_get_context` | Helper to inject the current active working memory slots into the context window. |
| `smriti_how_well_do_i_know` | Performs a meta-memory confidence check on a given topic. |
| `smriti_knowledge_gaps` | Identifies topics the agent has identified it needs more information on. |
| `smriti_pin` | Marks a memory as permanent (protects it from strength decay). |
| `smriti_forget` | Soft-deletes/archives a memory, leaving a cryptographic tombstone. |
| `smriti_consolidate` | Triggers a background System 2 consolidation run. |
| `smriti_stats` | Returns system-wide statistics (total memories, rooms, private counts). |
| `smriti_create_private_room` | Spawns a private room. All memories inside this room are visibility-isolated. |
| `smriti_open_ui` | Launches the interactive visual D3.js memory graph in your default browser. |
| `smriti_sync_obsidian` | Exports the Semantic Palace graph structures to markdown files in an Obsidian Vault. |

### AMP v1.0 Alias Tools
These endpoints ensure complete conformance with the standard Agent Memory Protocol specification:

| AMP Tool | Native Mapping | Return Format |
|---|---|---|
| `amp.encode` | `smriti_encode` | AMP standard JSON response |
| `amp.recall` | `smriti_recall` | Array of `{id, content, score, timestamp, status}` |
| `amp.forget` | `smriti_forget` | `{status: "forgotten" \| "not_found"}` |
| `amp.stats` | `smriti_stats` | `{memory_count, ...}` |
| `amp.pin` | `smriti_pin` | `{status: "pinned" \| "not_found"}` |
| `amp.consolidate` | `smriti_consolidate` | `{status: "ok", memories_processed: int}` |

---

## 🔌 Framework Integrations

### LangChain Integration
Use `SmritiLangChainMemory` as a drop-in replacement for default chat buffers. It limits active context using Working Memory and offloads the conversational history to the Semantic Palace graph in the background.

```python
from langchain.chains import ConversationChain
from smriti.integrations.langchain_memory import SmritiLangChainMemory
from smriti import SMRITI

smriti_engine = SMRITI(storage_path="./langchain_smriti_db")
smriti_memory = SmritiLangChainMemory(smriti_client=smriti_engine, top_k=3)

conversation = ConversationChain(
    llm=my_llm,
    memory=smriti_memory,
)
conversation.predict(input="I prefer backend APIs in Python.")
```

---

## 📊 Benchmarks & Performance

### 1. LoCoMo (Multi-System Context Retrieval)
Tested against four architectures on the [LoCoMo](https://github.com/snap-research/locomo) long-context dialogue dataset (28 turns, 15 evaluation questions):

| System | F1 Score | Latency | Tokens/Query | Consolidation |
|---|---|---|---|---|
| FullContext | **0.345** | 1147ms | 550 | — |
| MemGPT-style | 0.334 | 1397ms | 478 | — |
| NaiveRAG | 0.312 | 1387ms | 145 | — |
| **SMRITI** | 0.279 | 1317ms | **146** | 41.2s (async) |
| Mem0-style | 0.235 | 1088ms | 106 | — |

*SMRITI retains high recall while drastically reducing query context size. Consolidation runs in the background and does not block client interactions.*

### 2. LongMemEval (Long-Term Chat Sessions)
Evaluated over 50+ chat sessions using the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) harness:

| System Configuration | Exact Match Accuracy | Average Query Latency |
|---|---|---|
| Baseline (Full Context) | **100.0%** | 11.98s |
| **SMRITI Dual-Process** | **80.0%** | **0.98s** (12× latency reduction) |

---

## ⚙️ Configuration Parameters

Initialize `SmritiConfig` with custom parameters to tune the cognitive weights:

```python
from smriti import SmritiConfig

config = SmritiConfig(
    working_memory_slots=7,          # Capacity limit (Miller's Law)
    
    # Retrieval scoring weights (sum to 1.0)
    recency_weight=0.2,
    relevance_weight=0.4,
    strength_weight=0.2,
    salience_weight=0.2,

    # Forgetting & Temporal Decay
    decay_rate=0.99,                 # Strength multiplier per day
    strength_hard_threshold=0.05,    # Memories dropping below this are forgotten
    
    # Palace Graph
    room_merge_threshold=0.85,       # Cosine similarity for auto-merging semantic rooms
)
```

---

## 📄 Citation

If you use SMRITI in your research, please cite our technical paper:

```bibtex
@article{tyagi2025smriti,
  title={SMRITI: A Scalable, Neuro-Inspired Architecture for Long-Term Event Memory in LLM Agents},
  author={Tyagi, Shivam},
  year={2025},
  doi={10.13140/RG.2.2.25477.82407}
}
```

---

## 📄 License

SMRITI is licensed under the MIT License. See [LICENSE](LICENSE) for details.
