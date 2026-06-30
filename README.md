<h1 align="center">SMRITI Memcore</h1>

<p align="center">
  <strong>Enterprise-grade, privacy-first Long-Term Memory (LTM) engine for LLM agents, multi-agent frameworks, and MCP clients.</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/smriti-memcore.svg)](https://pypi.org/project/smriti-memcore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<!-- mcp-name: io.github.shivamtyagi18/smriti-memory -->

---

## 💡 What is SMRITI?

SMRITI is a high-performance, neuro-inspired long-term memory layer designed to give AI agents persistent, adaptive recall without blocking their real-time execution loop. 

Inspired by human Dual-Process cognitive theory, SMRITI splits memory operations into:
1. **System 1 (Immediate Heuristics)**: Decoupled, millisecond-level ingestion of raw interactions into an append-only Episode Buffer.
2. **System 2 (Async Consolidation)**: Background LLM-driven consolidation that extracts knowledge graphs, resolves contradictions, identifies skills, and decays weak memories.

---

## ⚔️ SMRITI vs. Naive RAG & Vector Databases

| Feature | Naive RAG / Vector DBs | SMRITI Memory Engine |
|---|---|---|
| **Latency** | Scales linearly with context size; blocks agent loops | **Sub-5ms ingestion** (System 1); System 2 is asynchronous |
| **Context Window** | Stuffs raw logs, leading to prompt bloat and distraction | **Miller's Law (7 ± 2 slots)** capacity-bounded Working Memory |
| **Data Evolution** | Static embeddings; struggles with contradictions/corrections | **Automatic conflict resolution**, abstraction, and temporal decay |
| **Relationships** | Flat vector search; no concept of entity links | **Semantic Palace Graph** showing structured Room/Topic associations |
| **Privacy & Sync** | All-or-nothing storage; complex namespace routing | **Private Rooms** and `private=True` tags natively isolating user syncs |

---

## 🚀 Key Capabilities

*   🧠 **Dual-Process Performance**: Zero-blocking real-time loops. Write immediately, analyze when idle.
*   🔒 **Privacy-First (Private Rooms)**: Create local semantic rooms whose memories are automatically excluded from shared/team-wide sync.
*   🔌 **Model Context Protocol (MCP)**: Native MCP server integration with Claude Code, Claude Desktop, Gemini Antigravity, and Codex.
*   📦 **AMP v1.0 Spec Compliant**: Drop-in compatibility with any agent framework conforming to the Agent Memory Protocol.
*   📊 **Visual Graph Explorer**: Clean D3.js-based visualization interface with Prometheus metrics monitoring.
*   📂 **Obsidian Vault Integration**: Automatically syncs your agent's memory graph into an Obsidian vault for human curation.
*   🧩 **Framework Agnostic**: Integrates natively with LangChain, LlamaIndex, CrewAI, and AutoGen.

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
| `smriti_recall` | Retrieves memories using semantic and graph-based retrieval. |
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
