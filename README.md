<h1 align="center">SMRITI Memcore</h1>

<p align="center">
  <strong>Enterprise-grade, privacy-first Long-Term Memory (LTM) engine for LLM agents, multi-agent frameworks, and MCP clients.</strong>
</p>

<p align="center">
  🌐 <a href="https://www.smriti-memcore.com"><strong>www.smriti-memcore.com</strong></a>
</p>

[![PyPI](https://img.shields.io/pypi/v/smriti-memcore.svg)](https://pypi.org/project/smriti-memcore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Website](https://img.shields.io/badge/website-smriti--memcore.com-c5a059?style=flat&logo=vercel)](https://www.smriti-memcore.com)
[![SMRITI Memory MCP server](https://mcprush.com/badge/smriti-memory-mcp/card.svg)](https://mcprush.com/to/smriti-memory-mcp?ref=badge)

<!-- mcp-name: io.github.smriti-memcore/smriti-memory -->

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
*   🎭 **Persona Palaces & On-Demand Hub**: Download pre-curated cognitive memory palaces for historical figures (e.g., Albert Einstein, Isaac Newton) and converse with them with authentic mental models.
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

### 1. Install SMRITI

Install the core memory engine and CLI via pip:

```bash
# Recommended for standalone CLI (isolated environment + automatic PATH setup):
pipx install smriti-memcore
# or inside a virtualenv:
pip install smriti-memcore
```

*(Optional: Ensure [Ollama](https://ollama.com) is running locally for 100% private, free LLM inference: `ollama run mistral`)*

---

### 2. Instructions for Professors & Course Creators: Package Your Curriculum

Turn your raw lecture folder into a self-contained, offline-first **Cognitive Palace Bundle** (`.smritibundle`) that acts as a 24/7 Socratic Teaching Assistant for your students.

```bash
# Package a folder of slides, transcripts, readings, or notebooks into an offline bundle:
smriti pack ./my-course-folder -o cs101.smritibundle

# Or package directly from a declarative YAML/JSON persona spec:
smriti pack ./course-spec.yaml -o cs101.smritibundle --install
```

* **Supported Formats**: PDF slides, WebVTT/SRT video lecture transcripts, Markdown, plain text, and Jupyter notebooks (`.ipynb`).
* **Zero Cloud Costs**: All vector embeddings are calculated locally ($0 OpenAI/HuggingFace API bills).
* **Lightweight Distribution**: Generates a self-contained archive (typically 5–15 MB) containing the vector index, full-text search, and semantic graph. Upload it directly to Canvas, Blackboard, Piazza, Google Drive, or email it to students.

---

### 3. Instructions for Students: 24/7 Socratic Office Hours

Students download the professor's `.smritibundle` and converse with an interactive course mentor directly in their terminal—100% offline, private, and subscription-free.

```bash
# Step 1: Import the course bundle provided by your professor
smriti pull ./cs101.smritibundle

# Step 2: Start your 24/7 Socratic Office Hours
smriti chat cs101

# (Optional) Use a specific local Ollama model (e.g. Mistral, Qwen, Llama)
smriti chat cs101 --model mistral
smriti chat cs101 --model qwen3.5
```

* **Pinpoint Attribution**: Asks about any concept and surfaces exact slide page numbers and video lecture transcript quotes.
* **Socratic Pedagogy**: Rather than simply giving away homework answers, the course persona guides students step-by-step from first principles using course analogies.
* **100% Local & Private**: All conversations stay on the student's laptop—zero student data is sent to external servers.

#### 🌐 Pre-Curated Mentors in Official Registry
You can also explore out-of-the-box mentors right away:
```bash
# List all available mentors and locally installed bundles
smriti list

# Socratic Distributed Systems Professor (Raft, Paxos, CAP theorem)
smriti chat prof-systems

# 100% Private, Local CBT & Mindfulness Companion
smriti chat mindful-counsellor

# Historical Thinkers
smriti chat einstein
smriti chat newton
smriti chat marcus-aurelius
```

---

### 4. Smriti DevBrain: Universal Developer Memory (Cursor, Claude, Windsurf, Gemini, Zed)

SMRITI solves cross-agent amnesia by acting as a shared, local cognitive brain across all your AI coding tools. A bug solved in Claude Code or Gemini is immediately remembered by Cursor.

#### Step 1: Auto-Configure All Installed IDEs & Agents
Run one command to scan and configure Cursor, Claude Code, Windsurf, Gemini, and Zed in seconds:
```bash
smriti install
```
*(Or target specific tools: `smriti install --client cursor,windsurf`)*

#### Step 2: Initialize a Codebase Workspace
Inside any repository, run:
```bash
smriti init
```
This automatically:
- Creates a dedicated Semantic Room in the Palace for the project (`repo:<name>`).
- Scaffolds `.cursorrules`, `.cursor/rules/smriti.mdc`, `CLAUDE.md`, and project `.cursor/mcp.json`.
- Installs a lightweight, non-blocking git `post-commit` hook that records recent commits and architectural changes into memory.

#### Git Commit Memory Hook Management:
```bash
smriti hook install    # Enable automatic commit context logging
smriti hook uninstall  # Remove the post-commit hook
```

---

### 3. Python SDK

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

# --- Or Load a Pre-Curated Persona (e.g. Albert Einstein) ---
einstein, manifest = SMRITI.from_persona("albert-einstein")
memories = einstein.recall("What thought experiment led to Special Relativity?")
for mem in memories:
    print(f"[{mem.strength:.2f}] {mem.content}")
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
