"""
Benchmark: hybrid (FTS5+RRF) vs. vector-only retrieval.

Usage:
    python3 -m benchmarks.bench_hybrid_search

No LLM required — encodes memories with use_llm=False.
"""
from __future__ import annotations

import tempfile
import time
from typing import List, Tuple

from smriti_memcore.core import SMRITI
from smriti_memcore.models import MemorySource, SmritiConfig

# ── Dataset ───────────────────────────────────────────────────────────────────

_MEMORIES = [
    "ticket YEP-293 causes auth regression in login flow",
    "version 2.3.1 introduced breaking API change in payments module",
    "smriti_encode function crashes when content exceeds max_length",
    "PR #456 merges the refactor of the episode buffer subsystem",
    "JIRA-881 database deadlock on concurrent palace writes",
    "Python is preferred for data science and machine learning work",
    "Test-driven development improves long-term code quality and design",
    "Retrieval pipeline scores memories on cosine similarity and recency",
    "Working memory holds 7 plus or minus 2 items at any time",
    "Obsidian vault provides a human-readable mirror of the semantic palace",
] + [f"background noise topic number {i} unrelated to everything" for i in range(10)]

# (query, expected_substring, category)
_QUERIES: List[Tuple[str, str, str]] = [
    ("YEP-293",              "YEP-293",          "exact"),
    ("version 2.3.1",        "2.3.1",            "exact"),
    ("smriti_encode crash",  "smriti_encode",    "exact"),
    ("PR 456 episode buffer","PR #456",          "exact"),
    ("JIRA-881 deadlock",    "JIRA-881",         "exact"),
    ("best language for data science",    "data science",     "semantic"),
    ("how to write tests well",           "Test-driven",      "semantic"),
    ("how does retrieval scoring work",   "cosine similarity","semantic"),
    ("how many slots in working memory",  "7 plus",           "semantic"),
    ("palace sync to obsidian",           "Obsidian",         "semantic"),
]

# ── Runner ────────────────────────────────────────────────────────────────────

def _run_queries(smriti: SMRITI) -> dict:
    hits5 = hits1 = 0
    mrr = 0.0
    latencies = []
    by_cat: dict = {"exact": [], "semantic": []}

    for query, expected, cat in _QUERIES:
        t0 = time.perf_counter()
        results = smriti.recall(query, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

        rank = next(
            (i + 1 for i, m in enumerate(results)
             if expected.lower() in m.content.lower()),
            None,
        )
        hits5 += rank is not None
        hits1 += rank == 1
        mrr += (1.0 / rank) if rank else 0.0
        if cat in by_cat:
            by_cat[cat].append(rank is not None)

    n = len(_QUERIES)
    return {
        "hit@5":       hits5 / n,
        "hit@1":       hits1 / n,
        "mrr":         mrr / n,
        "avg_ms":      sum(latencies) / n,
        "exact_hit@5": sum(by_cat["exact"]) / len(by_cat["exact"]),
        "sem_hit@5":   sum(by_cat["semantic"]) / len(by_cat["semantic"]),
    }


def _print_table(vo: dict, hy: dict):
    rows = [
        ("Hit@5 — all queries",   "hit@5",       ".0%"),
        ("Hit@1 — all queries",   "hit@1",       ".0%"),
        ("MRR  — all queries",    "mrr",         ".3f"),
        ("Hit@5 — exact-term",    "exact_hit@5", ".0%"),
        ("Hit@5 — semantic",      "sem_hit@5",   ".0%"),
        ("Avg latency (ms)",      "avg_ms",      ".1f"),
    ]
    W = 62
    print("\n" + "=" * W)
    print(f"  {'Metric':<32} {'Vector-only':>12} {'Hybrid FTS':>12}")
    print("=" * W)
    for label, key, fmt in rows:
        print(f"  {label:<32} {vo[key]:>12{fmt}} {hy[key]:>12{fmt}}")
    print("=" * W + "\n")


def main():
    with tempfile.TemporaryDirectory(prefix="smriti_bench_") as tmp:
        config = SmritiConfig(storage_path=tmp)
        smriti = SMRITI(config=config)

        print(f"Encoding {len(_MEMORIES)} memories (no LLM)...")
        encoded = sum(
            1 for c in _MEMORIES
            if smriti.encode(c, source=MemorySource.USER_STATED, use_llm=False)
        )
        print(f"  {encoded}/{len(_MEMORIES)} encoded\n")

        # Vector-only: disable FTS temporarily on the shared palace
        saved_fts = smriti.retrieval_engine.fts_index
        smriti.retrieval_engine.fts_index = None
        print("Running vector-only queries...")
        vo = _run_queries(smriti)

        # Hybrid: restore FTS
        smriti.retrieval_engine.fts_index = saved_fts
        print("Running hybrid queries...")
        hy = _run_queries(smriti)

        smriti.close()

    _print_table(vo, hy)


if __name__ == "__main__":
    main()
