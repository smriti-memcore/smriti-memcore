#!/usr/bin/env python3
"""
Benchmark recall quality and token cost.

Seeds N=500 synthetic memories (each ≥400 chars to exercise the snippet path)
across 20 topics; runs 50 paraphrased queries with rewrite='auto', snippet='auto'
vs rewrite='none', snippet='none'; reports hit-rate@10, avg tokens/query,
p95 latency.

Not in CI. Run twice — once on `main` (baseline), once on the feature branch.
Put numbers in the PR description.

Usage:
    # On main:
    python3 scripts/bench_recall.py --label main_baseline --out /tmp/main.json

    # On feature branch:
    python3 scripts/bench_recall.py --label feature --out /tmp/feature.json --baseline /tmp/main.json
"""
import argparse
import json
import random
import statistics
import tempfile
import time

from smriti_memcore.core import SMRITI
from smriti_memcore.models import MemorySource, SmritiConfig

TOPICS = [
    "python", "javascript", "rust", "go", "java",
    "kubernetes", "docker", "terraform", "ansible", "helm",
    "postgres", "mysql", "sqlite", "redis", "mongodb",
    "react", "vue", "angular", "svelte", "ember",
]


def seed(s: SMRITI, n: int = 500) -> dict:
    """Seed `n` memories. Each content body is ≥ 400 chars so snippet_min_chars=300
    is exceeded and snippet="auto" actually trims content (otherwise tokens would
    not measurably reduce)."""
    target_ids = {}
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        content = (
            f"This memory {i} is about {topic} and how it relates to general programming patterns. "
            f"The {topic} ecosystem includes packaging tools, build systems, testing frameworks, and deployment helpers. "
            f"People who use {topic} often pair it with adjacent tooling and integrate it into broader workflows. "
            f"Communities around {topic} maintain documentation, tutorials, and reference implementations. "
            f"Random salt {random.randint(0, 1_000_000)} for uniqueness. "
            f"Performance characteristics of {topic} matter for production deployment. "
        )
        assert len(content) >= 400, f"seed content too short: {len(content)} chars"
        mid = s.encode(content, source=MemorySource.USER_STATED, use_llm=False)
        if mid:
            target_ids.setdefault(topic, []).append(mid)
    return target_ids


def _call_recall_compat(s: SMRITI, q: str, rewrite: str, snippet: str, top_k: int):
    """Call SMRITI.recall compatibly with both pre-change `main` and the feature branch.

    Pre-change main lacks rewrite/snippet kwargs. Detect via inspect at first call;
    when running on main we always do raw recall regardless of the requested mode."""
    import inspect
    sig = inspect.signature(s.recall)
    if "rewrite" in sig.parameters:
        return s.recall(q, rewrite=rewrite, snippet=snippet, top_k=top_k)
    return s.recall(q, top_k=top_k)


def run_queries(s: SMRITI, target_ids: dict, queries: list, rewrite: str, snippet: str):
    hits_at_10 = 0
    total_tokens = 0
    latencies = []
    per_query_ids = []  # for cross-branch ID-order baseline check (spec §13.4)
    for q, expected_topic in queries:
        t0 = time.perf_counter()
        results = _call_recall_compat(s, q, rewrite=rewrite, snippet=snippet, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        result_ids = [m.id for m in results]
        per_query_ids.append({"query": q, "ids": result_ids})
        if set(result_ids) & set(target_ids.get(expected_topic, [])):
            hits_at_10 += 1
        for m in results:
            content = getattr(m, "snippet", None) or m.content
            total_tokens += len(content.split())
    n = max(1, len(queries))
    return {
        "hit_rate_at_10": hits_at_10 / n,
        "avg_tokens_per_query": total_tokens / n,
        "p95_latency_ms": (
            statistics.quantiles(latencies, n=20)[-1]
            if len(latencies) >= 20 else max(latencies)
        ),
        "per_query_ids": per_query_ids,   # captured for ID-order comparison
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run", help="label for this run (for output JSON)")
    ap.add_argument("--out", default=None, help="write the result JSON to this path")
    ap.add_argument("--baseline", default=None, help="JSON from a prior run (typically main) to diff against")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = ap.parse_args()

    random.seed(args.seed)

    with tempfile.TemporaryDirectory() as d:
        s = SMRITI(SmritiConfig(storage_path=d))
        print(f"Seeding 500 memories (seed={args.seed})…")
        target_ids = seed(s, n=500)

        queries = []
        for topic in TOPICS:
            for paraphrase in [
                f"how does {topic} work",
                f"tell me about {topic}",
                f"{topic} ecosystem",
            ]:
                queries.append((paraphrase, topic))
        random.shuffle(queries)
        queries = queries[:50]

        print(f"\n[{args.label}] rewrite=none, snippet=none …")
        before = run_queries(s, target_ids, queries, rewrite="none", snippet="none")
        # Strip per_query_ids from printed output for readability
        before_display = {k: v for k, v in before.items() if k != "per_query_ids"}
        print(f"  → {json.dumps(before_display, indent=2)}")

        print(f"\n[{args.label}] rewrite=auto, snippet=auto …")
        after = run_queries(s, target_ids, queries, rewrite="auto", snippet="auto")
        after_display = {k: v for k, v in after.items() if k != "per_query_ids"}
        print(f"  → {json.dumps(after_display, indent=2)}")

        result = {"label": args.label, "disabled": before, "enabled": after}

        tok_delta = (before["avg_tokens_per_query"] - after["avg_tokens_per_query"]) / max(before["avg_tokens_per_query"], 1) * 100
        hit_delta = (after["hit_rate_at_10"] - before["hit_rate_at_10"]) * 100
        print()
        print(f"Within-branch deltas:")
        print(f"  Token reduction: {tok_delta:.1f}%")
        print(f"  Hit-rate@10 delta: {hit_delta:+.1f} percentage points")

        if args.baseline:
            try:
                with open(args.baseline) as f:
                    base = json.load(f)
                base_disabled = base["disabled"]
                # The "disabled" mode on this branch must match the baseline (regression guard).
                hit_drift = (before["hit_rate_at_10"] - base_disabled["hit_rate_at_10"]) * 100
                print(f"\nCross-branch regression check (disabled-features vs baseline):")
                print(f"  Hit-rate@10 drift: {hit_drift:+.1f}pp (acceptance: ≤ 2pp)")

                # Spec §13.4 — same memory IDs in the same order vs pre-change main.
                # Compare per-query result lists.
                exact_match_count = 0
                total = 0
                for cur, prev in zip(before["per_query_ids"], base_disabled["per_query_ids"]):
                    total += 1
                    if cur["ids"] == prev["ids"]:
                        exact_match_count += 1
                pct = (exact_match_count / total * 100) if total else 0.0
                print(f"  ID-order exact-match: {exact_match_count}/{total} queries ({pct:.0f}%)")
                if exact_match_count != total:
                    print("  WARNING: ID-order drift detected — spec §13.4 acceptance criterion requires investigation.")
            except Exception as e:
                print(f"\n(could not read baseline {args.baseline}: {e})")

        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults written to {args.out}")

        s.close()


if __name__ == "__main__":
    main()
