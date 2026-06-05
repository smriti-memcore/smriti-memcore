"""
SMRITI v2 — Metrics & Observability.
Zero-dependency, thread-safe metrics for monitoring load, latency,
performance, and memory health. Supports JSON snapshot and
Prometheus text format export.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class _Counter:
    """Thread-safe monotonic counter."""
    __slots__ = ("_value", "_lock")

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, n: int = 1):
        with self._lock:
            self._value += n

    @property
    def value(self) -> int:
        return self._value


class _Gauge:
    """Thread-safe gauge (can go up or down)."""
    __slots__ = ("_value", "_lock")

    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def set(self, v: float):
        with self._lock:
            self._value = v

    def inc(self, n: float = 1):
        with self._lock:
            self._value += n

    def dec(self, n: float = 1):
        with self._lock:
            self._value -= n

    @property
    def value(self) -> float:
        return self._value


class _Histogram:
    """Thread-safe histogram tracking count, sum, min, max, and recent values for percentiles."""
    __slots__ = ("_count", "_sum", "_min", "_max", "_recent", "_lock")

    def __init__(self, window: int = 500):
        self._count = 0
        self._sum = 0.0
        self._min = float("inf")
        self._max = 0.0
        self._recent: deque = deque(maxlen=window)
        self._lock = threading.Lock()

    def observe(self, value: float):
        with self._lock:
            self._count += 1
            self._sum += value
            if value < self._min:
                self._min = value
            if value > self._max:
                self._max = value
            self._recent.append(value)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self._count == 0:
                return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0,
                        "p50": 0, "p95": 0, "p99": 0}
            sorted_vals = sorted(self._recent)
            n = len(sorted_vals)
            return {
                "count": self._count,
                "sum": round(self._sum, 2),
                "avg": round(self._sum / self._count, 2),
                "min": round(self._min, 2),
                "max": round(self._max, 2),
                "p50": round(sorted_vals[int(n * 0.5)], 2) if n else 0,
                "p95": round(sorted_vals[int(n * 0.95)], 2) if n else 0,
                "p99": round(sorted_vals[min(int(n * 0.99), n - 1)], 2) if n else 0,
            }


class SmritiMetrics:
    """
    Centralized metrics collector for SMRITI.

    Usage:
        metrics = SmritiMetrics()
        metrics.encode_count.inc()
        metrics.encode_latency.observe(42.5)
        print(metrics.snapshot())     # JSON-friendly dict
        print(metrics.prometheus())   # Prometheus text format
    """

    def __init__(self):
        # ── Counters ──
        self.encode_count = _Counter()
        self.encode_discarded = _Counter()
        self.recall_count = _Counter()
        self.recall_empty = _Counter()
        self.consolidation_count = _Counter()
        self.consolidation_errors = _Counter()
        self.compression_count = _Counter()
        self.original_retrieval_count = _Counter()
        self.llm_call_count = _Counter()
        self.llm_errors = _Counter()

        # ── Histograms (latency) ──
        self.encode_latency = _Histogram()      # ms
        self.recall_latency = _Histogram()       # ms
        self.consolidation_latency = _Histogram()  # seconds
        self.compression_ratio = _Histogram()    # compressed/original ratio
        self.llm_latency = _Histogram()          # ms

        # ── Gauges (current state) ──
        self.memory_count = _Gauge()
        self.room_count = _Gauge()
        self.episode_count = _Gauge()
        self.vector_count = _Gauge()
        self.working_memory_occupancy = _Gauge()

        # ── Startup time ──
        self._start_time = time.time()

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._start_time, 1)

    def snapshot(self) -> Dict[str, Any]:
        """Return all metrics as a JSON-serializable dict."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "operations": {
                "encode": {
                    "total": self.encode_count.value,
                    "discarded": self.encode_discarded.value,
                    "latency_ms": self.encode_latency.snapshot(),
                },
                "recall": {
                    "total": self.recall_count.value,
                    "empty": self.recall_empty.value,
                    "latency_ms": self.recall_latency.snapshot(),
                },
                "consolidation": {
                    "total": self.consolidation_count.value,
                    "errors": self.consolidation_errors.value,
                    "latency_s": self.consolidation_latency.snapshot(),
                },
                "compression": {
                    "total_attempts": self.compression_count.value,
                    "original_retrievals": self.original_retrieval_count.value,
                    "ratio": self.compression_ratio.snapshot(),
                },
                "llm": {
                    "total_calls": self.llm_call_count.value,
                    "errors": self.llm_errors.value,
                    "latency_ms": self.llm_latency.snapshot(),
                },
            },
            "state": {
                "memories": self.memory_count.value,
                "rooms": self.room_count.value,
                "episodes": self.episode_count.value,
                "vectors": self.vector_count.value,
                "working_memory_slots_used": self.working_memory_occupancy.value,
            },
        }

    def prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines = []

        def _counter(name: str, help_text: str, value: int):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        def _gauge(name: str, help_text: str, value: float):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        def _hist(name: str, help_text: str, h: _Histogram):
            snap = h.snapshot()
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} summary")
            lines.append(f'{name}{{quantile="0.5"}} {snap["p50"]}')
            lines.append(f'{name}{{quantile="0.95"}} {snap["p95"]}')
            lines.append(f'{name}{{quantile="0.99"}} {snap["p99"]}')
            lines.append(f"{name}_sum {snap['sum']}")
            lines.append(f"{name}_count {snap['count']}")

        _counter("smriti_encode_total", "Total encode operations", self.encode_count.value)
        _counter("smriti_encode_discarded_total", "Discarded by attention gate", self.encode_discarded.value)
        _hist("smriti_encode_latency_ms", "Encode latency in ms", self.encode_latency)

        _counter("smriti_recall_total", "Total recall operations", self.recall_count.value)
        _counter("smriti_recall_empty_total", "Recalls returning zero results", self.recall_empty.value)
        _hist("smriti_recall_latency_ms", "Recall latency in ms", self.recall_latency)

        _counter("smriti_consolidation_total", "Total consolidations", self.consolidation_count.value)
        _counter("smriti_consolidation_errors_total", "Failed consolidation processes", self.consolidation_errors.value)
        _hist("smriti_consolidation_latency_seconds", "Consolidation duration", self.consolidation_latency)

        _counter("smriti_compression_total", "Total compression attempts", self.compression_count.value)
        _counter("smriti_original_retrieval_total", "Requests for uncompressed originals", self.original_retrieval_count.value)
        _hist("smriti_compression_ratio", "Ratio of compressed to original length", self.compression_ratio)

        _counter("smriti_llm_calls_total", "Total LLM calls", self.llm_call_count.value)
        _counter("smriti_llm_errors_total", "Failed LLM calls", self.llm_errors.value)
        _hist("smriti_llm_latency_ms", "LLM call latency in ms", self.llm_latency)

        _gauge("smriti_memories", "Current stored memories", self.memory_count.value)
        _gauge("smriti_rooms", "Current palace rooms", self.room_count.value)
        _gauge("smriti_episodes", "Current episodes", self.episode_count.value)
        _gauge("smriti_vectors", "Current vector store entries", self.vector_count.value)
        _gauge("smriti_working_memory_slots", "Working memory slots used", self.working_memory_occupancy.value)
        _gauge("smriti_uptime_seconds", "Time since initialization", self.uptime_seconds)

        return "\n".join(lines) + "\n"
