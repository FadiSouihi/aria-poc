"""Counters and histograms with percentile snapshots.

Phase-0 scope: in-process metrics for tests, the governor, and the Phase 6
dashboard. Async-safe via a lock; histograms are bounded (last N samples).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, Iterator, List

_HIST_CAP = 4096


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = {}
        self._hists: Dict[str, Deque[float]] = {}

    # -- writes ----------------------------------------------------------
    def inc(self, name: str, n: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + n

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            hist = self._hists.setdefault(name, deque(maxlen=_HIST_CAP))
            hist.append(float(value))

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - start)

    # -- reads -----------------------------------------------------------
    def counter(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, 0.0)

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        """Aggregate view: counters plus per-histogram count/p50/p95/max."""
        with self._lock:
            counters = dict(self._counters)
            hists = {name: list(vals) for name, vals in self._hists.items()}
        out: Dict[str, Dict[str, object]] = {"counters": counters, "histograms": {}}
        for name, values in hists.items():
            if not values:
                out["histograms"][name] = {"count": 0}
                continue
            ordered = sorted(values)
            out["histograms"][name] = {
                "count": len(ordered),
                "p50": _pctl(ordered, 50),
                "p95": _pctl(ordered, 95),
                "max": ordered[-1],
            }
        return out


def _pctl(sorted_values: List[float], pct: float) -> float:
    """Nearest-rank percentile with round-half-up (deterministic, no numpy)."""
    if not sorted_values:
        return 0.0
    idx = int((pct / 100) * (len(sorted_values) - 1) + 0.5)
    idx = min(len(sorted_values) - 1, max(0, idx))
    return sorted_values[idx]
