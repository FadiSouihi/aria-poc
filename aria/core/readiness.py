"""Readiness signals — lets a producer wait for a slow consumer to be ready.

Services that load models take seconds to become useful (Whisper: model load +
warmup ≈ 8 s). Anything published before such a service subscribes is simply
lost, so two things must be true:

1. the slow service subscribes *before* its model loads (so nothing is missed);
2. a replay source (a fixture file, or a demo script) can wait until the slow
   service is actually ready before it starts feeding audio.

This is a tiny in-process registry for (2); it deliberately has no bus or
telemetry dependencies so it can be used from any thread or task.
"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, Iterable, List

_ready: Dict[str, float] = {}


def mark_ready(name: str) -> None:
    """Announce that ``name`` can now do its job (e.g. the STT engine is warm)."""
    _ready[name] = time.monotonic()


def mark_not_ready(name: str) -> None:
    _ready.pop(name, None)


def is_ready(name: str) -> bool:
    return name in _ready


def ready_names() -> List[str]:
    return sorted(_ready)


def reset() -> None:
    _ready.clear()


async def wait_ready(names: Iterable[str], timeout: float = 30.0,
                     poll_s: float = 0.05) -> bool:
    """Wait until every name is ready; → True if all ready, False on timeout."""
    wanted = [n for n in names if n]
    if not wanted:
        return True
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if all(is_ready(n) for n in wanted):
            return True
        await asyncio.sleep(poll_s)
    return False