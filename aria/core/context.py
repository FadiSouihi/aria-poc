"""Correlation context: session/track/utterance ids bound to the current task.

Every log line and event carries whatever context is currently bound, so a
single line can be traced back to the interaction it belongs to. Binding is
scoped via a context manager and is asyncio-safe (contextvars).
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Dict, Iterator

session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("session_id", default=None)
track_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("track_id", default=None)
utterance_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("utterance_id", default=None)

_VARS: Dict[str, contextvars.ContextVar] = {
    "session_id": session_id,
    "track_id": track_id,
    "utterance_id": utterance_id,
}


@contextlib.contextmanager
def bind(**values: Any) -> Iterator[None]:
    """Bind context values (e.g. ``bind(utterance_id="u7")``) for the block."""
    known = {k: v for k, v in values.items() if k in _VARS}
    unknown = set(values) - set(_VARS)
    if unknown:
        raise KeyError(f"Unknown context keys: {sorted(unknown)}; known: {sorted(_VARS)}")
    tokens = [_VARS[k].set(v) for k, v in known.items()]
    try:
        yield
    finally:
        for var, token in zip((_VARS[k] for k in known), tokens):
            var.reset(token)


def snapshot() -> Dict[str, Any]:
    """Return the currently bound context (only non-None values)."""
    return {k: v.get() for k, v in _VARS.items() if v.get() is not None}
