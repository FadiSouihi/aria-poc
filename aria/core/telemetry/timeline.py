"""JSONL event timeline ("flight recorder") — ROADMAP §9.

Every bus event is tapped into ``logs/timeline_full.jsonl``; a ring buffer of
the last N events lives in memory and is dumped to a timestamped file when an
anomaly event occurs (crash, throttle, tamper, device loss). This is what
makes on-site malfunctions reproducible after the fact.

Two cost controls, because the tap runs inline on the *publisher's* task:

- the file handle is kept open (a 25 s run writes ~1,900 lines; opening the file
  per event meant ~76 opens/second inside the camera/vision/mic tasks);
- ``sample`` can keep 1-in-N of the high-rate metadata events (``Frame``,
  ``AudioChunk``) or drop them entirely (``0``). Fidelity is a config choice:
  test/bench profiles keep everything, the demo profile samples the noise.
"""
from __future__ import annotations

import json
import pathlib
import time
from collections import deque
from typing import Deque, Dict, Iterable, Optional

from aria.core.events import ANOMALY_EVENTS, Event
from aria.core.telemetry.logging import get_logger

_SEQ = 0


def _next_seq() -> int:
    global _SEQ
    _SEQ += 1
    return _SEQ


class TimelineRecorder:
    def __init__(
        self,
        dir_name: str = "logs",
        ring_size: int = 600,
        dump_on: Optional[Iterable[str]] = None,
        full_log_name: str = "timeline_full.jsonl",
        sample: Optional[Dict[str, int]] = None,
        flush_every_s: float = 1.0,
    ) -> None:
        self._dir = pathlib.Path(dir_name)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._full_path = self._dir / full_log_name
        self._ring: Deque[dict] = deque(maxlen=ring_size)
        self._dump_on = frozenset(dump_on) if dump_on is not None else ANOMALY_EVENTS
        self._sample = {str(k): max(0, int(v)) for k, v in dict(sample or {}).items()}
        self._high_rate = frozenset(self._sample)   # names we buffer instead of flushing
        self._counts: Dict[str, int] = {}
        self._flush_every_s = float(flush_every_s)
        self._fh = None
        self._last_flush = time.monotonic()
        self.skipped = 0
        self.log = get_logger("aria.core.timeline")

    # -- capture ---------------------------------------------------------
    def _keep(self, name: str) -> bool:
        """Sampling policy: 1 = every event, N = 1-in-N, 0 = drop."""
        every = self._sample.get(name, 1)
        if every == 0:
            return False
        if every == 1:
            return True
        count = self._counts.get(name, 0) + 1
        self._counts[name] = count
        return count % every == 0

    def record(self, event: Event) -> None:
        """Tap callback: append to the full log and the ring buffer (cheap)."""
        if not self._keep(event.name):
            self.skipped += 1
            return
        entry = {
            "seq": _next_seq(),
            "ts": event.ts,
            "name": event.name,
            "payload": event.payload,
            "context": event.context,
        }
        self._ring.append(entry)
        try:
            if self._fh is None:
                self._fh = self._full_path.open("a", encoding="utf-8")
            self._fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            now = time.monotonic()
            # Only the deliberately high-rate events stay buffered: tools that
            # tail the log while the app is still running need the interesting
            # ones (transcripts, decisions) visible immediately.
            if event.name not in self._high_rate or now - self._last_flush >= self._flush_every_s:
                self.flush()
        except OSError as exc:  # never let telemetry kill the pipeline
            self.log.warning("Timeline write failed", error=str(exc))
            self._fh = None
        if event.name in self._dump_on:
            # Anomaly: make the full log durable up to this event, then dump the
            # ring. Buffering is fine during normal operation; it is not fine when
            # something just went wrong.
            self.flush()
            self.dump(reason=event.name)

    def flush(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._last_flush = time.monotonic()
            except OSError:
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def install(self, bus) -> None:
        """Attach to an EventBus: tap for all events."""
        bus.attach_tap(self.record)

    # -- dumps -----------------------------------------------------------
    def dump(self, reason: str) -> str:
        """Write the ring buffer to a timestamped JSONL file; return path."""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self._dir / f"timeline_{stamp}_{reason}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for entry in self._ring:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self.log.info("Timeline dumped", reason=reason, events=len(self._ring), path=str(path))
        return str(path)

    def __len__(self) -> int:
        return len(self._ring)
