"""Vision resource policy: when and how much to yield, and honest pacing.

Two things are worth locking down here:

1. The "yield while speaking" triggers. An earlier version also yielded on
   ``SpeechStarted`` (the *listening* phase), which throttled detection almost
   continuously and suppressed the engagement signal the voice gate depends on.
2. The detect loop's pacing. It used to sleep the interval *after* each
   inference, so inference time silently added to the period and a 15 Hz loop
   behaved like 4 Hz.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

from aria.core.config import Config
from aria.perception.vision import BUSY_EVENTS, PersonVisionService, busy_seconds_for
from tests.helpers import attach, make_bus, run


def _cfg(**overrides) -> Config:
    return Config(dict(overrides), "<test>")


# -- yield policy --------------------------------------------------------

def test_only_transcription_and_speech_throttle_vision():
    assert set(BUSY_EVENTS) == {"TurnCompleted", "SpeechSynthesized"}
    assert "SpeechStarted" not in BUSY_EVENTS      # listening must not throttle


def test_busy_seconds_are_zero_for_non_triggers():
    assert busy_seconds_for("SpeechStarted", {}, _cfg()) == 0.0
    assert busy_seconds_for("Frame", {}, _cfg()) == 0.0
    assert busy_seconds_for("TrackStates", {}, _cfg()) == 0.0


def test_busy_seconds_follow_the_configured_grace_period():
    assert busy_seconds_for("TurnCompleted", {}, _cfg(busy_after_turn_s=2.5)) == 2.5
    assert busy_seconds_for("TurnCompleted", {}, _cfg()) == 1.5     # default


def test_busy_seconds_match_the_spoken_clause_length():
    payload = {"audio_s": 2.0}
    assert busy_seconds_for("SpeechSynthesized", payload, _cfg()) == 2.0 + 0.15
    # A clause with no duration must not wedge vision at the busy rate.
    assert busy_seconds_for("SpeechSynthesized", {}, _cfg()) == 0.15


def test_mark_busy_takes_the_longest_deadline_and_expires():
    async def scenario():
        svc = attach(PersonVisionService(), make_bus(), {})
        assert svc._busy() is False
        svc._mark_busy(0.05)
        svc._mark_busy(0.01)                      # shorter: must not shrink it
        assert svc._busy() is True
        await asyncio.sleep(0.08)
        assert svc._busy() is False
    run(scenario())


def test_busy_handler_maps_events_to_deadlines():
    async def scenario():
        svc = attach(PersonVisionService(), make_bus(), {"busy_after_turn_s": 0.5})
        await svc._on_busy_event(type("E", (), {"name": "TurnCompleted", "payload": {}})())
        assert svc._busy() is True
        svc._busy_until = 0.0
        await svc._on_busy_event(type("E", (), {"name": "SpeechStarted", "payload": {}})())
        assert svc._busy() is False               # not a trigger
    run(scenario())


# -- pacing --------------------------------------------------------------

class FakeStore:
    """Hands out a fresh frame id on every read."""

    def __init__(self) -> None:
        self.reads = 0

    def latest(self):
        self.reads += 1
        return self.reads, np.zeros((8, 8, 3), dtype=np.uint8)


def test_detect_loop_does_not_let_inference_time_stretch_its_period():
    """20 Hz target with 30 ms inferences must still yield ~20 Hz worth of work.

    The old sleep-after-work loop ran at 1/(50 ms + 30 ms) = 12.5 Hz.
    """
    async def scenario():
        svc = attach(PersonVisionService(), make_bus(), {"detect_hz": 20.0, "busy_detect_hz": 20.0})
        svc._store = FakeStore()
        svc.device_str = "cpu"

        def slow_infer(frame):
            time.sleep(0.03)
            return []

        svc._infer = slow_infer

        async def noop(tracks, fid, latency):
            return None

        svc._process_tracks = noop
        task = asyncio.ensure_future(svc._detect_loop())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        detects = svc.metrics.snapshot()["counters"].get("vision.idle_detects", 0)
        # 0.5 s at 20 Hz ≈ 10; the buggy version managed ~6.
        assert 8 <= detects <= 12, f"expected ~10 detects, got {detects}"
    run(scenario())


def test_detect_loop_uses_the_busy_rate_while_yielding():
    async def scenario():
        svc = attach(PersonVisionService(), make_bus(), {"detect_hz": 50.0, "busy_detect_hz": 10.0})
        svc._store = FakeStore()
        svc.device_str = "cpu"
        svc._infer = lambda frame: []

        async def noop(tracks, fid, latency):
            return None

        svc._process_tracks = noop
        svc._mark_busy(5.0)                       # busy for the whole window
        task = asyncio.ensure_future(svc._detect_loop())
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        counters = svc.metrics.snapshot()["counters"]
        assert counters.get("vision.busy_detects", 0) >= 3
        assert counters.get("vision.idle_detects", 0) == 0
    run(scenario())
