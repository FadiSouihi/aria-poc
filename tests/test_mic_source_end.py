"""Replay sources must announce when they finish.

Why it matters: a benchmark or regression run replays a WAV fixture and has to
know when the audio has been *processed*. A fixed sleep cannot know that — STT
readiness alone takes 12-15 s — so `tools/bench_audio.py` used to shut the app
down before the turn completed and reported "no response" for every fixture.
"""
from __future__ import annotations

import asyncio
import pathlib
import threading
import time

import numpy as np

from aria.audio.mic import MicService
from aria.audio.store import DEFAULT_AUDIO_STORE
from tests.helpers import Collector, attach, make_bus, run

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "short_click.wav"


def _service(bus, **config) -> MicService:
    cfg = {"source": "file", "path": str(FIXTURE), "loop": False, "rate": 16000, "block_ms": 32}
    cfg.update(config)
    svc = attach(MicService(), bus, cfg)
    svc._store = DEFAULT_AUDIO_STORE
    svc._block = 512
    return svc


def test_finished_replay_publishes_the_end_event_once():
    async def scenario():
        bus = make_bus()
        svc = _service(bus)
        svc._store.append(np.zeros(512, dtype=np.float32))
        svc._source_finished = True          # what the grabber thread sets at EOF
        collector = Collector()
        bus.subscribe("AudioSourceFinished", collector, policy="drop_new", maxsize=1)
        await bus.start()
        task = asyncio.ensure_future(svc._publish_loop())
        await collector.wait_for(1, timeout=2.0)
        await asyncio.sleep(0.1)             # it must not repeat
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert len(collector.events) == 1
        assert collector.events[0].payload["source"] == "file"
    run(scenario())


def test_end_signal_can_be_disabled():
    async def scenario():
        bus = make_bus()
        svc = _service(bus, signal_end=False)
        svc._store.append(np.zeros(512, dtype=np.float32))
        svc._source_finished = True
        collector = Collector()
        bus.subscribe("AudioSourceFinished", collector, policy="drop_new", maxsize=1)
        await bus.start()
        task = asyncio.ensure_future(svc._publish_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert collector.events == []
    run(scenario())


def test_grab_loop_flags_a_non_looping_fixture_and_not_a_looping_one():
    """The flag is set at real EOF only when the fixture is not looping."""
    bus = make_bus()
    one_shot = _service(bus)
    thread = threading.Thread(target=one_shot._grab_loop, daemon=True)
    thread.start()
    thread.join(timeout=10.0)
    assert one_shot._source_finished is True

    looping = _service(bus, loop=True)
    loop_thread = threading.Thread(target=looping._grab_loop, daemon=True)
    loop_thread.start()
    time.sleep(0.3)
    assert looping._source_finished is False
    looping._stop_evt.set()
    loop_thread.join(timeout=3.0)


def test_mic_declares_the_event_it_produces():
    assert "AudioSourceFinished" in MicService.produces
