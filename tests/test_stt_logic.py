"""Contract tests: STT admission policy, staleness and language pass-through.

These run without the Whisper model: a fake engine stands in, so the *policy*
(what gets transcribed, what is dropped while a transcript is in flight, what
language is reported) is tested deterministically.
"""
import asyncio

import numpy as np

from aria.audio.stt import SttService
from aria.audio.store import AudioStore
from aria.core.events import Event
from helpers import Collector, attach, make_bus, run

RATE = 16000


class FakeEngine:
    def __init__(self, text="Bonjour, où est la bibliothèque ?", language="fr", prob=0.98):
        self.text = text
        self.language = language
        self.prob = prob
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.text, 0.05, self.language, self.prob

    def warmup(self):
        return 1.0


class BlockingEngine(FakeEngine):
    """Transcription that stays in flight until the test releases it."""

    def __init__(self, release, **kwargs):
        super().__init__(**kwargs)
        self.release = release          # threading.Event
        self.started = False

    def transcribe(self, audio):
        self.calls += 1
        self.started = True
        self.release.wait(5.0)          # runs in a worker thread
        return self.text, 0.05, self.language, self.prob


class FakeStt(SttService):
    """SttService with the model swapped out (no load, no GPU)."""

    def __init__(self, engine=None, retained_s=20.0):
        super().__init__()
        self._fake_engine = engine or FakeEngine()
        self._retained_s = retained_s

    async def init(self) -> None:
        self._store = AudioStore(sample_rate=RATE, retained_s=self._retained_s)
        self._engine = self._fake_engine

    async def _load_engine(self) -> None:
        from aria.core import readiness

        self._engine = self._fake_engine      # already injected
        readiness.mark_ready("stt")


def _cfg(**overrides):
    base = {"language": "auto", "max_stale_s": 2.5, "warmup": False,
            "heartbeat_interval": 0.05}
    base.update(overrides)
    return base


def test_language_is_reported_and_text_is_not_translated():
    async def scenario():
        bus = make_bus()
        await bus.start()
        heard = Collector()
        bus.subscribe("UtteranceHeard", heard, policy="block", maxsize=8)
        svc = attach(FakeStt(), bus, _cfg())
        await svc.start()
        svc._store.append(np.zeros(RATE, dtype=np.float32))          # 1 s of audio

        await svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "u1", "start_index": 0, "end_index": RATE}))
        await heard.wait_for(1, timeout=3)
        payload = heard.events[0].payload
        assert payload["language"] == "fr"
        assert payload["text"].startswith("Bonjour")           # French kept as-is
        assert payload["language_probability"] == 0.98
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_auto_language_passes_none_to_the_engine():
    """'auto' must reach faster-whisper as None (detect), never a forced code."""
    svc = FakeStt()
    from aria.core.config import Config

    svc.attach(make_bus(), Config(_cfg(language="auto"), "<test>"))
    assert svc.config.get("language") == "auto"


def test_stale_turn_is_dropped_instead_of_answered_late():
    async def scenario():
        bus = make_bus()
        await bus.start()
        heard = Collector()
        bus.subscribe("UtteranceHeard", heard, policy="block", maxsize=4)
        svc = attach(FakeStt(), bus, _cfg(max_stale_s=2.0))
        await svc.start()
        svc._store.append(np.zeros(RATE * 6, dtype=np.float32))      # 6 s captured
        total = svc._store.total_samples()

        # turn ended 4 s ago → beyond max_stale_s
        await svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "old", "start_index": 0, "end_index": total - RATE * 4}))
        assert svc.metrics.snapshot()["counters"].get("stt.dropped_stale") == 1
        assert svc._busy is False

        # a fresh turn is accepted
        await svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "new", "start_index": 0, "end_index": total}))
        await heard.wait_for(1, timeout=3)
        assert heard.events[0].payload["utterance_id"] == "new"
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_turn_arriving_while_transcribing_is_dropped_not_queued():
    """The core fix: a turn that arrives mid-transcription must not queue behind
    the one in flight (that backlog is what made ARIA answer old noise and look
    stuck). It is dropped, and the *next* turn after the transcript is accepted."""
    import threading

    release = threading.Event()
    engine = BlockingEngine(release)

    async def scenario():
        bus = make_bus()
        await bus.start()
        heard = Collector()
        bus.subscribe("UtteranceHeard", heard, policy="block", maxsize=8)
        svc = attach(FakeStt(engine=engine), bus, _cfg())
        await svc.start()
        svc._store.append(np.zeros(RATE, dtype=np.float32))
        total = svc._store.total_samples()

        first = asyncio.create_task(svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "u1", "start_index": 0, "end_index": total})))
        for _ in range(200):                       # wait until Whisper is busy
            if engine.started:
                break
            await asyncio.sleep(0.005)
        assert engine.started, "transcription never started"

        for uid in ("u2", "u3"):                   # noise arriving mid-transcript
            await svc._on_turn(Event("TurnCompleted", {
                "utterance_id": uid, "start_index": 0, "end_index": total}))
        counters = svc.metrics.snapshot()["counters"]
        assert counters.get("stt.dropped_busy") == 2
        assert engine.calls == 1                   # nothing queued, nothing extra run

        release.set()
        await first
        await heard.wait_for(1, timeout=3)
        assert [e.payload["utterance_id"] for e in heard.events] == ["u1"]

        # The next genuine turn is transcribed once the service is free again.
        await svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "u4", "start_index": 0, "end_index": total}))
        await heard.wait_for(2, timeout=3)
        assert [e.payload["utterance_id"] for e in heard.events] == ["u1", "u4"]
        assert engine.calls == 2
        assert svc._busy is False
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_turn_that_goes_stale_while_waiting_is_dropped():
    """A turn accepted while the engine warms up must not be answered minutes
    later: staleness is re-checked *after* the readiness wait, not only at the
    door (that is what made replies feel out of sync at start-up)."""
    from aria.core import readiness

    release = asyncio.Event()

    class SlowLoadStt(FakeStt):
        async def _load_engine(self) -> None:
            self._engine = None                # not warm yet
            readiness.mark_not_ready("stt")
            await release.wait()
            self._engine = self._fake_engine
            readiness.mark_ready("stt")

    async def scenario():
        bus = make_bus()
        await bus.start()
        heard = Collector()
        bus.subscribe("UtteranceHeard", heard, policy="block", maxsize=4)
        svc = attach(SlowLoadStt(), bus, _cfg(max_stale_s=1.0))
        await svc.start()
        svc._store.append(np.zeros(RATE, dtype=np.float32))
        total = svc._store.total_samples()
        # Fresh at the door, but the engine is cold → it waits inside the service.
        task = asyncio.create_task(svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "u1", "start_index": 0, "end_index": total})))
        await asyncio.sleep(0.05)
        assert svc._busy is True and heard.events == []      # held, not transcribed

        svc._store.append(np.zeros(RATE * 4, dtype=np.float32))   # 4 s of engine warm-up
        release.set()
        await asyncio.wait_for(task, timeout=5)

        assert svc.metrics.snapshot()["counters"].get("stt.dropped_stale_late") == 1
        assert svc._busy is False
        assert heard.events == []
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_queue_wait_is_measured():
    async def scenario():
        bus = make_bus()
        await bus.start()
        heard = Collector()
        bus.subscribe("UtteranceHeard", heard, policy="block", maxsize=4)
        svc = attach(FakeStt(), bus, _cfg())
        await svc.start()
        svc._store.append(np.zeros(RATE, dtype=np.float32))
        total = svc._store.total_samples()
        await svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "u1", "start_index": 0, "end_index": total}))
        await heard.wait_for(1, timeout=3)
        hist = svc.metrics.snapshot()["histograms"]
        assert hist["stt.queue_wait_s"]["count"] == 1
        await svc.stop()
        await bus.stop()
    run(scenario())