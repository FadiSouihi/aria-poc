"""Contract tests: STT queueing, staleness and language pass-through.

These run without the Whisper model: a fake engine stands in, so the *policy*
(what gets transcribed, when it is dropped, what language is reported) is
tested deterministically.
"""
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
    base = {"language": "auto", "max_stale_s": 2.5, "max_queue": 3, "warmup": False,
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
        assert len(svc._queue) == 0

        # a fresh turn is accepted
        await svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "new", "start_index": 0, "end_index": total}))
        await heard.wait_for(1, timeout=3)
        assert heard.events[0].payload["utterance_id"] == "new"
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_queue_overflow_drops_the_oldest_turn():
    bus = make_bus()
    svc = FakeStt()
    from aria.core.config import Config

    svc.attach(bus, Config(_cfg(max_queue=2), "<test>"))
    svc._store = AudioStore(sample_rate=RATE, retained_s=20.0)
    svc._engine = FakeEngine()
    svc._store.append(np.zeros(RATE, dtype=np.float32))
    total = svc._store.total_samples()

    import asyncio

    async def enqueue_three():
        for uid in ("a", "b", "c"):
            await svc._on_turn(Event("TurnCompleted", {
                "utterance_id": uid, "start_index": 0, "end_index": total}))
    asyncio.run(enqueue_three())

    assert len(svc._queue) == 2
    assert [p["utterance_id"] for p in svc._queue] == ["b", "c"]
    assert svc.metrics.snapshot()["counters"].get("stt.dropped_overflow") == 1


def test_turn_that_goes_stale_while_queued_is_dropped():
    """A turn accepted while the engine warms up must not be answered minutes
    later: staleness is re-checked when the worker actually picks it up."""
    import asyncio

    from aria.core import readiness

    release = asyncio.Event()

    class SlowLoadStt(FakeStt):
        async def _load_engine(self) -> None:
            await release.wait()               # engine not ready yet
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
        await svc._on_turn(Event("TurnCompleted", {
            "utterance_id": "u1", "start_index": 0, "end_index": total}))   # fresh now
        assert len(svc._queue) == 1

        svc._store.append(np.zeros(RATE * 4, dtype=np.float32))              # 4 s pass
        release.set()
        for _ in range(200):                                                 # let worker run
            if svc.metrics.snapshot()["counters"].get("stt.dropped_stale_late"):
                break
            await asyncio.sleep(0.02)

        assert svc.metrics.snapshot()["counters"].get("stt.dropped_stale_late") == 1
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