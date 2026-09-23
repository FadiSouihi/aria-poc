"""Contract tests: VAD state machine (FUNC-14 noise / turn-taking fixes).

Pure probability-driven logic: no microphone, no model, deterministic.
"""
from aria.audio.vad import VadStateMachine


def _machine(**kwargs):
    defaults = dict(threshold=0.5, min_speech_ms=250, min_silence_ms=400,
                    start_frames=2, frame_ms=32, max_utterance_s=15.0)
    defaults.update(kwargs)
    return VadStateMachine(**defaults)


def test_speech_started_after_consecutive_voiced_frames():
    m = _machine()
    assert m.update(0.9) is None          # one voiced frame is not enough
    assert m.update(0.8) == "started"


def test_single_spike_never_starts_speech():
    m = _machine(start_frames=3)
    assert m.update(0.99) is None
    assert m.update(0.1) is None
    assert m.update(0.99) is None
    assert m.speaking is False


def test_speech_ended_requires_min_silence():
    m = _machine(min_silence_ms=400)      # 400 ms / 32 ms ≈ 12 frames
    m.update(0.9)
    assert m.update(0.9) == "started"
    for _ in range(11):
        assert m.update(0.0) is None      # still inside the silence allowance
    assert m.update(0.0) == "ended"


def test_noise_burst_does_not_produce_long_utterance():
    """A click shorter than min_speech_ms must not open a turn."""
    m = _machine(start_frames=4, min_speech_ms=250)
    m.update(0.9)
    m.update(0.9)
    m.update(0.05)                        # burst ends before start_frames
    assert m.speaking is False


def test_hard_timeout_ends_runaway_utterance():
    """Tier 3: a very long speech run is force-ended (FUNC-14 hard pause)."""
    m = _machine(max_utterance_s=1.0, min_silence_ms=100000)
    m.update(0.9)
    assert m.update(0.9) == "started"
    result = None
    for _ in range(100):
        result = m.update(0.9)
        if result:
            break
    assert result == "ended_timeout"
    assert m.speaking is False


def test_probabilities_are_thresholded_not_averaged():
    m = _machine(threshold=0.5)
    m.update(0.51)
    assert m.update(0.51) == "started"
    assert m.speaking is True


def test_reset_abandons_in_flight_utterance():
    m = _machine()
    m.update(0.9)
    m.update(0.9)
    assert m.speaking is True
    m.reset()
    assert m.speaking is False and m.frames == 0


# -- half-duplex guard (echo protection while TTS plays) ------------------------
def test_ducking_window_opens_on_speech_and_covers_the_echo_tail():
    import time

    from aria.audio.vad import VadService
    from aria.core.events import Event
    from helpers import attach, make_bus, run

    async def scenario():
        bus = make_bus()
        await bus.start()
        svc = attach(VadService(), bus, {"duck_while_speaking": True, "duck_margin_s": 0.0,
                                        "post_speech_guard_s": 0.8})
        await svc.init()
        await svc.start()
        assert svc._ducking() is False

        svc._machine.update(0.9)
        svc._machine.update(0.9)                      # utterance in flight
        before = time.monotonic()
        await svc._on_synthesized(Event("SpeechSynthesized", {"audio_s": 5.0}))
        assert svc._ducking() is True
        assert svc._machine.speaking is False         # in-flight speech dropped
        assert svc.metrics.snapshot()["counters"]["vad.ducked_utterances"] == 1
        # The window must outlive the audio by the echo-tail guard, not end with it.
        assert svc._speaking_until - before >= 5.7

        # A barge-in stops playback but the speaker keeps ringing: the mic must
        # stay ducked through the tail, or ARIA immediately re-hears itself.
        await svc._on_barge_in(Event("BargeIn", {}))
        assert svc._ducking() is True
        assert svc._speaking_until >= time.monotonic() + 0.7
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_speech_started_carries_its_loudness():
    """TTS needs the level to tell a person from ARIA's own quieter echo, so the
    level must travel on ``SpeechStarted`` (a missing field would silently make
    every barge-in look genuine)."""
    import numpy as np

    from aria.audio.vad import VadService
    from helpers import Collector, attach, make_bus, run

    class StubStore:
        """Deterministic: exactly one loud 512-sample frame exists."""

        sample_rate = 16000

        def read_since(self, start_index):
            return np.full(max(0, 512 - int(start_index)), 0.25, dtype=np.float32)

        def total_samples(self):
            return 512

    async def scenario():
        bus = make_bus()
        await bus.start()
        seen = Collector()
        bus.subscribe("SpeechStarted", seen, policy="block", maxsize=4)
        svc = attach(VadService(), bus, {"duck_while_speaking": False})
        await svc.init()
        svc._vad = None                     # the level path must not need the model
        await svc.start()                   # start() calls init() → inject the stub after
        svc._store = StubStore()
        svc._cursor = 512
        assert svc._machine.update(0.9) is None      # one voiced frame so far
        await svc._handle(0.9)                       # second → speech started
        await seen.wait_for(1, timeout=2)
        payload = seen.events[0].payload
        assert payload["start_index"] == 0            # one frame back, not two
        assert payload["dbfs"] == -12.0               # 0.25 RMS ≈ -12 dBFS
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_ducking_can_be_disabled_for_headset_use():
    from aria.audio.vad import VadService
    from aria.core.events import Event
    from helpers import attach, make_bus, run

    async def scenario():
        bus = make_bus()
        await bus.start()
        svc = attach(VadService(), bus, {"duck_while_speaking": False})
        await svc.init()
        await svc.start()
        await svc._on_synthesized(Event("SpeechSynthesized", {"audio_s": 5.0}))
        assert svc._ducking() is False                # barge-in stays possible
        await svc.stop()
        await bus.stop()
    run(scenario())