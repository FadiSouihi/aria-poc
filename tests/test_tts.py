"""Contract tests: TTS language-matched voices, clause pipelining, cache, barge-in.

All hermetic: the ``fake`` provider produces silence, so no network, no audio
device and no model are involved.
"""
import asyncio

from aria.audio.tts import TtsService, split_clauses
from aria.core.events import Event
from helpers import Collector, attach, make_bus, run

CFG = {
    "provider": "fake",
    "playback": False,
    "voice": "en-US-AriaNeural",
    "voices": {"fr": "fr-FR-DeniseNeural", "ar": "ar-MA-MounaNeural"},
    "clause_pipelining": True,
    "cache_size": 8,
    "heartbeat_interval": 0.05,
}


class RecordingProvider:
    """Records exactly what the service asked it to speak."""

    name = "recording"

    def __init__(self, fail_languages=()):
        self.calls = []
        self.fail_languages = set(fail_languages)

    def synth(self, text, language="en", voice=None):
        self.calls.append((text, language, voice))
        if language in self.fail_languages:
            raise RuntimeError(f"cannot speak {language}")
        import numpy as np

        return np.zeros(1600, dtype=np.float32), 16000


def test_resolved_voice_reaches_the_provider():
    """Regression: the event said fr-FR-DeniseNeural while edge-tts kept using
    the constructor voice, so ARIA always spoke English."""
    async def scenario():
        bus = make_bus()
        await bus.start()
        said = Collector()
        bus.subscribe("SpeechSynthesized", said, policy="block", maxsize=8)
        svc = attach(TtsService(), bus, CFG)
        provider = RecordingProvider()
        await svc.start()
        svc._provider = provider
        await svc._on_speak(Event("SpeakRequest", {"text": "Bonjour.", "language": "fr"}))
        await said.wait_for(1, timeout=5)
        assert provider.calls == [("Bonjour.", "fr", "fr-FR-DeniseNeural")]
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_fallback_does_not_permanently_replace_the_primary_provider():
    """An offline provider has no Arabic voice: a network blip must not silently
    downgrade every later reply to SAPI."""
    async def scenario():
        bus = make_bus()
        await bus.start()
        svc = attach(TtsService(), bus, CFG)
        primary = RecordingProvider(fail_languages={"fr"})
        fallback = RecordingProvider()
        await svc.start()
        svc._provider, svc._fallback = primary, fallback

        await svc._on_speak(Event("SpeakRequest", {"text": "Bonjour.", "language": "fr"}))
        assert fallback.calls == [("Bonjour.", "fr", "fr-FR-DeniseNeural")]
        assert svc._provider is primary                      # not swapped out
        assert svc.metrics.snapshot()["counters"].get("tts.fallbacks") == 1

        await svc._on_speak(Event("SpeakRequest", {"text": "Salut.", "language": "fr"}))
        assert len(primary.calls) == 1                       # still in cooldown
        svc._primary_blocked_until = 0.0                     # cooldown expires
        await svc._on_speak(Event("SpeakRequest", {"text": "Encore.", "language": "fr"}))
        assert len(primary.calls) == 2                       # primary retried
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_warm_phrases_fill_the_cache_at_startup():
    async def scenario():
        bus = make_bus()
        await bus.start()
        cfg = dict(CFG, warm_phrases={"fr": ["J'ai entendu :"]})
        svc = attach(TtsService(), bus, cfg)
        await svc.start()
        for _ in range(100):                                  # warm task runs in background
            if svc.metrics.snapshot()["histograms"].get("tts.warm_ms", {}).get("count"):
                break
            await asyncio.sleep(0.02)
        assert ("fr-FR-DeniseNeural", "J'ai entendu :") in svc._cache
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_split_clauses_keeps_sentences_whole():
    assert split_clauses("Hello there. How are you?") == ["Hello there.", "How are you?"]
    assert split_clauses("") == []
    assert split_clauses("   ") == []
    assert split_clauses("One long sentence without punctuation") == [
        "One long sentence without punctuation"]


def test_voice_is_chosen_by_language_with_default_fallback():
    bus = make_bus()
    svc = attach(TtsService(), bus, CFG)
    assert svc.voice_for("fr") == "fr-FR-DeniseNeural"
    assert svc.voice_for("fr-FR") == "fr-FR-DeniseNeural"
    assert svc.voice_for("ar") == "ar-MA-MounaNeural"
    assert svc.voice_for("de") == "en-US-AriaNeural"          # no voice configured
    assert svc.voice_for(None) == "en-US-AriaNeural"
    counters = svc.metrics.snapshot()["counters"]
    assert counters.get("tts.no_voice.de") == 1


def test_reply_is_spoken_in_the_detected_language():
    async def scenario():
        bus = make_bus()
        await bus.start()
        said = Collector()
        bus.subscribe("SpeechSynthesized", said, policy="block", maxsize=8)
        svc = attach(TtsService(), bus, CFG)
        await svc.start()

        await svc._on_speak(Event("SpeakRequest", {
            "text": "Bonjour. Comment allez-vous ?", "language": "fr", "utterance_id": "u1"}))
        await said.wait_for(2, timeout=5)
        payloads = [e.payload for e in said.events]
        assert [p["clause_index"] for p in payloads] == [0, 1]      # pipelined per clause
        assert all(p["language"] == "fr" for p in payloads)
        assert all(p["voice"] == "fr-FR-DeniseNeural" for p in payloads)
        assert payloads[0]["full_text"] == "Bonjour. Comment allez-vous ?"
        assert svc.metrics.snapshot()["counters"]["tts.clauses"] == 2
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_repeated_phrase_is_served_from_cache():
    async def scenario():
        bus = make_bus()
        await bus.start()
        said = Collector()
        bus.subscribe("SpeechSynthesized", said, policy="block", maxsize=16)
        svc = attach(TtsService(), bus, CFG)
        await svc.start()
        request = Event("SpeakRequest", {"text": "Welcome to TekUp.", "language": "en"})
        await svc._on_speak(request)
        await said.wait_for(1, timeout=5)
        await svc._on_speak(request)
        await said.wait_for(2, timeout=5)
        counters = svc.metrics.snapshot()["counters"]
        assert counters.get("tts.cache_hits") == 1                  # second time: cached
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_barge_in_stops_playback_and_reports_interruption():
    async def scenario():
        bus = make_bus()
        await bus.start()
        barge = Collector()
        bus.subscribe("BargeIn", barge, policy="block", maxsize=4)
        svc = attach(TtsService(), bus, CFG)
        await svc.start()

        svc._speaking = True                     # as if a reply were playing
        svc._current_text = "Welcome to the university."
        await svc._on_speech_started(Event("SpeechStarted", {"utterance_id": "u9"}))
        await barge.wait_for(1, timeout=3)
        assert barge.events[0].payload["interrupted_text"] == "Welcome to the university."
        assert svc._stop_playback.is_set()
        assert svc.metrics.snapshot()["counters"]["tts.barge_ins"] == 1
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_no_barge_in_when_not_speaking():
    async def scenario():
        bus = make_bus()
        await bus.start()
        barge = Collector()
        bus.subscribe("BargeIn", barge, policy="block", maxsize=4)
        svc = attach(TtsService(), bus, CFG)
        await svc.start()
        await svc._on_speech_started(Event("SpeechStarted", {}))   # idle: nothing happens
        assert barge.events == []
        await svc.stop()
        await bus.stop()
    run(scenario())