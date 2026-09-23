"""TtsService — speech output, language-aware, with barge-in (Phase 2).

Providers are pluggable by config:
- ``edge``  : edge-tts (cloud, best quality) → MP3 decoded with PyAV;
- ``sapi``  : pyttsx3 → Windows SAPI5, fully offline, writes WAV;
- ``fake``  : silence of the right length (hermetic tests, no device/network).

Three behaviours that matter in conversation:

* **Language-matched voices.** ``SpeakRequest`` carries the language detected by
  STT, and ``voices`` maps a language to a voice (``fr`` → a French voice,
  ``ar`` → an Arabic one). Replying in the speaker's language is a *voice*
  choice, not a translation step: the text spoken is exactly the text received.
* **Clause pipelining.** A reply is split into clauses; clause *n+1* is
  synthesized while clause *n* plays, so first audio starts much sooner than
  waiting for the whole utterance (measured: edge-tts needs ~0.7–1.2 s).
* **Synthesis cache.** Repeated phrases (greetings, directions) replay
  instantly.

Playback runs in a worker thread and is written in chunks so barge-in stops the
audio immediately; each clause publishes ``SpeechSynthesized`` just before it
plays, which also keeps the VAD's half-duplex duck window alive for its exact
duration plus a guard that covers the speaker's echo tail.

Barge-in is *evidence-gated* (``barge_in_min_dbfs``). Without AEC the mic hears
ARIA's own reply; treating that as an interruption stopped playback, reopened the
VAD window and made the robot answer itself in a loop. A barge-in now requires
the loudness of a real person — quiet echo is ignored and the reply is finished.
"""
from __future__ import annotations

import asyncio
import io
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from aria.audio.store import read_wav
from aria.core.events import Event
from aria.core.service import Service

_SCHEMA = {
    "provider": ("auto", (str,)),        # auto | edge | sapi | fake
    "voice": ("en-US-AriaNeural", (str,)),
    "voices": ({}, (dict,)),             # {"fr": "fr-FR-DeniseNeural", "ar": "ar-MA-MounaNeural"}
    "default_language": ("en", (str,)),
    "rate": (1.0, (int, float)),         # 1.0 = provider default speed
    "playback": (True, (bool,)),
    "barge_in": (True, (bool,)),
    # A genuine interruption is loud and close; the robot's own voice coming back
    # through the speaker is much quieter (measured ~-40 dBFS vs -21..-27 dBFS
    # for the person). Anything quieter than this never counts as a barge-in.
    "barge_in_min_dbfs": (-32.0, (int, float)),
    "clause_pipelining": (True, (bool,)),
    "cache_size": (32, (int,)),
    "warm_phrases": ({}, (dict,)),       # {"en": ["I heard you say:"], "fr": [...]}: pre-synthesized at start
    "fallback_cooldown_s": (60.0, (int, float)),
    "playback_latency": ("low", (str,)),
    "fake_words_per_minute": (150, (int,)),
    "heartbeat_interval": (5.0, (int, float)),
}

RATE = 16000


def split_clauses(text: str, max_chars: int = 120) -> list[str]:
    """Split a reply into speakable clauses (pure; used for pipelining)."""
    import re

    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    clauses: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars or not clauses:
            clauses.append(part)
        else:
            clauses.append(part)      # long clause: keep whole, never cut mid-word
    return clauses or [text]


class TtsProvider:
    name = "base"

    def synth(self, text: str, language: str = "en", voice: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """Synthesize ``text``.

        ``language`` is the language detected from the speaker; ``voice`` is the
        voice the service resolved for that language. A provider must honour
        ``voice`` when it can — returning audio in the wrong voice is the bug
        where ARIA "knew" the language but still spoke English.
        """
        raise NotImplementedError


class EdgeTtsProvider(TtsProvider):
    """edge-tts (needs network). MP3 → float32 16 kHz mono via PyAV."""

    name = "edge-tts"

    def __init__(self, voice: str = "en-US-AriaNeural", rate: float = 1.0) -> None:
        self.voice = voice
        self.rate = rate

    def synth(self, text: str, language: str = "en", voice: Optional[str] = None) -> Tuple[np.ndarray, int]:
        import edge_tts

        chosen = str(voice or self.voice)
        percent = int(round((float(self.rate) - 1.0) * 100))
        rate_arg = f"{percent:+d}%"

        async def _collect() -> bytes:
            communicate = edge_tts.Communicate(text, chosen, rate=rate_arg)
            buffer = bytearray()
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    buffer += chunk["data"]
            return bytes(buffer)

        mp3 = asyncio.run(_collect())
        if not mp3:
            raise RuntimeError(f"edge-tts returned no audio for voice {chosen}")
        return decode_audio_bytes(mp3), RATE


class SapiTtsProvider(TtsProvider):
    """Windows SAPI5 via pyttsx3 — offline, no network, no model download.

    Picks an installed voice matching the requested language when one exists
    (Windows ships extra voices with language packs). If the language has no
    installed voice, this **raises** instead of quietly speaking English: the
    offline path must never change what language ARIA answers in.
    """

    name = "sapi-offline"

    def __init__(self, rate: float = 1.0) -> None:
        self.rate = rate

    def synth(self, text: str, language: str = "en", voice: Optional[str] = None) -> Tuple[np.ndarray, int]:
        import pyttsx3

        engine = pyttsx3.init()
        base = 180  # SAPI words-per-minute default
        engine.setProperty("rate", int(base * float(self.rate)))
        voice_id = self._voice_for(engine, language)
        if voice_id is None:
            raise RuntimeError(f"no installed SAPI voice for language {language!r}")
        engine.setProperty("voice", voice_id)
        out = Path(tempfile.gettempdir()) / f"aria_tts_sapi_{abs(hash((text, voice_id))) % 10**8}.wav"
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        audio, rate = read_wav(out)
        if audio.size == 0:
            raise RuntimeError(f"SAPI produced no audio for language {language!r} (voice {voice_id})")
        return audio.astype(np.float32), rate

    @staticmethod
    def _voice_for(engine, language: str) -> Optional[str]:
        want = (language or "")[:2].lower()
        if not want:
            return None
        try:
            for voice in engine.getProperty("voices"):
                langs = getattr(voice, "languages", None) or []
                for entry in langs:
                    code = entry.decode("utf-8", "ignore") if isinstance(entry, bytes) else str(entry)
                    if code.lower().startswith(want):
                        return voice.id
        except Exception:
            return None
        return None


class FakeTtsProvider(TtsProvider):
    """Deterministic silence — lets tests/benches run the TTS path with no
    network and no audio device."""

    name = "fake-silence"

    def __init__(self, words_per_minute: int = 150) -> None:
        self.wpm = max(30, int(words_per_minute))

    def synth(self, text: str, language: str = "en", voice: Optional[str] = None) -> Tuple[np.ndarray, int]:
        words = max(1, len(text.split()))
        seconds = words / (self.wpm / 60.0)
        return np.zeros(int(seconds * RATE), dtype=np.float32), RATE


def decode_audio_bytes(data: bytes, target_rate: int = RATE) -> np.ndarray:
    """Decode compressed audio (mp3/ogg/wav) to float32 mono at target_rate."""
    import av

    frames = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                frames.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):   # flush
            frames.append(resampled.to_ndarray().reshape(-1))
    if not frames:
        raise RuntimeError("no decodable audio frames")
    return np.concatenate(frames).astype(np.float32)


class TtsService(Service):
    name = "audio.tts"
    produces = ("SpeechSynthesized", "BargeIn")
    consumes = ("SpeakRequest", "SpeechStarted")
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._subs = []
        self._provider: Optional[TtsProvider] = None
        self._fallback: Optional[TtsProvider] = None
        self._primary_blocked_until = 0.0     # fallback cooldown, not a permanent downgrade
        self._stop_playback = threading.Event()
        self._speaking = False
        self._current_text = ""
        self._cache: "OrderedDict[tuple, tuple]" = OrderedDict()

    # -- setup -------------------------------------------------------------
    async def init(self) -> None:
        self._provider = self._build_provider(str(self.config.get("provider", "auto")))
        if self._provider.name == "edge-tts":
            self._fallback = SapiTtsProvider(rate=float(self.config.get("rate", 1.0)))
        self.log.info("TTS ready", provider=self._provider.name,
                      fallback=self._fallback.name if self._fallback else None,
                      voices=dict(self.config.get("voices") or {}))

    def _build_provider(self, want: str) -> TtsProvider:
        voice = str(self.config.get("voice", "en-US-AriaNeural"))
        rate = float(self.config.get("rate", 1.0))
        if want == "fake":
            return FakeTtsProvider(int(self.config.get("fake_words_per_minute", 150)))
        if want == "sapi":
            return SapiTtsProvider(rate=rate)
        if want == "edge":
            return EdgeTtsProvider(voice=voice, rate=rate)
        # auto: prefer edge, offline fallback tried at synth time
        try:
            import edge_tts  # noqa: F401

            return EdgeTtsProvider(voice=voice, rate=rate)
        except Exception:
            self.log.warning("edge-tts unavailable; using offline SAPI provider")
            return SapiTtsProvider(rate=rate)

    def voice_for(self, language: Optional[str]) -> str:
        """Voice matching a language code, falling back to the default voice."""
        voices = {str(k).lower()[:2]: str(v) for k, v in (self.config.get("voices") or {}).items()}
        code = (language or "")[:2].lower()
        if code and code in voices:
            return voices[code]
        if code and code not in voices:
            self.metrics.inc(f"tts.no_voice.{code}")
            self.log.info("No configured voice for language; using default",
                          language=code, voice=str(self.config.get("voice")))
        return str(self.config.get("voice", "en-US-AriaNeural"))

    # -- lifecycle ---------------------------------------------------------
    async def on_start(self) -> None:
        self._subs.append(self.bus.subscribe("SpeakRequest", self._on_speak, policy="drop_new", maxsize=4))
        if bool(self.config.get("barge_in", True)):
            self._subs.append(self.bus.subscribe("SpeechStarted", self._on_speech_started,
                                                 policy="drop_new", maxsize=4))
        if self.config.get("warm_phrases"):
            self.spawn(self._warm_cache(), "tts-warm")

    async def _warm_cache(self) -> None:
        """Pre-synthesize canned phrases (reply prefixes) so the first clause of
        the first reply is already in the cache — that removes the pause between
        "I heard you say:" and the rest of the reply."""
        phrases = self.config.get("warm_phrases") or {}
        started = time.perf_counter()
        done = 0
        for language, entries in dict(phrases).items():
            voice = self.voice_for(language)
            for text in ([entries] if isinstance(entries, str) else list(entries or [])):
                text = str(text).strip()
                if not text:
                    continue
                try:
                    await self._synth_clause(text, voice, str(language))
                    done += 1
                except Exception as exc:
                    self.log.warning("Warm phrase failed", language=language, error=str(exc))
        elapsed = (time.perf_counter() - started) * 1000
        self.metrics.observe("tts.warm_ms", elapsed)
        self.log.info("TTS cache warmed", phrases=done, ms=round(elapsed, 1))

    async def on_stop(self) -> None:
        self._stop_playback.set()
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs.clear()

    # -- barge-in ----------------------------------------------------------
    async def _on_speech_started(self, event: Event) -> None:
        if not self._speaking:
            return
        level = event.payload.get("dbfs")
        floor = float(self.config.get("barge_in_min_dbfs", -32.0))
        # Echo rejection: with a speaker and no AEC, ARIA hears its own reply
        # through the mic. That false "interruption" used to stop playback, which
        # then reopened the VAD's duck window, so the robot transcribed its own
        # words and replied to itself — an endless self-conversation. A barge-in
        # now needs the loudness of a real person; anything at or below the floor
        # is ignored (playback continues, the reply is finished).
        if level is not None and float(level) <= floor:
            self.metrics.inc("tts.barge_in_ignored_quiet")
            self.log.info("Ignoring quiet barge-in (own speaker echo)",
                          dbfs=round(float(level), 1), floor_dbfs=floor,
                          interrupted=self._current_text[:60])
            return
        self._stop_playback.set()
        self.metrics.inc("tts.barge_ins")
        self.log.info("Barge-in: playback stopped", interrupted=self._current_text[:60],
                      dbfs=None if level is None else round(float(level), 1))
        await self.bus.publish(Event("BargeIn", {
            "utterance_id": event.payload.get("utterance_id"),
            "interrupted_text": self._current_text,
            "dbfs": level,
        }))

    # -- synthesis + playback ----------------------------------------------
    def _cached(self, key: tuple) -> Optional[tuple]:
        hit = self._cache.get(key)
        if hit is None:
            return None
        self._cache.move_to_end(key)
        self.metrics.inc("tts.cache_hits")
        return hit

    def _store(self, key: tuple, value: tuple) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        limit = max(0, int(self.config.get("cache_size", 32)))
        while len(self._cache) > limit:
            self._cache.popitem(last=False)

    async def _synth_clause(self, text: str, voice: str, language: str) -> tuple:
        key = (voice, text)
        hit = self._cached(key)
        if hit is not None:
            return hit
        started = time.perf_counter()
        primary = self._provider
        # The fallback is a cooldown, not a one-way door: an offline provider has
        # no French/Arabic voice, so a network blip must not permanently change
        # the language ARIA speaks in.
        use_primary = primary is not None and time.monotonic() >= self._primary_blocked_until
        provider = primary if use_primary else (self._fallback or primary)
        try:
            if provider is None:
                raise RuntimeError("no TTS provider available")
            # ``voice`` (resolved from the language) must reach the provider:
            # otherwise ARIA reports the right voice but speaks the default one.
            audio, rate = await asyncio.to_thread(provider.synth, text, language, voice)
        except Exception as exc:
            if self._fallback is None or provider is self._fallback:
                self.metrics.inc(f"tts.failed.{str(language)[:2].lower() or 'xx'}")
                self.log.warning("No voice available for this reply",
                                 language=language, voice=voice, error=str(exc))
                raise
            self.log.warning("Primary TTS failed; using offline fallback", error=str(exc))
            self._primary_blocked_until = time.monotonic() + float(self.config.get("fallback_cooldown_s", 60.0))
            self.metrics.inc("tts.fallbacks")
            audio, rate = await asyncio.to_thread(self._fallback.synth, text, language, voice)
        self.metrics.observe("tts.synth_ms", (time.perf_counter() - started) * 1000)
        self._store(key, (audio, rate))
        return audio, rate

    async def _on_speak(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            return
        language = str(event.payload.get("language") or self.config.get("default_language", "en"))
        voice = self.voice_for(language)
        clauses = split_clauses(text) if bool(self.config.get("clause_pipelining", True)) else [text]
        playback = bool(self.config.get("playback", True))

        started = time.perf_counter()
        self._stop_playback.clear()
        self._speaking = True
        self._current_text = text
        total_audio = 0.0
        first_audio_ms: Optional[float] = None
        interrupted = False
        # ONE output stream for the whole reply: reopening it per clause left an
        # audible gap between "I heard you say:" and the words that followed.
        stream = None
        stream_rate = None
        write_finished: Optional[float] = None
        try:
            pending = asyncio.create_task(self._synth_clause(clauses[0], voice, language))
            for index, clause in enumerate(clauses):
                audio, rate = await pending
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - started) * 1000
                    self.metrics.observe("tts.first_audio_ms", first_audio_ms)
                duration = audio.size / float(rate or RATE)
                total_audio += duration
                self.metrics.inc("tts.clauses")
                await self.bus.publish(Event("SpeechSynthesized", {
                    "text": clause,
                    "full_text": text,
                    "utterance_id": event.payload.get("utterance_id"),
                    "language": language,
                    "voice": voice,
                    "provider": self._provider.name if self._provider else "none",
                    "clause_index": index,
                    "clause_count": len(clauses),
                    "audio_s": round(duration, 3),
                    "first_audio_ms": round(first_audio_ms, 1) if index == 0 else None,
                }))
                if playback and duration > 0:
                    next_task = (asyncio.create_task(
                        self._synth_clause(clauses[index + 1], voice, language))
                        if index + 1 < len(clauses) else None)
                    # Dead air between clauses (the stream ran dry) — this is the
                    # "pause in the middle of the reply" the user hears. Must stay ~0.
                    if write_finished is not None:
                        self.metrics.observe("tts.clause_gap_ms",
                                             (time.perf_counter() - write_finished) * 1000)
                    if stream is None or int(rate) != stream_rate:
                        if stream is not None:
                            await asyncio.to_thread(self._close_stream, stream)
                        stream = await asyncio.to_thread(self._open_stream, int(rate or RATE))
                        stream_rate = int(rate)
                    interrupted = await asyncio.to_thread(self._play_blocking, audio, rate, stream)
                    write_finished = time.perf_counter()
                    if next_task is not None:
                        pending = next_task
                    if interrupted:
                        self.log.info("Playback interrupted by barge-in")
                        break
        finally:
            if stream is not None:
                await asyncio.to_thread(self._close_stream, stream)
            self._speaking = False
            self._current_text = ""
        if first_audio_ms is not None:
            gaps = self.metrics.snapshot()["histograms"].get("tts.clause_gap_ms", {})
            self.log.info("Speech synthesized", provider=self._provider.name, language=language,
                          voice=voice, clauses=len(clauses), audio_s=round(total_audio, 2),
                          first_audio_ms=round(first_audio_ms, 1),
                          clause_gap_ms=gaps.get("max"),
                          total_ms=round((time.perf_counter() - started) * 1000, 1))

    def _open_stream(self, rate: int):
        import sounddevice as sd

        stream = sd.OutputStream(samplerate=int(rate), channels=1, dtype="float32",
                                 blocksize=0,
                                 latency=str(self.config.get("playback_latency", "low")))
        stream.start()
        return stream

    @staticmethod
    def _close_stream(stream) -> None:
        try:
            stream.stop()
        finally:
            stream.close()

    def _play_blocking(self, audio: np.ndarray, rate: int, stream=None) -> bool:
        """Write one clause; returns True if a barge-in stopped playback.

        ``stream`` is reused across the clauses of one reply (opening a stream
        costs ~100-300 ms on WASAPI, which was audible as a gap mid-reply).
        """
        owns_stream = stream is None
        if owns_stream:
            stream = self._open_stream(int(rate))
        step = 1024
        try:
            for i in range(0, audio.size, step):
                if self._stop_playback.is_set():
                    return True
                stream.write(np.asarray(audio[i: i + step], dtype=np.float32).reshape(-1, 1))
            return False
        finally:
            if owns_stream:
                self._close_stream(stream)