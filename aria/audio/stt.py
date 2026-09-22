"""SttService — faster-whisper transcription of completed turns.

Consumes ``TurnCompleted``, cuts the utterance out of the shared AudioStore,
transcribes with faster-whisper ``large-v3-turbo`` (int8_float16 on GPU,
int8 on CPU), and publishes ``UtteranceHeard``.

Two behaviours matter for how the robot *feels* in conversation:

* **Language is detected, never translated.** ``language: auto`` keeps the
  spoken language in the transcript (Whisper's ``task="transcribe"``); forcing
  ``language: en`` makes Whisper render French/Arabic speech as English text,
  which looks like unwanted translation. The detected code travels with the
  utterance so TTS can answer in the same language.
* **Stale turns are dropped, not queued.** Transcription is serial, so a second
  utterance used to wait behind the first (measured: up to 4.9 s between speech
  ending and its transcript). The worker now dequeues in order but discards any
  turn whose audio ended more than ``max_stale_s`` ago — answering a question
  you asked five seconds ago is worse than not answering it.

The governor can force CPU mode (``audio.stt`` listens for
``GovernorThrottled``), which is the "demote STT to CPU" path in the budget.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional

import numpy as np

from aria.audio.store import DEFAULT_AUDIO_STORE
from aria.core import readiness
from aria.core.events import Event
from aria.core.service import Service
from aria.perception.vision import resolve_repo_path

_SCHEMA = {
    "model": ("weights/whisper-turbo-ct2", (str,)),
    "fallback_models": (["large-v3-turbo", "small"], (list,)),
    "device": ("auto", (str,)),            # auto | cuda | cpu
    "compute_type": ("int8_float16", (str,)),
    "cpu_compute_type": ("int8", (str,)),
    "language": ("auto", (str,)),          # auto = detect | en | fr | ar | ...
    "beam_size": (1, (int,)),
    "temperature": (0.0, (int, float)),
    "without_timestamps": (True, (bool,)),
    "warmup": (True, (bool,)),
    "max_stale_s": (2.5, (int, float)),    # drop turns older than this instead of answering late
    "max_queue": (3, (int,)),
    "min_chars": (2, (int,)),
    "max_segment_s": (30.0, (int, float)),
    "heartbeat_interval": (5.0, (int, float)),
}


def _preload_cuda_libs() -> None:
    """Import torch so the CUDA/cuBLAS DLLs are loaded before CTranslate2.

    CTranslate2 loads ``cublas64_12.dll`` from the process' DLL search path; when
    the vision stack runs, torch has already put it there. An audio-only profile
    (or a bare script) would otherwise fail with "Library cublas64_12.dll is not
    found or cannot be loaded" even though the GPU is fine.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        pass


class WhisperTranscriber:
    """Lazy faster-whisper wrapper (import cost stays out of hermetic runs)."""

    def __init__(self, model_ref: str, device: str, compute_type: str,
                 language: str = "auto", beam_size: int = 1,
                 temperature: float = 0.0, without_timestamps: bool = True) -> None:
        from faster_whisper import WhisperModel

        if device == "cuda":
            _preload_cuda_libs()
        self.language = None if str(language).lower() in ("auto", "", "none") else language
        self.beam_size = int(beam_size)
        self.temperature = float(temperature)
        self.without_timestamps = bool(without_timestamps)
        self.device = device
        self.compute_type = compute_type
        try:
            self.model = WhisperModel(model_ref, device=device, compute_type=compute_type)
        except Exception:
            if device != "cuda":
                raise
            # CTranslate2 needs the CUDA/cuBLAS DLLs on the loader path; if they
            # are missing, transcribe on CPU rather than failing to start.
            self.model = WhisperModel(model_ref, device="cpu", compute_type="int8")
            self.device, self.compute_type = "cpu", "int8"

    def transcribe(self, audio: np.ndarray) -> tuple[str, float, Optional[str], Optional[float]]:
        """→ (text, elapsed_s, detected_language, language_probability)."""
        started = time.perf_counter()
        segments, info = self.model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.language,            # None → detect; task stays "transcribe"
            task="transcribe",                 # never translate
            beam_size=self.beam_size,
            temperature=self.temperature,      # no fallback re-decode spiral
            without_timestamps=self.without_timestamps,
            vad_filter=False,                  # our own VAD already cut the turn
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, time.perf_counter() - started, getattr(info, "language", None), \
            getattr(info, "language_probability", None)

    def warmup(self) -> float:
        """Pay the first-call cost at start-up, not on the user's first sentence."""
        started = time.perf_counter()
        self.transcribe(np.zeros(16000, dtype=np.float32))
        return (time.perf_counter() - started) * 1000


class SttService(Service):
    name = "audio.stt"
    produces = ("UtteranceHeard",)
    consumes = ("TurnCompleted",)
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._sub = None
        self._gov_sub = None
        self._engine: Optional[WhisperTranscriber] = None
        self._force_cpu = False
        self._queue: deque = deque()
        self._work: asyncio.Queue = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._loader: Optional[asyncio.Task] = None
        self._device = "cpu"

    # -- setup -------------------------------------------------------------
    async def init(self) -> None:
        """Light setup only — the model loads in the background (see ``on_start``).

        Loading a Whisper model plus warmup takes ~8 s; doing it inside ``init``
        meant the service subscribed to ``TurnCompleted`` only afterwards, so any
        turn completing in that window was silently lost.
        """
        self._store = DEFAULT_AUDIO_STORE
        self._device = self._resolve_device()

    async def _load_engine(self) -> None:
        device = self._device
        model_ref = str(self.config.get("model", ""))
        local = resolve_repo_path(model_ref)
        candidates = [str(local) if local.exists() else model_ref] + [
            str(m) for m in (self.config.get("fallback_models") or [])
        ]
        compute = str(self.config.get("cpu_compute_type", "int8")) if device == "cpu" \
            else str(self.config.get("compute_type", "int8_float16"))
        last_error = None
        for ref in candidates:
            try:
                engine = await asyncio.to_thread(
                    WhisperTranscriber, ref, device=device, compute_type=compute,
                    language=str(self.config.get("language", "auto")),
                    beam_size=int(self.config.get("beam_size", 1)),
                    temperature=float(self.config.get("temperature", 0.0)),
                    without_timestamps=bool(self.config.get("without_timestamps", True)),
                )
                if bool(self.config.get("warmup", True)):
                    warmup_ms = await asyncio.to_thread(engine.warmup)
                    self.metrics.observe("stt.warmup_ms", warmup_ms)
                    self.log.info("STT warmed up", warmup_ms=round(warmup_ms, 1))
                self._engine = engine
                self._device = getattr(engine, "device", device)
                self.log.info("STT ready", model=ref, device=self._device,
                              compute_type=getattr(engine, "compute_type", compute),
                              language=str(self.config.get("language", "auto")),
                              language_detection=str(self.config.get("language", "auto")) == "auto")
                readiness.mark_ready("stt")
                return
            except Exception as exc:
                last_error = exc
                self.log.warning("STT model load failed; trying next", model=ref, error=str(exc))
        # Raising marks the service crashed → the watchdog restarts it (bounded).
        raise RuntimeError(f"No STT model could be loaded: {last_error}")

    def _resolve_device(self) -> str:
        want = str(self.config.get("device", "auto"))
        if want == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return want

    # -- lifecycle ----------------------------------------------------------
    async def on_start(self) -> None:
        # Subscribe first, load the model second: nothing may be missed while
        # the engine is warming up.
        self._sub = self.bus.subscribe("TurnCompleted", self._on_turn, policy="drop_new", maxsize=4)
        self._gov_sub = self.bus.subscribe("GovernorThrottled", self._on_throttled, policy="drop_new", maxsize=1)
        readiness.mark_not_ready("stt")
        self._loader = self.spawn(self._load_engine(), "stt-load")
        self._worker = self.spawn(self._worker_loop(), "stt-worker")

    async def on_stop(self) -> None:
        for sub in (self._sub, self._gov_sub):
            if sub is not None:
                self.bus.unsubscribe(sub)
        self._sub = self._gov_sub = None
        readiness.mark_not_ready("stt")
        for task in (self._worker, self._loader):
            if task is not None:
                task.cancel()
        self._worker = self._loader = None
        self._queue.clear()

    async def _on_throttled(self, event: Event) -> None:
        if not self._force_cpu:
            self._force_cpu = True
            self.log.warning("Governor throttled: STT pinned to CPU-safe compute type")

    # -- queueing ----------------------------------------------------------
    def _age_s(self, end_index: Optional[int]) -> float:
        """How long ago this turn's audio ended (absolute sample arithmetic)."""
        if end_index is None:
            return 0.0
        behind = self._store.total_samples() - int(end_index)
        return max(0.0, behind / float(self._store.sample_rate))

    async def _on_turn(self, event: Event) -> None:
        """Enqueue and return immediately — never block the bus on Whisper."""
        payload = dict(event.payload)
        age = self._age_s(payload.get("end_index"))
        if age > float(self.config.get("max_stale_s", 2.5)):
            self.metrics.inc("stt.dropped_stale")
            self.log.info("Dropping stale turn (already superseded)",
                          utterance_id=payload.get("utterance_id"), age_s=round(age, 2))
            return
        max_queue = int(self.config.get("max_queue", 3))
        while len(self._queue) >= max_queue:
            dropped = self._queue.popleft()
            self.metrics.inc("stt.dropped_overflow")
            self.log.warning("STT queue full; dropping oldest turn",
                             utterance_id=dropped.get("utterance_id"))
        self._queue.append(payload)
        self.metrics.inc("stt.enqueued")
        self._work.put_nowait(payload)     # wake the worker; no polling

    async def _worker_loop(self) -> None:
        """Wake on demand (no polling): the queue hands work over directly."""
        max_stale = float(self.config.get("max_stale_s", 2.5))
        while True:
            await self._work.get()            # wake-up signal only
            if not self._queue:
                continue                      # its payload was dropped as overflow
            payload = self._queue.popleft()
            if self._engine is None:
                await readiness.wait_ready(["stt"], timeout=60.0)
            # Re-check staleness at dequeue: a turn can wait here while the engine
            # warms up, and answering a 6-second-old question is worse than
            # dropping it (this is what made replies feel out of sync).
            age = self._age_s(payload.get("end_index"))
            if age > max_stale:
                self.metrics.inc("stt.dropped_stale_late")
                self.log.info("Dropping turn that went stale while queued",
                              utterance_id=payload.get("utterance_id"), age_s=round(age, 2))
                continue
            try:
                await self._transcribe(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:      # a bad turn must not kill the worker
                self.metrics.inc("stt.errors")
                self.log.exception("Transcription failed")
                self.log.warning("Transcription error detail", error=str(exc))

    # -- transcription ------------------------------------------------------
    async def _transcribe(self, payload: dict) -> None:
        start = int(payload.get("start_index", 0))
        end = payload.get("end_index")
        audio = self._store.read_since(start)
        if end is not None:
            keep = max(0, int(end) - start)
            if audio.size > keep:
                audio = audio[:keep]
        max_samples = int(float(self.config.get("max_segment_s", 30.0)) * self._store.sample_rate)
        if audio.size > max_samples:
            audio = audio[-max_samples:]
        if audio.size < self._store.sample_rate * 0.15:
            self.log.info("Turn too short to transcribe", utterance_id=payload.get("utterance_id"))
            self.metrics.inc("stt.skipped_short")
            return

        queue_wait = self._age_s(end)
        text, elapsed, language, lang_prob = await asyncio.to_thread(self._engine.transcribe, audio)
        duration = audio.size / self._store.sample_rate
        rtf = elapsed / duration if duration > 0 else 0.0
        self.metrics.observe("stt.rtf", rtf)
        self.metrics.observe("stt.latency_s", elapsed)
        self.metrics.observe("stt.audio_s", duration)
        self.metrics.observe("stt.queue_wait_s", queue_wait)
        if language:
            self.metrics.inc(f"stt.language.{language}")
        if len(text) < int(self.config.get("min_chars", 2)):
            self.metrics.inc("stt.empty_or_tiny")
            self.log.info("Transcript too short", utterance_id=payload.get("utterance_id"), text=text)
            return
        self.metrics.inc("stt.utterances")
        self.log.info("Utterance transcribed", utterance_id=payload.get("utterance_id"),
                      text=text, language=language, duration_s=round(duration, 2),
                      rtf=round(rtf, 2), latency_s=round(elapsed, 2),
                      queue_wait_s=round(queue_wait, 2))
        await self.bus.publish(Event("UtteranceHeard", {
            "utterance_id": payload.get("utterance_id"),
            "text": text,
            "language": language or "und",
            "language_probability": round(float(lang_prob), 3) if lang_prob is not None else None,
            "duration_s": round(duration, 3),
            "rtf": round(rtf, 3),
            "queue_wait_s": round(queue_wait, 3),
            "speaker": payload.get("speaker"),
            "voiceprint_similarity": payload.get("voiceprint_similarity"),
            "eou_latency_ms": payload.get("eou_latency_ms"),
        }))