"""VadService — Silero VAD v5 with hot-reconnect-safe streaming.

Tier 1 of the three-tier end-of-turn design (ROADMAP §5.2): speech start and
the *candidate* end of speech. The decision that a turn is actually finished
belongs to ``turn.py`` (semantic model) with this service's hard-timeout as
the last resort.

The gating logic is a pure class (``VadStateMachine``) driven by probabilities,
so the FUNC-14 noise/overheard-speech behaviour is unit-testable without a
microphone or the model.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import numpy as np

from aria.audio.store import DEFAULT_AUDIO_STORE, dbfs
from aria.core.events import Event
from aria.core.service import Service

FRAME = 512  # Silero v5 @16 kHz = 32 ms exactly

_SCHEMA = {
    "model": ("weights/silero_vad.onnx", (str,)),
    "duck_while_speaking": (True, (bool,)),
    "duck_margin_s": (0.3, (int, float)),
    "threshold": (0.5, (int, float)),
    "min_speech_ms": (250, (int,)),
    "min_silence_ms": (400, (int,)),
    "start_frames": (2, (int,)),
    "max_utterance_s": (15.0, (int, float)),
    "rate": (16000, (int,)),
    "ort_threads": (1, (int,)),        # tiny model called ~31x/s: one thread, no spin-wait
    "ort_spinning": (False, (bool,)),
    "heartbeat_interval": (5.0, (int, float)),
}


class VadStateMachine:
    """Pure streaming VAD state: probabilities in, speech events out."""

    def __init__(self, threshold: float = 0.5, min_speech_ms: int = 250,
                 min_silence_ms: int = 400, start_frames: int = 2,
                 frame_ms: int = 32, max_utterance_s: float = 15.0) -> None:
        self.threshold = float(threshold)
        self.start_frames = int(start_frames)
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.min_silence_frames = max(1, int(min_silence_ms / frame_ms))
        self.max_frames = max(1, int(max_utterance_s * 1000 / frame_ms))
        self.speaking = False
        self.voiced = 0
        self.silence = 0
        self.frames = 0

    def update(self, prob: float) -> Optional[str]:
        """Feed one frame probability → 'started' | 'ended' | None."""
        if not self.speaking:
            if prob >= self.threshold:
                self.voiced += 1
                if self.voiced >= self.start_frames:
                    self.speaking = True
                    self.frames = self.voiced
                    self.voiced = 0
                    self.silence = 0
                    return "started"
            else:
                self.voiced = 0
            return None

        self.frames += 1
        if prob >= self.threshold:
            self.silence = 0
        else:
            self.silence += 1
            if self.silence >= self.min_silence_frames:
                self.speaking = False
                self.silence = 0
                return "ended"
        if self.frames >= self.max_frames:      # tier 3: hard pause guard
            self.speaking = False
            self.silence = 0
            return "ended_timeout"
        return None

    def speech_duration_s(self, frame_ms: int = 32) -> float:
        return self.frames * frame_ms / 1000.0

    def reset(self) -> None:
        """Abandon any in-flight utterance (playback started / stream reset)."""
        self.speaking = False
        self.voiced = 0
        self.silence = 0
        self.frames = 0


class SileroVadModel:
    """Silero VAD v5 driven directly through onnxruntime.

    No torch and no torchaudio: the ONNX graph is called with its own state
    tensor, which keeps the VAD path light (a few MB) and independent of the
    PyTorch stack — deliberate, since torchaudio has no build matching the
    torch version the vision stack is pinned to.
    """

    def __init__(self, model_path: str, rate: int = 16000, frame: int = FRAME,
                 threads: Optional[int] = None, spinning: Optional[bool] = None) -> None:
        import onnxruntime as ort

        from aria.core.onnx import make_session

        ort.set_default_logger_severity(3)
        # VAD is called ~31x/second on a tiny graph: one thread, no spin-wait.
        # ORT's defaults (one thread per core + spinning) made this the single
        # largest CPU consumer in the app for microseconds of real work.
        self.session = make_session(str(model_path), kind="vad", threads=threads, spinning=spinning)
        names = {i.name for i in self.session.get_inputs()}
        self.v5 = "state" in names
        if not self.v5 and not {"h", "c"} <= names:
            raise RuntimeError(f"Unrecognised Silero VAD graph inputs: {sorted(names)}")
        self.rate = int(rate)
        self.frame = int(frame)
        self._state = None
        self._context = None

    def reset(self) -> None:
        self._state = None
        self._context = None

    def prob(self, frame: np.ndarray) -> float:
        samples = np.asarray(frame, dtype=np.float32).reshape(1, -1)
        if self.v5:
            # v5 expects 64 samples of context prepended to every 512-sample
            # frame; without it the graph silently returns ~0 probability.
            if self._context is None:
                self._context = np.zeros((1, 64), dtype=np.float32)
            if self._state is None:
                self._state = np.zeros((2, 1, 128), dtype=np.float32)
            padded = np.concatenate([self._context, samples], axis=1)
            out, self._state = self.session.run(None, {
                "input": padded, "state": self._state,
                "sr": np.array(self.rate, dtype=np.int64),
            })
            self._context = padded[:, -64:]
        else:  # v4 fallback (h/c state)
            if self._state is None:
                self._state = (np.zeros((2, 1, 64), dtype=np.float32),
                               np.zeros((2, 1, 64), dtype=np.float32))
            hidden, cell = self._state
            out, hidden, cell = self.session.run(None, {
                "input": samples, "h": hidden, "c": cell,
                "sr": np.array(self.rate, dtype=np.int64),
            })
            self._state = (hidden, cell)
        return float(np.asarray(out).reshape(-1)[0])


class VadService(Service):
    name = "audio.vad"
    produces = ("SpeechStarted", "SpeechEnded")
    consumes = ("AudioChunk",)
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._sub = None
        self._subs = []
        self._cursor = 0            # absolute sample index already processed
        self._vad = None
        self._machine: Optional[VadStateMachine] = None
        self._utt_id: Optional[str] = None
        self._start_index = 0
        self._speaking_until = 0.0  # half-duplex window while TTS plays

    async def init(self) -> None:
        from aria.perception.vision import resolve_repo_path

        self._store = DEFAULT_AUDIO_STORE
        model_path = resolve_repo_path(str(self.config.get("model", "weights/silero_vad.onnx")))
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
        self._vad = SileroVadModel(str(model_path), rate=int(self.config.get("rate", 16000)),
                                   threads=self.config.get("ort_threads"),
                                   spinning=self.config.get("ort_spinning"))
        self._machine = VadStateMachine(
            threshold=float(self.config.get("threshold", 0.5)),
            min_speech_ms=int(self.config.get("min_speech_ms", 250)),
            min_silence_ms=int(self.config.get("min_silence_ms", 400)),
            start_frames=int(self.config.get("start_frames", 2)),
            max_utterance_s=float(self.config.get("max_utterance_s", 15.0)),
        )
        self.log.info("Silero VAD ready", provider="silero-vad v5 (onnxruntime)", graph="v5" if self._vad.v5 else "v4")

    async def on_start(self) -> None:
        self._sub = self.bus.subscribe("AudioChunk", self._on_chunk, policy="drop_oldest", maxsize=4)
        if bool(self.config.get("duck_while_speaking", True)):
            # Half-duplex guard: with a speaker + mic, ARIA hears itself. Ducking
            # the mic for the playback duration stops self-triggered barge-in.
            # Set false (headset / no echo path) to allow true barge-in.
            self._subs.append(self.bus.subscribe("SpeechSynthesized", self._on_synthesized,
                                                 policy="drop_new", maxsize=4))
            self._subs.append(self.bus.subscribe("BargeIn", self._on_barge_in,
                                                 policy="drop_new", maxsize=4))
        self.spawn(self._scan_loop(), "scan")

    async def on_stop(self) -> None:
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs.clear()
        if self._sub is not None:
            self.bus.unsubscribe(self._sub)
            self._sub = None

    async def _on_synthesized(self, event: Event) -> None:
        # TTS publishes once per clause just before that clause plays, so the
        # window is extended (never shortened) to cover the whole reply.
        margin = float(self.config.get("duck_margin_s", 0.3))
        until = time.monotonic() + float(event.payload.get("audio_s", 0.0)) + margin
        if until > self._speaking_until:
            self._speaking_until = until
        if self._machine is not None and self._machine.speaking:
            # Drop the utterance that was in flight when playback started.
            self._machine.reset()
            self._utt_id = None
            self.metrics.inc("vad.ducked_utterances")
        self.metrics.inc("vad.duck_windows")

    async def _on_barge_in(self, event: Event) -> None:
        self._speaking_until = 0.0

    def _ducking(self) -> bool:
        return bool(self.config.get("duck_while_speaking", True)) and time.monotonic() < self._speaking_until

    async def _on_chunk(self, event: Event) -> None:
        self.metrics.inc("vad.chunks")

    async def _scan_loop(self) -> None:
        """Walk every captured frame in order, exactly once."""
        period = FRAME / float(self.config.get("rate", 16000))
        while True:
            await asyncio.sleep(period)
            available = self._store.total_samples() - self._cursor
            frames = available // FRAME
            if frames <= 0:
                continue
            audio = self._store.read_since(self._cursor)
            if audio.size < FRAME:
                continue
            if self._ducking():
                # Skip analysis entirely: keep the cursor moving, ignore audio.
                self._cursor += frames * FRAME
                self.metrics.inc("vad.ducked_frames", frames)
                continue
            for i in range(frames):
                frame = audio[i * FRAME: (i + 1) * FRAME]
                if frame.size < FRAME:
                    break
                prob = self._vad.prob(frame)
                self._cursor += FRAME
                await self._handle(prob)

    async def _handle(self, prob: float) -> None:
        decision = self._machine.update(prob)
        self.metrics.observe("vad.prob", prob)
        if decision == "started":
            self._utt_id = uuid.uuid4().hex[:8]
            self._start_index = self._cursor - FRAME * self._machine.frames
            self.metrics.inc("vad.speech_started")
            self.log.info("Speech started", utterance_id=self._utt_id,
                          dbfs=round(dbfs(self._store.read_since(self._start_index)), 1))
            await self.bus.publish(Event("SpeechStarted", {
                "utterance_id": self._utt_id, "start_index": int(self._start_index),
            }))
        elif decision in ("ended", "ended_timeout"):
            duration = self._machine.speech_duration_s()
            payload = {
                "utterance_id": self._utt_id,
                "start_index": int(self._start_index),
                "end_index": int(self._cursor),
                "duration_s": round(duration, 3),
                "reason": "silence" if decision == "ended" else "hard_timeout",
            }
            self.metrics.inc("vad.speech_ended")
            if decision == "ended_timeout":
                self.metrics.inc("vad.hard_timeouts")
            self.log.info("Speech ended", **payload)
            await self.bus.publish(Event("SpeechEnded", payload))
            self._vad.reset()