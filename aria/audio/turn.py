"""TurnDetectorService — three-tier end-of-turn (the FUNC-14 fix).

Tier 1: VAD silence (candidate end, from ``vad.py``).
Tier 2: a semantic turn model — Smart Turn v3.2 (ONNX, CPU) run on the tail of
        the utterance; a low probability means "they are only pausing", so we
        keep listening and the segment extends.
Tier 3: hard limits — ``pending_timeout_s`` when the model says "incomplete"
        but nobody continues, and ``max_utterance_s`` from the segment start.

Both the classifier and the decision logic are pluggable/pure:
``EouClassifier`` implementations are swappable by config, and
``EouTracker`` decides complete/wait/discard with no model at all, so the
timing behaviour is unit-tested deterministically.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import numpy as np

from aria.audio.store import DEFAULT_AUDIO_STORE
from aria.core.events import Event
from aria.core.service import Service
from aria.perception.vision import resolve_repo_path

_SCHEMA = {
    "classifier": ("auto", (str,)),        # auto | smart_turn | heuristic
    "model": ("weights/smart-turn-v3.2-cpu.onnx", (str,)),
    "threshold": (0.5, (int, float)),
    "pending_timeout_s": (1.5, (int, float)),   # tier 3: model said "incomplete", speaker stopped
    "max_utterance_s": (20.0, (int, float)),    # tier 3: absolute segment cap
    "min_utterance_s": (0.6, (int, float)),     # below this a discard is deemed noise
    "tail_s": (8.0, (int, float)),              # how much audio the model sees
    "ort_threads": (1, (int,)),        # per-turn inference: one thread is plenty
    "ort_spinning": (False, (bool,)),
    "heartbeat_interval": (5.0, (int, float)),
}


class EouClassifier:
    name = "base"

    def prob(self, audio: np.ndarray) -> float:
        raise NotImplementedError


def mel_filterbank_slaney(rate: int = 16000, n_fft: int = 400, n_mels: int = 80,
                          f_min: float = 0.0, f_max: float = 8000.0) -> np.ndarray:
    """Slaney-normalized mel filterbank (matches whisper/librosa numerics).

    Implemented in numpy so the Smart Turn path needs no torchaudio/librosa
    build (torchaudio has no 2.14 wheel; this keeps the Phase 1 torch pair).
    """
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(f):
        return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp

    def mel_to_hz(m):
        return min_log_hz * np.exp(logstep * (m - min_log_mel)) if m >= min_log_mel else f_sp * m

    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    freqs = np.array([mel_to_hz(m) for m in mels])
    fft_freqs = np.linspace(0.0, rate / 2.0, 1 + n_fft // 2)
    weights = np.zeros((n_mels, fft_freqs.size), dtype=np.float64)
    for i in range(n_mels):
        lower, center, upper = freqs[i], freqs[i + 1], freqs[i + 2]
        for j, f in enumerate(fft_freqs):
            if lower <= f <= center and center > lower:
                weights[i, j] = (f - lower) / (center - lower)
            elif center < f <= upper and upper > center:
                weights[i, j] = (upper - f) / (upper - center)
    enorm = 2.0 / (freqs[2: n_mels + 2] - freqs[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


def _log_mel_whisper(audio: np.ndarray, rate: int = 16000, n_mels: int = 80,
                     frames: int = 800, n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """Whisper-style log-mel spectrogram (80 × frames), pure numpy."""
    target = frames * hop
    audio = np.asarray(audio, dtype=np.float32)
    # Right-align the window: it must always END at the latest sample, because
    # end-of-turn is a statement about the tail of the audio. Left-padding keeps
    # that invariant for utterances shorter than the window.
    if audio.size < target:
        audio = np.pad(audio, (target - audio.size, 0))
    else:
        audio = audio[-target:]
    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)
    n_frames = 1 + (audio.size - n_fft) // hop
    strided = np.lib.stride_tricks.as_strided(
        audio, shape=(n_frames, n_fft), strides=(audio.strides[0] * hop, audio.strides[0]),
    )
    spectra = np.fft.rfft(strided * window, n=n_fft, axis=1)
    power = (np.abs(spectra) ** 2).T                      # (freq, frames)
    mel = np.dot(mel_filterbank_slaney(rate, n_fft, n_mels), power)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    if log_spec.shape[-1] != frames:
        if log_spec.shape[-1] > frames:
            log_spec = log_spec[:, :frames]
        else:
            log_spec = np.pad(log_spec, ((0, 0), (0, frames - log_spec.shape[-1])))
    return log_spec[np.newaxis].astype(np.float32)


class SmartTurnClassifier(EouClassifier):
    """Smart Turn v3.2 (BSD-2) ONNX: P(turn complete) from the audio tail."""

    name = "smart-turn-v3.2"

    def __init__(self, model_path: str, rate: int = 16000, frames: int = 800,
                 threads: Optional[int] = None, spinning: Optional[bool] = None) -> None:
        import onnxruntime as ort

        from aria.core.onnx import make_session

        ort.set_default_logger_severity(3)
        self.session = make_session(model_path, kind="turn", threads=threads, spinning=spinning)
        self.input_name = self.session.get_inputs()[0].name
        self.rate = rate
        self.frames = frames

    def prob(self, audio: np.ndarray) -> float:
        features = _log_mel_whisper(np.asarray(audio, dtype=np.float32), self.rate, frames=self.frames)
        raw = np.asarray(self.session.run(None, {self.input_name: features})[0]).reshape(-1)
        value = float(raw[0])
        if 0.0 <= value <= 1.0:
            return value
        return float(1.0 / (1.0 + np.exp(-value)))   # logit → probability


class HeuristicEouClassifier(EouClassifier):
    """Fallback when the semantic model is unavailable: trailing silence and
    utterance length. Pure logic — no model, deterministic, testable."""

    name = "heuristic-pause"

    def __init__(self, rate: int = 16000, long_utterance_s: float = 2.5,
                 trailing_silence_s: float = 0.25) -> None:
        self.rate = rate
        self.long_utterance_s = float(long_utterance_s)
        self.trailing_silence_s = float(trailing_silence_s)

    def prob(self, audio: np.ndarray) -> float:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return 0.0
        duration = audio.size / self.rate
        tail_n = int(self.trailing_silence_s * self.rate)
        tail = audio[-tail_n:] if audio.size > tail_n else audio
        quiet = float(np.sqrt(np.mean(np.square(tail, dtype=np.float64)))) < 0.01
        if duration >= self.long_utterance_s and quiet:
            return 0.9
        if duration >= self.long_utterance_s:
            return 0.6
        return 0.2


class EouTracker:
    """Pure three-tier decision machine (no model, no clock side effects)."""

    def __init__(self, threshold: float = 0.5, pending_timeout_s: float = 1.5,
                 max_utterance_s: float = 20.0, min_utterance_s: float = 0.6) -> None:
        self.threshold = float(threshold)
        self.pending_timeout_s = float(pending_timeout_s)
        self.max_utterance_s = float(max_utterance_s)
        self.min_utterance_s = float(min_utterance_s)

    def on_speech_end(self, now: float, segment_started: float, p_turn: float,
                      duration_s: float, reason: str = "silence") -> str:
        """→ 'complete' | 'wait' | 'discard'."""
        if reason == "hard_timeout":
            return "complete"
        if duration_s >= self.max_utterance_s:
            return "complete"
        if p_turn >= self.threshold:
            return "complete"
        if duration_s < self.min_utterance_s:
            return "discard"          # cough/click/backchannel: never worth a reply
        return "wait"

    def on_pending_expiry(self, now: float, segment_started: float, duration_s: float) -> str:
        """Nobody continued speaking → finish long speech, drop tiny noise."""
        if duration_s < self.min_utterance_s:
            return "discard"
        return "complete"


class TurnDetectorService(Service):
    name = "audio.turn"
    produces = ("TurnCompleted",)
    consumes = ("SpeechStarted", "SpeechEnded")
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._subs = []
        self._classifier: Optional[EouClassifier] = None
        self._tracker: Optional[EouTracker] = None
        self._segment: Optional[dict] = None
        self._pending_task: Optional[asyncio.Task] = None

    # -- setup -------------------------------------------------------------
    async def init(self) -> None:
        self._store = DEFAULT_AUDIO_STORE
        self._tracker = EouTracker(
            threshold=float(self.config.get("threshold", 0.5)),
            pending_timeout_s=float(self.config.get("pending_timeout_s", 1.5)),
            max_utterance_s=float(self.config.get("max_utterance_s", 20.0)),
            min_utterance_s=float(self.config.get("min_utterance_s", 0.6)),
        )
        self._classifier = self._build_classifier()

    def _build_classifier(self) -> EouClassifier:
        want = str(self.config.get("classifier", "auto"))
        model = resolve_repo_path(str(self.config.get("model", "")))
        if want in ("auto", "smart_turn") and model.exists():
            try:
                classifier = SmartTurnClassifier(str(model),
                                                 threads=self.config.get("ort_threads"),
                                                 spinning=self.config.get("ort_spinning"))
                self.log.info("End-of-turn classifier ready", classifier=classifier.name, model=model.name)
                return classifier
            except Exception as exc:
                if want == "smart_turn":
                    raise
                self.log.warning("Smart Turn unavailable; using heuristic fallback", error=str(exc))
        elif want == "smart_turn":
            raise FileNotFoundError(f"Smart Turn model not found: {model}")
        self.log.info("End-of-turn classifier ready", classifier="heuristic-pause")
        return HeuristicEouClassifier()

    # -- lifecycle ----------------------------------------------------------
    async def on_start(self) -> None:
        self._subs.append(self.bus.subscribe("SpeechStarted", self._on_speech_started, policy="drop_new", maxsize=4))
        self._subs.append(self.bus.subscribe("SpeechEnded", self._on_speech_ended, policy="drop_new", maxsize=4))

    async def on_stop(self) -> None:
        if self._pending_task is not None:
            self._pending_task.cancel()
            self._pending_task = None
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs.clear()

    # -- handlers -----------------------------------------------------------
    async def _on_speech_started(self, event: Event) -> None:
        if self._segment is None:
            self._segment = {
                "utterance_id": event.payload.get("utterance_id"),
                "start_index": int(event.payload.get("start_index", 0)),
                "started": time.monotonic(),
            }
        else:  # continuation: the segment keeps its original id and start
            self._cancel_pending()
            self.log.debug("Turn extended by new speech", utterance_id=self._segment["utterance_id"])

    async def _on_speech_ended(self, event: Event) -> None:
        if self._segment is None:
            self._segment = {
                "utterance_id": event.payload.get("utterance_id"),
                "start_index": int(event.payload.get("start_index", 0)),
                "started": time.monotonic(),
            }
        self._cancel_pending()
        segment = self._segment
        end_index = int(event.payload.get("end_index", self._store.total_samples()))
        duration = float(event.payload.get("duration_s", 0.0))
        reason = str(event.payload.get("reason", "silence"))
        audio = self._store.read_since(segment["start_index"])
        tail = int(float(self.config.get("tail_s", 8.0)) * self._store.sample_rate)
        if audio.size > tail:
            audio = audio[-tail:]
        started = time.monotonic()
        p_turn = await asyncio.to_thread(self._classifier.prob, audio)
        classify_ms = (time.monotonic() - started) * 1000
        self.metrics.observe("eou.classify_ms", classify_ms)
        self.metrics.observe("eou.p_turn", p_turn)
        decision = self._tracker.on_speech_end(
            now=time.monotonic(), segment_started=segment["started"],
            p_turn=p_turn, duration_s=duration, reason=reason,
        )
        self.log.info("End-of-turn decision", utterance_id=segment["utterance_id"],
                      decision=decision, p_turn=round(p_turn, 3),
                      classifier=self._classifier.name, duration_s=round(duration, 2))
        if decision == "complete":
            await self._publish_turn(segment, end_index, duration, p_turn, reason, classify_ms)
        elif decision == "discard":
            self.metrics.inc("eou.discarded")
            self._segment = None
        else:  # wait: the speaker is probably only pausing
            self.metrics.inc("eou.deferred")
            if reason == "hard_timeout":
                await self._publish_turn(segment, end_index, duration, p_turn, reason, classify_ms)
                return
            timeout = self._tracker.pending_timeout_s
            self._pending_task = self.spawn(
                self._pending_expiry(segment["utterance_id"], timeout), "pending_expiry"
            )

    async def _pending_expiry(self, utterance_id: str, timeout: float) -> None:
        await asyncio.sleep(timeout)
        segment = self._segment
        if segment is None or segment["utterance_id"] != utterance_id:
            return
        duration = (time.monotonic() - segment["started"])
        decision = self._tracker.on_pending_expiry(
            now=time.monotonic(), segment_started=segment["started"], duration_s=duration,
        )
        self.log.info("End-of-turn resolved by timeout", utterance_id=utterance_id, decision=decision)
        if decision == "complete":
            await self._publish_turn(segment, int(self._store.total_samples()), duration,
                                     p_turn=float("nan"), reason="pending_timeout", classify_ms=0.0)
        else:
            self.metrics.inc("eou.discarded")
            self._segment = None

    async def _publish_turn(self, segment: dict, end_index: int, duration: float,
                            p_turn: float, reason: str, classify_ms: float) -> None:
        self.metrics.inc("eou.turns_completed")
        await self.bus.publish(Event("TurnCompleted", {
            "utterance_id": segment["utterance_id"],
            "start_index": int(segment["start_index"]),
            "end_index": int(end_index),
            "duration_s": round(duration, 3),
            "p_turn": None if p_turn != p_turn else round(p_turn, 3),
            "classifier": self._classifier.name,
            "reason": reason,
            "classify_ms": round(classify_ms, 1),
        }))
        self._segment = None

    def _cancel_pending(self) -> None:
        if self._pending_task is not None:
            self._pending_task.cancel()
            self._pending_task = None