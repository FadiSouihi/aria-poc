"""Audio buffer: the audio twin of ``perception/framestore.py``.

The bus carries only metadata (``AudioChunk`` events stay JSON/timeline
friendly); samples live here. Single writer (MicService), many readers
(VAD, end-of-turn, STT, voice-prints, barge-in), bounded memory (a rolling
window of the last ``retained_s`` seconds at 16 kHz mono float32).

Readers work in two ways:
- ``read_window(seconds)`` — the last N seconds (used by VAD/Smart Turn);
- ``read_since(ts)`` — everything after a timestamp (used to cut an
  utterance segment for STT/voice-prints).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000


class AudioStore:
    def __init__(self, sample_rate: int = SAMPLE_RATE, retained_s: float = 30.0) -> None:
        self.sample_rate = int(sample_rate)
        self.retained_samples = int(retained_s * self.sample_rate)
        self._lock = threading.Lock()
        # list of (start_index, samples) where start_index counts absolute samples
        self._chunks: Deque[Tuple[int, np.ndarray]] = deque()
        self._total = 0          # absolute sample count written
        self._dropped = 0        # samples evicted by the retention window

    # -- writes -----------------------------------------------------------
    def append(self, samples: np.ndarray) -> None:
        block = np.asarray(samples, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return
        with self._lock:
            self._chunks.append((self._total, block))
            self._total += block.size
            cutoff = self._total - self.retained_samples
            while self._chunks:
                start, head = self._chunks[0]
                end = start + head.size
                if end <= cutoff:            # whole chunk is older than the window
                    self._chunks.popleft()
                    self._dropped += head.size
                elif start < cutoff:         # trim the head chunk to the window edge
                    trim = cutoff - start
                    self._chunks[0] = (start + trim, head[trim:])
                    self._dropped += trim
                    break
                else:
                    break

    # -- reads ------------------------------------------------------------
    def total_samples(self) -> int:
        with self._lock:
            return self._total

    def duration_s(self) -> float:
        return self.total_samples() / self.sample_rate

    def read_window(self, seconds: float) -> np.ndarray:
        """Last ``seconds`` of audio, oldest first (possibly shorter at start)."""
        want = int(seconds * self.sample_rate)
        with self._lock:
            chunks = list(self._chunks)
            total = self._total
        out: List[np.ndarray] = []
        have = 0
        for start, block in reversed(chunks):
            out.append(block)
            have += block.size
            if have >= want:
                break
        if not out:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(list(reversed(out)))
        return audio[-want:] if audio.size > want else audio

    def read_since(self, start_index: int) -> np.ndarray:
        """Audio from absolute sample index ``start_index`` to now."""
        with self._lock:
            chunks = list(self._chunks)
        out: List[np.ndarray] = []
        for start, block in chunks:
            end = start + block.size
            if end <= start_index:
                continue
            offset = max(0, start_index - start)
            out.append(block[offset:])
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out)

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_samples": self._total,
                "chunks": len(self._chunks),
                "dropped_samples": self._dropped,
                "retained_s": round(self.retained_samples / self.sample_rate, 2),
            }


DEFAULT_AUDIO_STORE = AudioStore()


# -- helpers shared by VAD / STT / voice-prints --------------------------------
def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def dbfs(samples: np.ndarray) -> float:
    level = rms(samples)
    return -120.0 if level <= 1e-9 else float(20.0 * np.log10(level))


def write_wav(path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Minimal 16-bit PCM WAV writer (no scipy/soundfile dependency)."""
    import struct
    import wave

    data = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm)


def read_wav(path) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV to (mono float32, sample_rate)."""
    import wave

    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        raw = fh.readframes(fh.getnframes())
    if width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got {width * 8}-bit")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Cheap linear resampler (fixtures/tools only — models get 16 kHz)."""
    if src_rate == dst_rate or samples.size == 0:
        return np.asarray(samples, dtype=np.float32)
    duration = samples.size / float(src_rate)
    n_out = max(1, int(round(duration * dst_rate)))
    src_idx = np.linspace(0.0, samples.size - 1, n_out)
    return np.interp(src_idx, np.arange(samples.size), samples).astype(np.float32)