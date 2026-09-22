"""MicService — microphone capture with hot reconnect.

Sources:
- ``device``  : real input device via sounddevice (blocking reads on a thread);
- ``file``    : a WAV fixture replayed in real time (loop by default) — this is
  how replay tests and benches run without a microphone;
- ``fake``    : synthetic pattern generated in-process (hermetic runs).

Publishes metadata-only ``AudioChunk`` events (samples go to the shared
``AudioStore``). Read failures publish ``DeviceLost`` a single time, retry
with backoff, and publish ``DeviceRestored`` on recovery — a USB mic unplug
never restarts the app (same contract as the camera).
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from aria.audio.store import DEFAULT_AUDIO_STORE, dbfs, read_wav, resample_linear
from aria.core import readiness
from aria.core.events import Event
from aria.core.service import Service

_SCHEMA = {
    "source": ("fake", (str,)),           # device | file | fake
    "device": (-1, (int,)),               # -1 = system default input
    "rate": (16000, (int,)),
    "block_ms": (32, (int,)),
    "path": ("", (str,)),
    "loop": (True, (bool,)),
    "signal_end": (True, (bool,)),        # announce AudioSourceFinished when a file ends
    "gain": (1.0, (int, float)),
    "reconnect_max": (50, (int,)),
    "reconnect_backoff": (0.5, (int, float)),
    "wait_for_ready": ([], (list,)),      # e.g. ["stt"]: don't feed audio before it can transcribe
    "readiness_timeout_s": (45.0, (int, float)),
    "heartbeat_interval": (5.0, (int, float)),
}


class MicService(Service):
    name = "audio.mic"
    produces = ("AudioChunk", "DeviceLost", "DeviceRestored", "AudioSourceFinished")
    consumes = ()
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._source_finished = False
        self._alive = True
        self._was_alive = True
        self._seq = 0

    # -- lifecycle --------------------------------------------------------
    async def on_start(self) -> None:
        self._store = DEFAULT_AUDIO_STORE
        self._block = max(1, int(int(self.config.get("block_ms", 32)) * int(self.config.get("rate", 16000)) / 1000))
        if str(self.config.get("source", "fake")) in ("device", "file"):
            self._thread = threading.Thread(target=self._grab_loop, name=f"{self.name}:grab", daemon=True)
            self._thread.start()
        self.spawn(self._publish_loop(), "publish")

    async def on_stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- grabber (device / file) -------------------------------------------
    def _open_stream(self):
        import sounddevice as sd

        device = int(self.config.get("device", -1))
        rate = int(self.config.get("rate", 16000))
        stream = sd.InputStream(
            samplerate=rate, channels=1, dtype="float32", blocksize=self._block,
            device=None if device < 0 else device,
        )
        stream.start()
        return stream

    def _file_blocks(self):
        path = str(self.config.get("path", ""))
        if not path:
            raise RuntimeError("audio file source needs a path")
        import pathlib

        p = pathlib.Path(path)
        if not p.is_absolute():
            p = pathlib.Path(__file__).resolve().parents[2] / p
        samples, rate = read_wav(p)
        samples = resample_linear(samples, rate, int(self.config.get("rate", 16000)))
        block = self._block
        return samples, block

    def _wait_ready_blocking(self) -> bool:
        """Hold the source until its dependencies are ready (thread-side poll).

        A fixture replay must not start before the transcriber is warm, or the
        only utterance in the file is dropped as stale.
        """
        names = [str(n) for n in (self.config.get("wait_for_ready") or [])]
        if not names:
            return True
        started = time.monotonic()
        deadline = started + float(self.config.get("readiness_timeout_s", 45.0))
        while not self._stop_evt.is_set() and time.monotonic() < deadline:
            if all(readiness.is_ready(name) for name in names):
                self.log.info("Audio source starting (dependencies ready)",
                              waited_s=round(time.monotonic() - started, 2), waiting_for=names)
                return True
            self._stop_evt.wait(0.05)
        if not self._stop_evt.is_set():
            self.log.warning("Readiness wait timed out; starting audio anyway",
                             waiting_for=names, timeout_s=float(self.config.get("readiness_timeout_s", 45.0)))
        return False

    def _grab_loop(self) -> None:
        source = str(self.config.get("source", "fake"))
        gain = float(self.config.get("gain", 1.0))
        backoff = float(self.config.get("reconnect_backoff", 0.5))
        reconnect_max = int(self.config.get("reconnect_max", 50))
        self._wait_ready_blocking()
        try:
            if source == "file":
                samples, block = self._file_blocks()
                period = block / float(self.config.get("rate", 16000))
                idx = 0
                while not self._stop_evt.is_set():
                    if idx + block > samples.size:
                        if not bool(self.config.get("loop", True)):
                            self.log.info("Audio fixture ended (loop disabled)")
                            self._source_finished = True   # publish loop announces it
                            return
                        idx = 0
                    self._store.append(samples[idx: idx + block] * gain)
                    idx += block
                    self._stop_evt.wait(period)   # real-time pacing
                    self._alive = True
                return

            stream = None
            while not self._stop_evt.is_set():
                if stream is None:
                    try:
                        stream = self._open_stream()
                    except Exception as exc:
                        if self._alive:
                            self._alive = False
                        self.log.warning("Microphone open failed; retrying", error=str(exc))
                        self._stop_evt.wait(backoff)
                        continue
                try:
                    data, overflowed = stream.read(self._block)
                    self._store.append(np.asarray(data, dtype=np.float32).reshape(-1) * gain)
                    self._alive = True
                except Exception as exc:
                    if self._alive:
                        self._alive = False
                        self.log.warning("Microphone read failed; reconnecting", error=str(exc))
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
                    stream = None
                    attempts = 0
                    while attempts < reconnect_max and not self._stop_evt.is_set():
                        self._stop_evt.wait(backoff)
                        attempts += 1
                        try:
                            stream = self._open_stream()
                        except Exception:
                            continue
                        self.log.info("Microphone reconnected", attempts=attempts)
                        break
                    if stream is None:
                        self._stop_evt.wait(5.0)
        except Exception:
            self.log.exception("Microphone grabber thread died")

    # -- publisher ---------------------------------------------------------
    async def _publish_loop(self) -> None:
        source = str(self.config.get("source", "fake"))
        rate = int(self.config.get("rate", 16000))
        period = self._block / float(rate)
        last_total = 0
        while True:
            await asyncio.sleep(period)
            self._seq += 1
            if source == "fake":
                # 1 s of quiet tone followed by 1 s of "speech-like" noise, repeating
                t = (self._seq * period) % 2.0
                block = np.full(self._block, 0.0, dtype=np.float32)
                if t >= 1.0:
                    rng = np.random.default_rng(self._seq)
                    block = (rng.standard_normal(self._block).astype(np.float32) * 0.05)
                self._store.append(block)
            else:
                self._sync_device_events()
            total = self._store.total_samples()
            if total == last_total:
                # A finished (non-looping) fixture is the end of the run for
                # replay profiles: announce it once so the app can shut down
                # deterministically instead of relying on a fixed sleep.
                if self._source_finished and bool(self.config.get("signal_end", True)):
                    self._source_finished = False
                    await self.bus.publish(Event("AudioSourceFinished", {"source": source}))
                continue
            last_total = total
            chunk = self._store.read_window(period * 2)
            await self.bus.publish(
                Event("AudioChunk", {"seq": self._seq, "source": source, "rate": rate,
                                     "block": self._block, "dbfs": round(dbfs(chunk), 1)})
            )
            self.metrics.inc("mic.chunks")

    def _sync_device_events(self) -> None:
        alive = self._alive
        if alive != self._was_alive:
            event = "DeviceRestored" if alive else "DeviceLost"
            self._was_alive = alive
            self.log.info("Device event", event=event, device=str(self.config.get("source")))
            asyncio.get_running_loop().create_task(
                self.bus.publish(Event(event, {"device": "mic"}))
            )