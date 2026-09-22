"""PersonVisionService — YOLO + ByteTrack: ONE code path for 0/1/N people.

The core of the unified-perception design (ROADMAP §7): every cycle pulls the
latest pixels from the shared FrameStore, runs Ultralytics person detection
+ tracking (person class only), diffs the track set against the previous
cycle, and publishes:
  - ``TrackAppeared`` / ``TrackLost`` exactly once per track episode;
  - ``SceneTick`` with the active tracks (ids, bboxes, conf) for the
    SceneManager and face pipeline.

Model and tracker are config: ByteTrack → BoT-SORT or YOLO26n → YOLO11n is a
config edit, not code. Heavy imports (ultralytics/torch) are lazy so the
hermetic test profile never pays for them.
"""
from __future__ import annotations

import asyncio
import pathlib
import time
from typing import List, Optional, Tuple

from aria.core.events import Event
from aria.core.service import Service

MB = 1024 * 1024

_SCHEMA = {
    "model": ("yolo26n.pt", (str,)),
    "fallback_models": (["yolo11n.pt"], (list,)),
    "device": ("auto", (str,)),           # auto | cuda | cpu
    "detect_hz": (15.0, (int, float)),
    "busy_detect_hz": (5.0, (int, float)),   # rate while a turn is being transcribed/spoken
    "busy_after_turn_s": (1.5, (int, float)),  # keep yielding this long after speech ends
    "yield_while_speaking": (True, (bool,)),
    "conf": (0.30, (int, float)),
    "iou": (0.50, (int, float)),
    "imgsz": (640, (int,)),
    "tracker_yaml": ("configs/tracker_bytetrack.yaml", (str,)),
    "warmup": (True, (bool,)),
    "heartbeat_interval": (5.0, (int, float)),
}


def resolve_repo_path(name: str) -> pathlib.Path:
    root = pathlib.Path(__file__).resolve().parents[2]
    p = pathlib.Path(name)
    return p if p.is_absolute() else root / p


# Events that mean "the machine is needed elsewhere": a turn being transcribed
# and a reply being spoken. Deliberately NOT SpeechStarted — that is the
# *listening* phase, where cheap VAD runs and vision is what feeds engagement
# tracking; throttling there suppressed the very signal the gate needs.
BUSY_EVENTS = ("TurnCompleted", "SpeechSynthesized")


def busy_seconds_for(name: str, payload: dict, config) -> float:
    """How long vision should yield for an event (0.0 = not a yield trigger)."""
    if name == "TurnCompleted":
        return float(config.get("busy_after_turn_s", 1.5))
    if name == "SpeechSynthesized":
        # Published just before each clause plays, so the yield lasts exactly as
        # long as ARIA is talking.
        return float(payload.get("audio_s", 0.0)) + 0.15
    return 0.0


class PersonVisionService(Service):
    name = "perception.vision"
    produces = ("TrackAppeared", "TrackLost", "SceneTick")
    consumes = ("Frame",)
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.device_str = "cpu"
        self._sub = None
        self._tracks: dict = {}      # track_id -> last bbox
        self._last_fid = None
        self._busy_until = 0.0       # set while STT/TTS need the machine
        self._busy_subs: list = []

    # -- setup -----------------------------------------------------------
    async def init(self) -> None:
        from aria.perception.framestore import DEFAULT_STORE

        self._store = DEFAULT_STORE
        self._tracker_yaml = resolve_repo_path(str(self.config.get("tracker_yaml", "")))
        self._model_names = [str(self.config.get("model", "yolo26n.pt"))] + [
            str(m) for m in (self.config.get("fallback_models") or [])
        ]
        self.device_str = self._resolve_device()
        self.model = self._load_model()

    def _resolve_device(self) -> str:
        want = str(self.config.get("device", "auto"))
        if want == "auto":
            try:
                import torch

                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return want

    def _load_model(self):
        from ultralytics import YOLO

        last_error = None
        for name in self._model_names:
            for candidate in (resolve_repo_path(f"weights/{name}"), resolve_repo_path(name)):
                if candidate.exists():
                    try:
                        model = YOLO(str(candidate))
                        self.log.info("Detector loaded", model=candidate.name, device=self.device_str)
                        return model
                    except Exception as exc:  # try next
                        last_error = exc
        # last resort: let ultralytics auto-download the bare name
        try:
            model = YOLO(self._model_names[0])
            self.log.info("Detector loaded (auto-download)", model=self._model_names[0])
            return model
        except Exception:
            raise RuntimeError(
                f"No detector model available from {self._model_names}: {last_error}"
            )

    # -- lifecycle -------------------------------------------------------
    async def on_start(self) -> None:
        self._sub = self.bus.subscribe("Frame", self._on_frame, policy="drop_oldest", maxsize=2)
        if bool(self.config.get("yield_while_speaking", True)):
            self._busy_subs = [
                self.bus.subscribe(name, self._on_busy_event, policy="drop_new", maxsize=4)
                for name in BUSY_EVENTS
            ]
        if bool(self.config.get("warmup", True)):
            await asyncio.to_thread(self._warmup)
        self.spawn(self._detect_loop(), "detect")

    async def on_stop(self) -> None:
        if self._sub is not None:
            self.bus.unsubscribe(self._sub)
            self._sub = None
        for sub in getattr(self, "_busy_subs", []):
            self.bus.unsubscribe(sub)
        self._busy_subs = []

    # -- "yield while speaking" -------------------------------------------
    def _busy(self) -> bool:
        return time.monotonic() < self._busy_until

    def _mark_busy(self, seconds: float) -> None:
        self._busy_until = max(self._busy_until, time.monotonic() + max(0.0, seconds))

    async def _on_busy_event(self, event: Event) -> None:
        self._mark_busy(busy_seconds_for(event.name, event.payload, self.config))

    async def _on_frame(self, event: Event) -> None:  # pacing happens in the detect loop
        self.metrics.inc("vision.frames_seen")

    def _warmup(self) -> None:
        import numpy as np

        start = time.perf_counter()
        self._infer(np.zeros((480, 640, 3), dtype=np.uint8))
        self.log.info(
            "Detector warmed up",
            device=self.device_str,
            seconds=round(time.perf_counter() - start, 2),
        )

    # -- detection loop ---------------------------------------------------
    async def _detect_loop(self) -> None:
        base_hz = max(float(self.config.get("detect_hz", 15.0)), 0.001)
        busy_hz = float(self.config.get("busy_detect_hz", base_hz))
        next_at = time.monotonic()
        while True:
            # "Yield while speaking": while a turn is being transcribed or a reply
            # is playing, the GPU/CPU are needed for STT and TTS, so detection
            # drops to busy_detect_hz instead of competing with them (measured:
            # this is the difference between 0.88 s and 1.44 s STT latency).
            busy = self._busy()
            interval = 1.0 / (busy_hz if busy else base_hz)
            # Pace against a schedule instead of sleep-after-work, so inference
            # time does not silently add to the period (that drift is what made a
            # 15 Hz loop behave like 4 Hz).
            next_at = max(next_at + interval, time.monotonic())
            delay = next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            fid, frame = self._store.latest()
            if frame is None or fid == self._last_fid:
                next_at = time.monotonic()
                continue
            self._last_fid = fid
            self.metrics.inc("vision.busy_detects" if busy else "vision.idle_detects")
            start = time.perf_counter()
            tracks = await asyncio.to_thread(self._infer, frame)
            latency = time.perf_counter() - start
            self.metrics.observe("vision.detect_latency_s", latency)
            if self.device_str.startswith("cuda"):
                try:
                    import torch

                    self.metrics.observe("vision.vram_used_mb", torch.cuda.memory_allocated() / MB)
                except Exception:
                    pass
            await self._process_tracks(tracks, fid, latency)

    def _infer(self, frame) -> List[Tuple[int, Tuple[int, int, int, int], float]]:
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],  # person only
            conf=float(self.config.get("conf", 0.30)),
            iou=float(self.config.get("iou", 0.50)),
            imgsz=int(self.config.get("imgsz", 640)),
            tracker=str(self._tracker_yaml),
            device=self.device_str,
            verbose=False,
        )
        tracks: List[Tuple[int, Tuple[int, int, int, int], float]] = []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.id is None:
            return tracks
        for bbox, tid, conf in zip(boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist()):
            x1, y1, x2, y2 = (int(v) for v in bbox)
            tracks.append((int(tid), (x1, y1, x2, y2), float(conf)))
        return tracks

    async def _process_tracks(self, tracks, fid: int, latency: float) -> None:
        hz = float(self.config.get("detect_hz", 15.0))
        now_ids = {tid for tid, _, _ in tracks}
        for tid, bbox, conf in tracks:
            if tid not in self._tracks:
                self.metrics.inc("vision.tracks_appeared")
                await self.bus.publish(
                    Event("TrackAppeared", {"track_id": tid, "bbox": list(bbox), "conf": conf})
                )
            self._tracks[tid] = bbox
        for tid in list(self._tracks):
            if tid not in now_ids:
                self.metrics.inc("vision.tracks_lost")
                await self.bus.publish(
                    Event("TrackLost", {"track_id": tid, "last_bbox": list(self._tracks.pop(tid))})
                )
        await self.bus.publish(
            Event(
                "SceneTick",
                {
                    "tracks": [{"id": tid, "bbox": list(bbox), "conf": conf} for tid, bbox, conf in tracks],
                    "frame_id": fid,
                    "latency_ms": round(latency * 1000, 1),
                    "detect_hz": hz,
                },
            )
        )
