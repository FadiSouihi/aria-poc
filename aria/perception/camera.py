"""Camera source service: fake | webcam | file — threaded grabber, hot reconnect.

- Publishes metadata-only ``Frame`` events at ``fps`` and stores pixels in
  the shared ``FrameStore`` (``framestore.DEFAULT_STORE``) for paced consumers.
- Hot reconnect (the NFR-03 Step 6 fix): a grabber-thread read failure
  publishes ``DeviceLost`` once and the thread retries with backoff
  (``reconnect_max`` attempts per episode, then a long cool-down and retry).
  On recovery ``DeviceRestored`` is published. The app never restarts.
- ``fake`` keeps Phase 0 behavior (metadata only, no pixels) for hermetic runs.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Tuple

from aria.core.events import Event
from aria.core.service import Service

_SCHEMA = {
    "source": ("fake", (str,)),           # fake | webcam | file
    "device": (0, (int,)),                # webcam index
    "path": ("", (str,)),                 # file path (file source)
    "fps": (15.0, (int, float)),
    "width": (640, (int,)),
    "height": (480, (int,)),
    "backend": ("dshow", (str,)),         # webcam backend: dshow | any
    "fake_pixels": (False, (bool,)),      # fake source: also feed noise pixels to the FrameStore
    "loop": (True, (bool,)),              # file source: loop at end
    "reconnect_max": (50, (int,)),
    "reconnect_backoff": (0.5, (int, float)),
    "heartbeat_interval": (5.0, (int, float)),
}


class CameraService(Service):
    name = "perception.camera"
    produces = ("Frame", "DeviceLost", "DeviceRestored")
    consumes = ()
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._alive = True            # last read OK (thread writes)
        self._was_alive = True        # publisher-side transition memory
        self._fid = 0                 # grabber-side frame counter
        self._fid_lock = threading.Lock()
        self._seq = 0                 # publisher-side event counter

    # -- lifecycle -------------------------------------------------------
    async def on_start(self) -> None:
        from aria.perception.framestore import DEFAULT_STORE

        self._store = DEFAULT_STORE
        source = str(self.config.get("source", "fake"))
        if source in ("webcam", "file"):
            self._thread = threading.Thread(target=self._grab_loop, name=f"{self.name}:grab", daemon=True)
            self._thread.start()
        self.spawn(self._publish_loop(), "publish")

    async def on_stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- grabber thread --------------------------------------------------
    def _open_capture(self):
        import cv2

        source = str(self.config.get("source", "fake"))
        if source == "webcam":
            backend = cv2.CAP_DSHOW if str(self.config.get("backend", "dshow")) == "dshow" else cv2.CAP_ANY
            cap = cv2.VideoCapture(int(self.config.get("device", 0)), backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.get("width", 640)))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.get("height", 480)))
        else:  # file
            path = str(self.config.get("path", ""))
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {path}")
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source={source}")
        return cap

    def _grab_loop(self) -> None:
        import cv2

        cap = None
        backoff = float(self.config.get("reconnect_backoff", 0.5))
        reconnect_max = int(self.config.get("reconnect_max", 50))
        try:
            while not self._stop_evt.is_set():
                if cap is None:
                    try:
                        cap = self._open_capture()
                    except Exception as exc:
                        if self._alive:
                            self._alive = False  # publisher emits DeviceLost
                        self.log.warning("Camera open failed; retrying", error=str(exc))
                        self._stop_evt.wait(backoff)
                        continue

                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._fid_lock:
                        self._fid += 1
                        self._store.put(self._fid, frame)
                    self._alive = True
                    continue

                # Quiet loop restart for looping file sources (not a device loss).
                if str(self.config.get("source")) == "file" and bool(self.config.get("loop", True)):
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if cap.isOpened():
                            continue
                    except Exception:
                        pass
                    try:
                        cap.release()
                        cap = self._open_capture()
                        continue
                    except Exception:
                        cap = None
                        continue

                # Read failed → hot-reconnect episode (DeviceLost → retries → DeviceRestored)
                if self._alive:
                    self._alive = False
                    self.log.warning("Camera read failed; reconnecting")
                cap.release()
                cap = None
                attempts = 0
                while attempts < reconnect_max and not self._stop_evt.is_set():
                    self._stop_evt.wait(backoff)
                    attempts += 1
                    try:
                        cap = self._open_capture()
                    except Exception:
                        continue
                    self.log.info("Camera reconnected", attempts=attempts)
                    break
                if cap is None:
                    # episode exhausted → cool down, then a fresh episode (keeps trying)
                    self._stop_evt.wait(5.0)
        except Exception:
            self.log.exception("Camera grabber thread died")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    # -- publisher loop ---------------------------------------------------
    async def _publish_loop(self) -> None:
        source = str(self.config.get("source", "fake"))
        fps = max(float(self.config.get("fps", 15.0)), 0.001)
        interval = 1.0 / fps
        last_fid = None
        while True:
            await asyncio.sleep(interval)
            self._seq += 1

            if source == "fake":
                if bool(self.config.get("fake_pixels", False)):
                    import numpy as np

                    rng = np.random.default_rng()
                    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
                    with self._fid_lock:
                        self._fid += 1
                        self._store.put(self._fid, frame)
                    await self.bus.publish(
                        Event("Frame", {"seq": self._seq, "source": source, "frame_id": self._fid,
                                        "width": 640, "height": 480})
                    )
                else:
                    await self.bus.publish(
                        Event("Frame", {"seq": self._seq, "source": source,
                                        "width": int(self.config.get("width", 640)),
                                        "height": int(self.config.get("height", 480))})
                    )
                self.metrics.inc("camera.frames")
                continue

            self._sync_device_events()
            with self._fid_lock:
                fid = self._fid
            if fid == 0 or fid == last_fid:
                continue
            last_fid = fid
            await self.bus.publish(
                Event("Frame", {"seq": self._seq, "source": source, "frame_id": fid,
                                "width": int(self.config.get("width", 640)),
                                "height": int(self.config.get("height", 480))})
            )
            self.metrics.inc("camera.frames")

    def _sync_device_events(self) -> None:
        alive = self._alive
        if alive != self._was_alive:
            event = "DeviceRestored" if alive else "DeviceLost"
            self._was_alive = alive
            self.log.info("Device event", event=event)
            asyncio.get_running_loop().create_task(
                self.bus.publish(Event(event, {"device": str(self.config.get("source"))}))
            )
