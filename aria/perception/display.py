"""DebugDisplayService — OpenCV HUD window (laptop profile only).

Reads the latest pixels from the shared FrameStore and overlays the
TrackStates/FaceMatched/tamper information the other services publish.
Headless machines and the hermetic test profile simply leave it disabled.
A crash or a missing GUI degrades to a logged no-op — telemetry never
depends on the display.

View options (all config, no code):
- ``flip``: ``horizontal`` mirrors the view like a selfie camera (default),
  ``vertical`` / ``both`` handle an upside-down mount, ``none`` is raw. Press
  **f** in the window to cycle it live. Mirroring happens *here*, not in the
  pipeline, so detection, tracking and face galleries keep using raw frames.
- ``fit``: ``letterbox`` (default) scales the frame into the window preserving
  aspect ratio, so a 4:3 camera is not stretched into a 16:9 window;
  ``stretch`` restores the old fill-the-window behaviour.
- ``window_width``: initial window width; the height follows the camera's own
  aspect ratio.
"""
from __future__ import annotations

import asyncio
import threading
import time

from aria.core.events import Event
from aria.core.service import Service
from aria.perception.transform import FLIP_MODES, apply_flip, flip_bbox, fit_to_window, normalize_flip

_SCHEMA = {
    "window": ("ARIA-POC debug", (str,)),
    "target_fps": (15.0, (int, float)),   # match the camera rate: redrawing faster is waste
    "window_width": (960, (int,)),
    "fit": ("letterbox", (str,)),         # letterbox | stretch
    "flip": ("horizontal", (str,)),       # none | horizontal (mirror) | vertical | both
    "heartbeat_interval": (5.0, (int, float)),
}

STATE_COLORS = {
    "PRESENCE": (160, 160, 160),
    "NEAR": (255, 180, 0),
    "ENGAGED": (0, 200, 0),
}


class DebugDisplayService(Service):
    name = "perception.display"
    produces = ()
    consumes = ("TrackStates", "FaceMatched", "TamperDetected", "TamperCleared")
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._subs = []
        self._scene = {"tracks": [], "ts": 0.0}
        self._tampered = False
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._flip = "horizontal"
        self._last_fid = -1

    async def on_start(self) -> None:
        self._subs.append(self.bus.subscribe("TrackStates", self._on_track_states, policy="drop_oldest", maxsize=2))
        self._subs.append(self.bus.subscribe("FaceMatched", self._on_face, policy="drop_new", maxsize=4))
        self._subs.append(self.bus.subscribe("TamperDetected", self._on_tamper, policy="drop_new", maxsize=1))
        self._subs.append(self.bus.subscribe("TamperCleared", self._on_tamper_clear, policy="drop_new", maxsize=1))
        self._thread = threading.Thread(target=self._window_loop, name=f"{self.name}:win", daemon=True)
        self._thread.start()

    async def on_stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs.clear()

    # -- event handlers -----------------------------------------------------
    async def _on_track_states(self, event: Event) -> None:
        self._scene = {"tracks": event.payload.get("tracks", []), "ts": time.monotonic()}

    async def _on_face(self, event: Event) -> None:
        for track in self._scene["tracks"]:
            if track["id"] == event.payload.get("track_id"):
                track["identity"] = event.payload.get("identity")
                track["similarity"] = event.payload.get("similarity")

    async def _on_tamper(self, event: Event) -> None:
        self._tampered = True

    async def _on_tamper_clear(self, event: Event) -> None:
        self._tampered = False

    # -- window thread --------------------------------------------------------
    def _window_loop(self) -> None:
        try:
            import cv2
            from aria.perception.framestore import DEFAULT_STORE

            window = str(self.config.get("window", "ARIA-POC debug"))
            interval = 1.0 / max(float(self.config.get("target_fps", 30.0)), 0.001)
            self._flip = normalize_flip(str(self.config.get("flip", "horizontal")))
            fit = str(self.config.get("fit", "letterbox")).lower()
            target_width = max(160, int(self.config.get("window_width", 960)))
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            sized = False
            self.log.info("Debug window open", flip=self._flip, fit=fit, window_width=target_width)
            while not self._stop_evt.is_set():
                frame = None
                fid, frame = DEFAULT_STORE.latest()
                # Redraw only when the source produced a new frame: at 15 fps this
                # halves the copy/draw/resize/imshow work of a 30 fps loop.
                if frame is None or fid == self._last_fid:
                    self._stop_evt.wait(min(interval, 0.02))
                    continue
                self._last_fid = fid
                if not sized:                     # match the camera's aspect ratio
                    height, width = frame.shape[:2]
                    win_h = max(120, int(round(target_width * height / max(width, 1))))
                    cv2.resizeWindow(window, target_width, win_h)
                    sized = True
                    self.log.info("Debug window sized to the camera aspect ratio",
                                  frame=f"{width}x{height}", window=f"{target_width}x{win_h}")
                shown = frame.copy()
                if self._flip != "none":
                    shown = apply_flip(shown, self._flip)
                self._draw(cv2, shown)
                if fit == "stretch":
                    cv2.imshow(window, shown)
                else:
                    win_w, win_h = self._window_size(cv2, window, target_width, shown)
                    cv2.imshow(window, fit_to_window(shown, win_w, win_h))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.log.info("Debug window closed by user (q)")
                    break
                if key == ord("f"):
                    self._flip = FLIP_MODES[(FLIP_MODES.index(self._flip) + 1) % len(FLIP_MODES)]
                    self.log.info("Debug view flip changed", flip=self._flip)
                self._stop_evt.wait(interval)
            cv2.destroyWindow(window)
        except Exception as exc:
            self.log.warning("Display unavailable; continuing without HUD", error=str(exc))

    @staticmethod
    def _window_size(cv2, window: str, fallback_w: int, frame) -> tuple[int, int]:
        """Current window size (the user may have resized it)."""
        try:
            _x, _y, width, height = cv2.getWindowImageRect(window)
            if width > 0 and height > 0:
                return int(width), int(height)
        except Exception:
            pass
        height, width = frame.shape[:2]
        return int(fallback_w), max(120, int(round(fallback_w * height / max(width, 1))))

    def _draw(self, cv2, frame) -> None:
        height, width = frame.shape[:2]
        for track in self._scene["tracks"]:
            x1, y1, x2, y2 = flip_bbox(track["bbox"], width, height, self._flip)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            state = track.get("state", "PRESENCE")
            color = STATE_COLORS.get(state, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            identity = track.get("identity") or "unknown"
            label = f"id{track['id']} {state} {identity}"
            if track.get("similarity") is not None:
                label += f" {track['similarity']:.2f}"
            cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        if self._tampered:
            cv2.putText(frame, "!! CAMERA TAMPERED !!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        fps = self._scene.get("ts", 0.0)
        age = time.monotonic() - fps if fps else -1
        cv2.putText(frame, f"scene age {age:.1f}s  flip {self._flip}  (f flip, q quit)",
                    (20, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
