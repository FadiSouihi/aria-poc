"""Tamper detection — fixes FUNC-05 Step 13 (no camera-tamper awareness).

Cheap frame statistics on the latest pixels: a uniform frame (low std) or a
too-dark frame held for ``consecutive`` samples means the camera is likely
covered or blocked. Transitions publish ``TamperDetected`` /
``TamperCleared`` (both trigger timeline dumps, so the incident is
reconstructable). The logic is a pure class so it is unit-testable without
a camera.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from aria.core.events import Event
from aria.core.service import Service


class TamperLogic:
    """Pure state machine: feed downscaled grayscale frames, get transitions."""

    def __init__(
        self,
        std_threshold: float = 6.0,
        dark_threshold: float = 18.0,
        consecutive: int = 3,
    ) -> None:
        self.std_threshold = float(std_threshold)
        self.dark_threshold = float(dark_threshold)
        self.consecutive = int(consecutive)
        self.state = "normal"
        self._bad_streak = 0

    def update(self, gray: np.ndarray) -> Optional[str]:
        """Return 'tampered' | 'cleared' on transitions, else None."""
        if gray.size == 0:
            return None
        flat = float(np.std(gray))
        dark = float(np.mean(gray))
        bad = flat < self.std_threshold or dark < self.dark_threshold
        if bad:
            self._bad_streak += 1
            if self.state == "normal" and self._bad_streak >= self.consecutive:
                self.state = "tampered"
                return "tampered"
        else:
            self._bad_streak = 0
            if self.state == "tampered":
                self.state = "normal"
                return "cleared"
        return None

    def stats(self, gray: np.ndarray) -> Tuple[float, float]:
        return float(np.std(gray)), float(np.mean(gray))


_SCHEMA = {
    "every": (5, (int,)),
    "std_threshold": (6.0, (int, float)),
    "dark_threshold": (18.0, (int, float)),
    "consecutive": (3, (int,)),
    "width_downscale": (160, (int,)),
    "heartbeat_interval": (5.0, (int, float)),
}


class TamperService(Service):
    name = "perception.tamper"
    produces = ("TamperDetected", "TamperCleared")
    consumes = ("Frame",)
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._sub = None
        self._count = 0
        self._logic: Optional[TamperLogic] = None

    async def on_start(self) -> None:
        from aria.core.telemetry.metrics import Metrics  # noqa: F401  (metrics via self)

        from aria.perception.framestore import DEFAULT_STORE

        self._store = DEFAULT_STORE
        self._logic = TamperLogic(
            std_threshold=float(self.config.get("std_threshold", 6.0)),
            dark_threshold=float(self.config.get("dark_threshold", 18.0)),
            consecutive=int(self.config.get("consecutive", 3)),
        )
        self._sub = self.bus.subscribe("Frame", self._on_frame, policy="drop_new", maxsize=1)

    async def on_stop(self) -> None:
        if self._sub is not None:
            self.bus.unsubscribe(self._sub)
            self._sub = None

    async def _on_frame(self, event: Event) -> None:
        self._count += 1
        if self._count % int(self.config.get("every", 5)) != 0:
            return
        _, frame = self._store.latest()
        if frame is None:
            return
        import cv2  # local import keeps module import light

        width = int(self.config.get("width_downscale", 160))
        h, w = frame.shape[:2]
        scale = width / max(w, 1)
        small = cv2.resize(frame, (width, max(1, int(h * scale)))) if scale < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        transition = self._logic.update(gray)
        std, mean = self._logic.stats(gray)
        self.metrics.observe("tamper.std", std)
        if transition == "tampered":
            self.metrics.inc("tamper.detections")
            self.log.warning("Camera feed looks tampered", std=round(std, 2), mean=round(mean, 2))
            await self.bus.publish(
                Event("TamperDetected", {"std": round(std, 2), "mean": round(mean, 2)})
            )
        elif transition == "cleared":
            self.log.info("Camera feed restored", std=round(std, 2), mean=round(mean, 2))
            await self.bus.publish(
                Event("TamperCleared", {"std": round(std, 2), "mean": round(mean, 2)})
            )
