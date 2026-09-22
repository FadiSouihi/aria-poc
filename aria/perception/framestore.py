"""In-process pixel exchange between frame producers and consumers.

The bus carries only metadata (Frame events stay JSON/timeline-friendly);
pixels travel through here. Single writer (CameraService), many readers,
bounded memory (last K frames). Consumers grab the latest at their own pace —
this is what makes paced inference safe under load: vision runs at
``detect_hz``, the face pipeline at a fraction of that, the display at
whatever the window pump allows, all from the same buffer.

Phase 0 kept a module-level default store (``DEFAULT_STORE``) for the
in-process POC; services that need pixels import it explicitly.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np


class FrameStore:
    def __init__(self, capacity: int = 4) -> None:
        self._lock = threading.Lock()
        self._latest_id: Optional[int] = None
        self._frames: Dict[int, np.ndarray] = {}
        self._order: Deque[int] = deque(maxlen=capacity)

    def put(self, frame_id: int, frame: np.ndarray) -> None:
        with self._lock:
            if frame_id not in self._frames:
                self._order.append(frame_id)
            self._frames[frame_id] = frame
            evict = [fid for fid in self._frames if fid not in self._order]
            for fid in evict:
                self._frames.pop(fid, None)
            self._latest_id = frame_id

    def latest(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        with self._lock:
            if self._latest_id is None:
                return None, None
            frame = self._frames.get(self._latest_id)
            return self._latest_id, None if frame is None else frame

    def get(self, frame_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._frames.get(frame_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


DEFAULT_STORE = FrameStore()
