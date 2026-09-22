"""Pure frame transforms for the debug view (and, if ever needed, the source).

Mirroring/flipping is deliberately a *display* concern by default: the pipeline
should keep working on the raw sensor image so face galleries stay valid. The
same helpers are reusable at the camera source if a robot ever mounts its camera
upside down (``camera.flip`` would call :func:`apply_flip` before storing the
frame, and the whole pipeline — boxes included — would stay consistent).

All functions are pure and dependency-light so they can be unit tested without a
camera or a GUI.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

FLIP_MODES = ("none", "horizontal", "vertical", "both")


def normalize_flip(mode: str) -> str:
    """Accept friendly spellings; unknown values degrade to ``none``."""
    text = str(mode or "").strip().lower()
    aliases = {
        "": "none", "none": "none", "off": "none", "false": "none",
        "horizontal": "horizontal", "h": "horizontal", "mirror": "horizontal",
        "true": "horizontal", "on": "horizontal",
        "vertical": "vertical", "v": "vertical", "flip": "vertical",
        "both": "both", "180": "both", "rotate180": "both",
    }
    return aliases.get(text, "none")


def apply_flip(frame: np.ndarray, mode: str) -> np.ndarray:
    """Mirror/flip a BGR frame. Returns a contiguous copy (safe for the store)."""
    mode = normalize_flip(mode)
    if mode == "horizontal":
        return np.ascontiguousarray(frame[:, ::-1])
    if mode == "vertical":
        return np.ascontiguousarray(frame[::-1, :])
    if mode == "both":
        return np.ascontiguousarray(frame[::-1, ::-1])
    return frame


def flip_bbox(bbox, width: int, height: int, mode: str) -> Tuple[float, float, float, float]:
    """Map a bbox from raw-frame coordinates into the flipped frame's."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    mode = normalize_flip(mode)
    if mode in ("horizontal", "both"):
        x1, x2 = float(width) - x2, float(width) - x1
    if mode in ("vertical", "both"):
        y1, y2 = float(height) - y2, float(height) - y1
    return (x1, y1, x2, y2)


def fit_to_window(frame: np.ndarray, window_w: int, window_h: int) -> np.ndarray:
    """Scale a frame to fit ``window_w × window_h`` **preserving aspect ratio**,
    padding with black bars (letterbox) instead of stretching it.

    A 4:3 camera shown in a 16:9 window is the classic case: ``imshow`` alone
    stretches it, which is what made the debug video look distorted.
    """
    window_w, window_h = max(1, int(window_w)), max(1, int(window_h))
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    scale = min(window_w / width, window_h / height)
    new_w = max(1, min(window_w, int(round(width * scale))))
    new_h = max(1, min(window_h, int(round(height * scale))))
    if (new_w, new_h) == (width, height) and (window_w, window_h) == (width, height):
        return frame
    import cv2

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    if (new_w, new_h) == (window_w, window_h):
        return resized
    canvas = np.zeros((window_h, window_w, frame.shape[2]), dtype=frame.dtype)
    x0 = (window_w - new_w) // 2
    y0 = (window_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas