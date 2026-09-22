"""ONNX Runtime session plumbing — one place, so threading is tunable per model.

Why this matters for a laptop/Jetson: ONNX Runtime's defaults use one thread per
physical core **and spin-wait** for work. A tiny model called at a fixed cadence
(Silero VAD runs ~31 times per second) therefore shows up as a large CPU load
even though the actual math is microseconds — the threads are busy spinning, not
computing. The default also makes the reported CPU% of the whole app misleading.

Rules of thumb used here:

- tiny + frequent (VAD): 1 thread, no spinning — latency budget per call is the
  32 ms frame, so a single thread is plenty.
- small + per-turn (Smart Turn, SFace): 1-2 threads, no spinning.
- heavy + per-turn (WavLM x-vectors): a few threads help *latency*, but spinning
  still wastes the whole time between turns, so it stays off.

Every service takes ``ort_threads`` / ``ort_spinning`` from its YAML config, so
this is tunable without touching models or code paths.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

DEFAULTS = {
    "vad": {"threads": 1, "spinning": False},
    "turn": {"threads": 1, "spinning": False},
    "face": {"threads": 1, "spinning": False},
    "voiceprint": {"threads": 2, "spinning": False},
    "default": {"threads": 1, "spinning": False},
}


def default_threads(kind: str) -> int:
    return int(DEFAULTS.get(kind, DEFAULTS["default"])["threads"])


def default_spinning(kind: str) -> bool:
    return bool(DEFAULTS.get(kind, DEFAULTS["default"])["spinning"])


def make_session_options(threads: Optional[int] = None, spinning: Optional[bool] = None,
                         kind: str = "default"):
    """Build ``ort.SessionOptions`` with sane, low-noise threading."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    count = default_threads(kind) if threads is None else max(1, int(threads))
    options.intra_op_num_threads = count
    options.inter_op_num_threads = 1              # these graphs are single-op chains
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    allow_spin = default_spinning(kind) if spinning is None else bool(spinning)
    try:
        options.add_session_config_entry("session.intra_op.allow_spinning", "1" if allow_spin else "0")
    except Exception:
        pass
    try:
        options.add_session_config_entry("session.inter_op.allow_spinning", "1" if allow_spin else "0")
    except Exception:
        pass
    return options


def make_session(model_path: str, kind: str = "default", threads: Optional[int] = None,
                 spinning: Optional[bool] = None,
                 providers: Optional[Sequence[str]] = None):
    """Create an InferenceSession with low-noise threading.

    Providers default to CPU; pass ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
    for models that should run on the GPU.
    """
    import onnxruntime as ort

    options = make_session_options(threads=threads, spinning=spinning, kind=kind)
    chosen: List[str] = list(providers) if providers else ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    filtered = [p for p in chosen if p in available] or ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=options, providers=filtered)


def limit_opencv_threads(threads: Optional[int] = None) -> int:
    """Cap OpenCV's own parallel loops (DNN inference, resize) — same reasoning.

    Reads ``ARIA_OPENCV_THREADS`` so a service can set it from config without a
    global import graph.
    """
    import cv2

    count = threads
    if count is None:
        raw = os.environ.get("ARIA_OPENCV_THREADS", "")
        count = int(raw) if raw.isdigit() and int(raw) > 0 else 2
    cv2.setNumThreads(max(1, int(count)))
    return cv2.getNumThreads()