"""FacePipelineService — detection + embeddings + gallery voting + re-verification.

Fixes the FUNC-06 false-positive class and the NFR-07 re-verification gap:
- YuNet (opencv_zoo) detects faces with 5 landmarks;
- an ONNX embedder (default: SFace 128-d from opencv_zoo — the public
  choice; ArcFace r50 is a config swap behind the same interface) embeds the
  aligned 112×112 crop;
- a track only *becomes* an identity after ``vote_frames`` consecutive
  consistent matches that beat the runner-up by ``margin`` (multi-frame
  voting), and matched identities are re-verified every ``reverify_s``.

Galleries live in ``data/faces/<name>.npz`` (built by ``tools/enroll.py``).
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from aria.core.events import Event
from aria.core.service import Service
from aria.perception.vision import resolve_repo_path

# ArcFace-style canonical 5-point template (112×112).
DST5 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [45.0122, 92.3655], [64.6567, 92.2041]], dtype=np.float32
)

_SCHEMA = {
    "yunet_model": ("weights/face_detection_yunet_2023mar.onnx", (str,)),
    "embedder_model": ("weights/face_recognition_sface_2021dec.onnx", (str,)),
    "score": (0.60, (int, float)),
    "pace_every": (3, (int,)),
    "match_threshold": (0.363, (int, float)),   # opencv_zoo SFace same-person threshold
    "vote_frames": (3, (int,)),
    "margin": (0.03, (int, float)),
    "reverify_s": (5.0, (int, float)),
    "faces_dir": ("data/faces", (str,)),
    "process_width": (480, (int,)),
    "ort_threads": (1, (int,)),        # SFace embedding is small and infrequent
    "ort_spinning": (False, (bool,)),
    "opencv_threads": (2, (int,)),     # cap cv2 parallel loops (YuNet DNN, resize)
    "heartbeat_interval": (5.0, (int, float)),
}


def align_crop(img_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
    """5-point similarity transform to the canonical 112×112 template."""
    import cv2

    src = np.asarray(landmarks5, dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, DST5)
    if matrix is None:
        raise ValueError("alignment failed")
    return cv2.warpAffine(img_bgr, matrix, (112, 112))


class SFaceEmbedder:
    """ONNX SFace: 112×112 aligned BGR crop → L2-normalized 128-d vector."""

    def __init__(self, model_path: str, threads: Optional[int] = None,
                 spinning: Optional[bool] = None) -> None:
        import onnxruntime as ort

        from aria.core.onnx import make_session

        ort.set_default_logger_severity(3)  # silence per-initializer warnings
        self.session = make_session(model_path, kind="face", threads=threads, spinning=spinning)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, aligned_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2_cvtColor(aligned_bgr)
        blob = ((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]
        emb = self.session.run(None, {self.input_name: blob})[0].reshape(-1)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


def cv2_cvtColor(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FaceGallery:
    """Enrolled identities: name → L2-normalized centroid (pure logic)."""

    def __init__(self) -> None:
        self.centroids: dict = {}

    def load(self, faces_dir: Path) -> "FaceGallery":
        faces_dir = Path(faces_dir)
        if faces_dir.exists():
            for npz in sorted(faces_dir.glob("*.npz")):
                data = np.load(npz)
                self.centroids[npz.stem] = np.asarray(data["emb"], dtype=np.float32)
        return self

    def match(self, emb: np.ndarray) -> Tuple[Optional[str], float, float]:
        """Return (best_name, best_sim, runner_up_sim) — None if empty."""
        if not self.centroids:
            return None, -1.0, -1.0
        scored = sorted(
            ((name, float(np.dot(emb, c))) for name, c in self.centroids.items()),
            key=lambda pair: pair[1], reverse=True,
        )
        best_name, best_sim = scored[0]
        runner = scored[1][1] if len(scored) > 1 else -1.0
        return best_name, best_sim, runner

    def __len__(self) -> int:
        return len(self.centroids)


def consistent_vote(votes, vote_frames: int) -> Optional[Tuple[str, float]]:
    """An identity wins when the last ``vote_frames`` votes agree on one name.

    votes: list of (name|None, similarity). Returns (name, best_sim) or None.
    Pure logic so the voting rule is unit-testable without any model.
    """
    if len(votes) < vote_frames:
        return None
    tail = list(votes)[-vote_frames:]
    names = {v for v, _ in tail}
    if len(names) != 1:
        return None
    name = next(iter(names))
    if name is None:
        return None
    return name, max(s for _, s in tail)


class FacePipelineService(Service):
    name = "perception.face"
    produces = ("FaceMatched",)
    consumes = ("SceneTick",)
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._sub = None
        self._detector = None
        self._embedder = None
        self._gallery = FaceGallery()
        self._tick_count = 0
        self._last_tracks: list = []          # from the last SceneTick
        self._track_state: dict = {}          # track_id -> {"votes": deque, "matched", "last_verify"}

    # -- setup -----------------------------------------------------------
    async def init(self) -> None:
        import cv2
        from aria.perception.framestore import DEFAULT_STORE

        self._store = DEFAULT_STORE
        from aria.core.onnx import limit_opencv_threads

        limit_opencv_threads(self.config.get("opencv_threads"))
        yunet_path = resolve_repo_path(str(self.config.get("yunet_model", "")))
        embed_path = resolve_repo_path(str(self.config.get("embedder_model", "")))
        if not yunet_path.exists() or not embed_path.exists():
            raise FileNotFoundError(f"face models missing: {yunet_path}, {embed_path}")
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet_path), "", (320, 240),
            score_threshold=float(self.config.get("score", 0.60)),
            nms_threshold=0.3, top_k=10,
        )
        self._embedder = SFaceEmbedder(str(embed_path),
                                       threads=self.config.get("ort_threads"),
                                       spinning=self.config.get("ort_spinning"))
        faces_dir = resolve_repo_path(str(self.config.get("faces_dir", "data/faces")))
        self._gallery.load(faces_dir)
        self.log.info("Face pipeline ready", gallery_size=len(self._gallery), embedder="sface")

    # -- lifecycle -------------------------------------------------------
    async def on_start(self) -> None:
        self._sub = self.bus.subscribe("SceneTick", self._on_scenetick, policy="drop_oldest", maxsize=2)

    async def on_stop(self) -> None:
        if self._sub is not None:
            self.bus.unsubscribe(self._sub)
            self._sub = None

    # -- pipeline --------------------------------------------------------
    async def _on_scenetick(self, event: Event) -> None:
        self._last_tracks = event.payload.get("tracks", [])
        self._tick_count += 1
        pace = max(int(self.config.get("pace_every", 3)), 1)
        if self._tick_count % pace != 0:
            return
        fid, frame = self._store.latest()
        if frame is None:
            return
        await asyncio.to_thread(self._process, frame)

    def _detect_faces(self, frame: np.ndarray):
        import cv2

        width = int(self.config.get("process_width", 480))
        h, w = frame.shape[:2]
        scale = min(1.0, width / max(w, 1))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        self._detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self._detector.detect(frame)
        return frame, (faces if faces is not None else []), scale

    def _process(self, frame: np.ndarray) -> None:
        # synchronous face pass (runs in a worker thread)
        proc, faces, scale = self._detect_faces(frame)
        now = time.monotonic()
        vote_frames = int(self.config.get("vote_frames", 3))
        threshold = float(self.config.get("match_threshold", 0.363))
        margin = float(self.config.get("margin", 0.03))
        reverify_s = float(self.config.get("reverify_s", 5.0))

        used_faces = set()
        for track in self._last_tracks:
            tid = track["id"]
            bx1, by1, bx2, by2 = (int(v / max(scale, 1e-6)) for v in track["bbox"]) if scale < 1.0 else track["bbox"]
            state = self._track_state.setdefault(
                tid, {"votes": deque(maxlen=vote_frames), "matched": None, "last_verify": 0.0}
            )
            # associate: best-score face whose center sits inside the person box
            best = None
            for i, f in enumerate(faces):
                if i in used_faces:
                    continue
                x, y, fw, fh = f[0], f[1], f[2], f[3]
                cx, cy = x + fw / 2, y + fh / 2
                if scale < 1.0:
                    cx, cy = cx / scale, cy / scale
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    if best is None or f[-1] > faces[best][-1]:
                        best = i
            if best is None:
                state["votes"].append((None, -1.0))
                continue
            used_faces.add(best)
            f = faces[best]
            landmarks = np.array(f[4:14], dtype=np.float32)
            score = float(f[-1])
            try:
                aligned = align_crop(proc, landmarks)
                emb = self._embedder(aligned)
            except Exception:
                state["votes"].append((None, -1.0))
                continue
            self.metrics.inc("face.embeds")
            name, sim, runner = self._gallery.match(emb)
            ok = name is not None and sim >= threshold and sim - runner >= margin
            state["votes"].append((name if ok else None, sim if ok else -1.0))
            self._maybe_publish_match(tid, state, vote_frames, reason="vote")
            self._maybe_reverify(tid, state, emb, now, reverify_s, threshold)

    def _maybe_publish_match(self, tid: int, state: dict, vote_frames: int, reason: str) -> None:
        won = consistent_vote(state["votes"], vote_frames)
        if won is None:
            return
        name, best_sim = won
        if state["matched"] != name:
            state["matched"] = name
            state["last_verify"] = time.monotonic()
            self.metrics.inc("face.matches")
            self.log.info("Identity matched", track_id=tid, identity=name,
                          similarity=round(best_sim, 3), votes=vote_frames, reason=reason)
            asyncio.get_running_loop().create_task(
                self.bus.publish(Event("FaceMatched", {
                    "track_id": tid, "identity": name,
                    "similarity": round(best_sim, 3), "votes": vote_frames, "reason": reason,
                }))
            )

    def _maybe_reverify(self, tid: int, state: dict, emb: np.ndarray, now: float,
                        reverify_s: float, threshold: float) -> None:
        if state["matched"] is None:
            return
        if now - state["last_verify"] < reverify_s:
            return
        state["last_verify"] = now
        name, sim, _ = self._gallery.match(emb)
        if name != state["matched"] or sim < threshold * 0.9:
            self.log.warning("Identity re-verification failed; dropping match",
                             track_id=tid, was=state["matched"], now=name, similarity=round(sim, 3))
            state["matched"] = None
            state["votes"].clear()
