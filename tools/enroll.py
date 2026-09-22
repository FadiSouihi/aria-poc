"""Enroll a face into the recognition gallery.

    python tools/enroll.py --name fedi                 # live webcam, 40 frames
    python tools/enroll.py --name visitor1 --source file --path fixtures/single/video.mp4

Runs YuNet + the configured embedder over N frames, keeps the best
(median-margin) embeddings, averages them into a centroid and writes
``data/faces/<name>.npz``. Re-run the same name to re-enroll (overwrite).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from aria.perception.face import align_crop, DST5  # noqa: E402
from aria.perception.vision import resolve_repo_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Enroll a face identity")
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", default="webcam", choices=["webcam", "file"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--path", default="")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--score", type=float, default=0.6)
    ap.add_argument("--embedder", default=str(ROOT / "weights" / "face_recognition_sface_2021dec.onnx"))
    ap.add_argument("--yunet", default=str(ROOT / "weights" / "face_detection_yunet_2023mar.onnx"))
    args = ap.parse_args()

    from aria.perception.face import SFaceEmbedder

    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW) if args.source == "webcam" else cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print("ERROR: cannot open source", file=sys.stderr)
        return 1
    detector = cv2.FaceDetectorYN.create(args.yunet, "", (320, 240), score_threshold=args.score, top_k=10)
    embedder = SFaceEmbedder(args.embedder)

    embeddings = []
    seen = 0
    while len(embeddings) < args.frames and seen < args.frames * 20:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        seen += 1
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            continue
        # biggest face = the person we're enrolling
        face = max(faces, key=lambda f: f[2] * f[3])
        try:
            aligned = align_crop(frame, np.array(face[4:14], dtype=np.float32))
            embeddings.append(embedder(aligned))
        except ValueError:
            continue
    cap.release()

    if len(embeddings) < 10:
        print(f"ERROR: only {len(embeddings)} usable faces (need ≥10) — better light/angle?", file=sys.stderr)
        return 1

    emb = np.mean(np.stack(embeddings), axis=0)
    emb = emb / max(np.linalg.norm(emb), 1e-9)
    out = ROOT / "data" / "faces" / f"{args.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=emb, count=len(embeddings))
    print(f"Enrolled '{args.name}': {len(embeddings)} frames → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())