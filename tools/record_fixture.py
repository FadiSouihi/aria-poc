"""Record a replayable camera fixture.

    python tools/record_fixture.py --name single --seconds 30
    python tools/record_fixture.py --name group --seconds 30 --device 0
    python tools/record_fixture.py --name office_walk --source file --path some.mp4

Writes ``fixtures/<name>/video.mp4`` + ``fixtures/<name>/meta.yaml`` so
profiles and tests can replay it with ``camera.source: file``. This is how
scenario fixtures (single, group, handoff, occlusion) get captured — no live
people required for later test runs.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import yaml  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Record a camera fixture (video + meta.yaml)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--source", default="webcam", choices=["webcam", "file"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--path", default="", help="source video (when --source file)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    out_dir = ROOT / "fixtures" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4"
    meta_path = out_dir / "meta.yaml"

    if args.source == "file":
        cap = cv2.VideoCapture(args.path)
    else:
        cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"ERROR: cannot open source", file=sys.stderr)
        return 1

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (actual_w, actual_h))
    stop = threading.Event()
    frames_written = 0
    t0 = time.time()
    print(f"Recording '{args.name}': {actual_w}x{actual_h} @ {args.fps}fps for {args.seconds}s — Ctrl+C to stop")
    while not stop.is_set() and (time.time() - t0) < args.seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            if args.source == "file":
                break  # source ended
            continue
        writer.write(frame)
        frames_written += 1
    cap.release()
    writer.release()
    duration = round(time.time() - t0, 2)

    meta = {
        "name": args.name,
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": args.source,
        "device": args.device if args.source == "webcam" else None,
        "origin_path": args.path or None,
        "fps": args.fps,
        "width": actual_w,
        "height": actual_h,
        "frames": frames_written,
        "duration_s": duration,
        "notes": args.notes,
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"OK: {video_path} ({frames_written} frames, {duration}s) + {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())