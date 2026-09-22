"""Vision benchmark — Phase 1 exit criterion "bench_vision within budget".

Runs the real stack (camera source + YOLO tracker service) for ``--seconds``
via a generated headless profile, then reports detection latency, effective
Hz, VRAM (CUDA) and tracks seen — with an optional hard budget assert.

    python tools/bench_vision.py --source file --path fixtures/bus.jpg --seconds 8 --assert
    python tools/bench_vision.py --source webcam --seconds 20 --assert
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def bench_profile(args) -> dict:
    """Config for a headless bench run (camera+vision only). safe_dump keeps
    Windows paths YAML-safe; `extends` gets an absolute path (pathlib join)."""
    return {
        "extends": str(ROOT / "configs" / "laptop.yaml"),
        "logging": {"level": "INFO", "dir": "logs_bench", "console": "quiet"},
        "services": {
            "camera": {
                "enabled": True,
                "source": args.source,
                "path": str(ROOT / args.path) if args.path else "",
                "device": args.device,
                "fps": args.fps,
            },
            "vision": {"enabled": True, "warmup": False, "detect_hz": args.detect_hz},
            "face": {"enabled": False},
            "display": {"enabled": False},
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Bench the vision stack")
    ap.add_argument("--source", default="file", choices=["file", "webcam"])
    ap.add_argument("--path", default="fixtures/bus.jpg")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--detect-hz", type=float, default=15.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--max-vram-mb", type=float, default=2600.0)
    ap.add_argument("--assert", dest="assert_budget", action="store_true")
    args = ap.parse_args()

    with tempfile_dir() as tmp:
        cfg_path = tmp / "bench.yaml"
        cfg_path.write_text(yaml.safe_dump(bench_profile(args), sort_keys=False), encoding="utf-8")
        proc = subprocess.run(
            [str(PY), str(ROOT / "main.py"), "--config", str(cfg_path), "--duration", str(args.seconds)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
        )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        return 1
    summary = json.loads(proc.stdout.strip())
    counters = summary["metrics"]["counters"]
    hist = summary["metrics"]["histograms"]

    lat = hist.get("vision.detect_latency_s") or {}
    frames = counters.get("camera.frames", 0)
    ticks = counters.get("scene.ticks", 0)
    vram_hist = hist.get("vision.vram_used_mb") or {}
    vram_peak = vram_hist.get("max") if vram_hist.get("count", 0) > 0 else None

    print("== bench_vision ==")
    print(f"source           : {args.source} {args.path or args.device}")
    print(f"camera frames    : {frames}  ({frames / args.seconds:.1f} fps)")
    print(f"detect cycles    : {ticks}  ({ticks / args.seconds:.1f} Hz)")
    if lat.get("count", 0) > 0:
        print(f"detect latency   : p50 {lat['p50'] * 1000:.0f} ms  p95 {lat['p95'] * 1000:.0f} ms")
    if vram_peak is not None:
        print(f"vram allocated   : peak {vram_peak:.0f} MB (budget {args.max_vram_mb:.0f} MB)")
    else:
        print("vram allocated   : n/a (CPU device)")

    failures = []
    if ticks <= 0:
        failures.append("no detection cycles ran")
    if vram_peak is not None and vram_peak > args.max_vram_mb:
        failures.append(f"VRAM peak {vram_peak:.0f} MB exceeds budget {args.max_vram_mb:.0f} MB")
    if failures:
        print("FAIL: " + "; ".join(failures))
        return 1
    print("OK: within budget")
    return 0


class tempfile_dir:
    def __enter__(self):
        import tempfile

        self.path = pathlib.Path(tempfile.mkdtemp(prefix="aria-bench-"))
        return self.path

    def __exit__(self, *exc):
        import shutil

        shutil.rmtree(self.path, ignore_errors=True)
        return False


if __name__ == "__main__":
    sys.exit(main())