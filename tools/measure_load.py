"""Measure the app's real resource cost (CPU cores busy, RSS, GPU utilisation).

Why this exists: "it uses 90% CPU" is not actionable. This runs the app for a
fixed time, skips the start-up burst (model loading), then reports steady-state
numbers that are comparable between profiles and across optimisations:

- ``cores_busy``  : CPU-seconds consumed per wall second (1.0 = one core fully
  busy). Multiply by 100/logical_cpus for the Task-Manager-style percentage.
- ``rss_mb``      : resident memory of the process tree.
- ``gpu_util``    : nvidia-smi utilisation, sampled on the same cadence.

Usage:
    python tools/measure_load.py --config configs/laptop.yaml
    python tools/measure_load.py --config configs/audio_only.yaml --duration 20 --json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def gpu_sample() -> tuple[float, float] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        first = out.stdout.strip().splitlines()[0]
        util, mem = (part.strip() for part in first.split(","))
        return float(util), float(mem)
    except Exception:
        return None


def rss_mb(proc) -> float:
    """RSS of the whole process tree (the venv launcher re-execs python)."""
    total = proc.memory_info().rss
    try:
        for child in proc.children(recursive=True):
            if child.is_running():
                total += child.memory_info().rss
    except Exception:
        pass
    return total / (1024 * 1024)


def cpu_seconds(proc) -> float:
    """CPU time of the process and any children, so nothing is missed."""
    total = proc.cpu_times().user + proc.cpu_times().system
    try:
        for child in proc.children(recursive=True):
            if child.is_running():
                times = child.cpu_times()
                total += times.user + times.system
    except Exception:
        pass
    return total


def main() -> int:
    import psutil

    ap = argparse.ArgumentParser(description="Measure app CPU/RSS/GPU load")
    ap.add_argument("--config", default="configs/laptop.yaml")
    ap.add_argument("--duration", type=float, default=25.0, help="measurement window (s)")
    ap.add_argument("--warmup", type=float, default=12.0, help="start-up time to ignore (s)")
    ap.add_argument("--interval", type=float, default=2.0, help="sampling interval (s)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    logical = psutil.cpu_count(logical=True) or 1
    command = [str(PY), str(ROOT / "main.py"), "--config", str(args.config),
               "--duration", str(int(args.warmup + args.duration + 4))]
    proc = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ps = psutil.Process(proc.pid)
    try:
        time.sleep(args.warmup)
        samples = []
        gpu_samples = []
        start = time.monotonic()
        cpu0 = cpu_seconds(ps)
        while time.monotonic() - start < args.duration and proc.poll() is None:
            time.sleep(args.interval)
            now = time.monotonic()
            cpu_now = cpu_seconds(ps)
            cores = (cpu_now - cpu0) / max(now - start, 1e-6)
            samples.append(cores)
            gpu = gpu_sample()
            if gpu:
                gpu_samples.append(gpu)
        rss = rss_mb(ps)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    if not samples:
        print("no samples collected (app exited early?)")
        return 1
    result = {
        "config": str(args.config),
        "cores_busy_p50": round(statistics.median(samples), 2),
        "cores_busy_max": round(max(samples), 2),
        "cpu_percent_of_all": round(statistics.median(samples) / logical * 100, 1),
        "logical_cpus": logical,
        "rss_mb": round(rss, 1),
        "gpu_util_p50": round(statistics.median([g[0] for g in gpu_samples]), 1) if gpu_samples else None,
        "gpu_mem_mb_p50": round(statistics.median([g[1] for g in gpu_samples]), 0) if gpu_samples else None,
        "samples": [round(s, 2) for s in samples],
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(f"== load: {args.config} ==")
        print(f"  CPU     : {result['cores_busy_p50']:.2f} cores busy "
              f"(peak {result['cores_busy_max']:.2f}) = {result['cpu_percent_of_all']:.0f}% "
              f"of {logical} logical CPUs")
        print(f"  RSS     : {result['rss_mb']:.0f} MB")
        print(f"  GPU     : {result['gpu_util_p50']}% util, {result['gpu_mem_mb_p50']} MB VRAM")
        print(f"  samples : {result['samples']}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())