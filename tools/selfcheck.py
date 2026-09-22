"""One-command health check for ARIA-POC.

    python tools/selfcheck.py

Verifies, in order:
  1. The full contract test suite (pytest);
  2. A real boot of the app shell (hermetic test profile, 3 seconds);
  3. That frames actually flowed end-to-end (produced > 0, consumed > 0);
  4. That no events were dropped at this load;
  5. That telemetry artifacts exist and parse (JSON log lines + timeline).

Exit code 0 = everything works; 1 = at least one check failed.
Use it any time you want proof the skeleton is healthy.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    print(f"ARIA-POC self-check (python {sys.version.split()[0]})")

    # 1. Contract test suite
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
    )
    tail = (proc.stdout or "").strip().splitlines()
    check("contract test suite", proc.returncode == 0, tail[-1] if tail else "")

    # 2. Real boot of the app shell (test profile → quiet console, pure JSON summary)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "main.py"),
         "--config", str(ROOT / "configs" / "test.yaml"), "--duration", "3"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
    )
    check("app shell boots and shuts down cleanly", proc.returncode == 0,
          "" if proc.returncode == 0 else "see stderr below")

    summary = {}
    try:
        summary = json.loads((proc.stdout or "").strip())
    except json.JSONDecodeError:
        pass
    counters = summary.get("metrics", {}).get("counters", {})
    histograms = summary.get("metrics", {}).get("histograms", {})
    bus = summary.get("bus", {})
    frames = counters.get("camera.frames", 0)
    consumed = histograms.get("tamper.std", {}).get("count", 0)  # pixel-consumer proof
    published = bus.get("published", 0)
    dropped = bus.get("dropped", -1)

    # 3. End-to-end event flow
    check("events flow end-to-end (camera -> bus -> pixel consumer)",
          frames > 0 and consumed > 0 and published > 0,
          f"frames={frames} consumed={consumed} published={published}")

    # 4. Backpressure at this load
    check("no dropped events at this load", dropped == 0, f"dropped={dropped}")

    # 5. Telemetry artifacts
    log_dir = ROOT / "logs_test"
    log_path = log_dir / "aria.log"
    parse_ok = False
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            try:
                rec = json.loads(lines[-1])
                parse_ok = all(k in rec for k in ("ts", "level", "service", "msg"))
            except json.JSONDecodeError:
                parse_ok = False
    check("structured JSON log written and parseable", parse_ok, str(log_path.relative_to(ROOT)))

    timeline_path = log_dir / "timeline_full.jsonl"
    tl_lines = 0
    if timeline_path.exists():
        tl_lines = len(timeline_path.read_text(encoding="utf-8").strip().splitlines())
    check("event timeline recorded", tl_lines > 0, f"{tl_lines} events")

    # 6. Audio model weights (cheap presence check; Phase 2)
    weights = ROOT / "weights"
    needed = {
        "silero_vad.onnx": weights / "silero_vad.onnx",
        "smart-turn-v3.2-cpu.onnx": weights / "smart-turn-v3.2-cpu.onnx",
        "wavlm-base-plus-sv.onnx": weights / "wavlm-base-plus-sv.onnx",
        "whisper-turbo-ct2/model.bin": weights / "whisper-turbo-ct2" / "model.bin",
    }
    missing = [name for name, path in needed.items() if not path.exists()]
    check("audio model weights present", not missing,
          "all present" if not missing else "missing: " + ", ".join(missing))

    # 7. (optional, --full) vision benchmark on the bundled fixture
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        bench = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "bench_vision.py"),
             "--source", "file", "--path", "fixtures/bus.jpg", "--seconds", "8", "--assert"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
        )
        tail = (bench.stdout or "").strip().splitlines()
        detail = next((l for l in tail if "vram" in l.lower() or "latency" in l.lower()), "")
        check("vision benchmark within budget (--full)", bench.returncode == 0, detail)
        if bench.returncode != 0:
            print((bench.stdout or "")[-1500:])
            print((bench.stderr or "")[-800:])

        # 8. (optional, --full) audio pipeline benchmark on the replay fixtures
        abench = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "bench_audio.py"),
             "--manifest", "fixtures/audio/manifest.yaml", "--assert"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
        )
        atail = (abench.stdout or "").strip().splitlines()
        adetail = next((l for l in atail if "false-response" in l.lower()), "")
        check("audio benchmark within budget (--full)", abench.returncode == 0, adetail)
        if abench.returncode != 0:
            print((abench.stdout or "")[-2000:])
            print((abench.stderr or "")[-800:])
    else:
        print("hint: python tools/selfcheck.py --full also benches the vision and audio stacks")

    if proc.returncode != 0:
        print("\n--- app shell stderr ---")
        print((proc.stderr or "")[-1500:])

    all_ok = all(ok for _, ok, _ in RESULTS)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'} "
          f"({sum(1 for _, ok, _ in RESULTS if ok)}/{len(RESULTS)})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
