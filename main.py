"""ARIA-POC app shell (Phase 0).

Wires the runtime together: config → telemetry → EventBus (+ timeline tap)
→ Registry → Watchdog/Governor → run → graceful shutdown with a summary.
Run:  python main.py --config configs/laptop.yaml [--duration 10]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

from aria import __version__
from aria.core.config import load_config
from aria.core.event_bus import EventBus
from aria.core.registry import Registry
from aria.core.telemetry.logging import get_logger, setup_logging
from aria.core.telemetry.metrics import Metrics
from aria.core.telemetry.timeline import TimelineRecorder


async def run(cfg, duration: float, log, log_dir) -> int:
    metrics = Metrics()
    bus = EventBus(metrics=metrics, maxsize=int(cfg.get("event_bus.default_maxsize", 256)))
    timeline = TimelineRecorder(
        dir_name=str(log_dir),
        ring_size=int(cfg.get("timeline.ring_size", 600)),
        sample=cfg.get("timeline.sample", {}) or {},
    )
    timeline.install(bus)

    registry = Registry(bus, cfg)
    registry.build()

    await bus.start()
    await registry.start_all()
    log.info("ARIA-POC running", version=__version__, services=registry.names())

    # Replay runs (mic.source: file, loop: false) can end when the fixture has
    # been *processed* instead of after a guessed sleep: the audio ends, then the
    # turn is decided, transcribed and spoken, so a short grace period follows.
    # `duration` remains a safety cap for runs that never finish.
    finished = asyncio.Event()
    exit_on_end = bool(cfg.get("runtime.exit_on_source_end", False))
    grace_s = float(cfg.get("runtime.exit_after_source_s", 6.0))
    end_sub = None

    async def _on_source_finished(_event) -> None:
        finished.set()

    if exit_on_end:
        end_sub = bus.subscribe("AudioSourceFinished", _on_source_finished,
                                policy="drop_new", maxsize=1)

    try:
        if exit_on_end:
            try:
                await asyncio.wait_for(finished.wait(), timeout=duration if duration > 0 else None)
                log.info("Replay source finished; draining the last turn", grace_s=grace_s)
                await asyncio.sleep(grace_s)
            except asyncio.TimeoutError:
                log.info("Replay source did not finish before the time cap", cap_s=duration)
        elif duration > 0:
            await asyncio.sleep(duration)
        else:
            while True:  # until Ctrl+C
                await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if end_sub is not None:
            bus.unsubscribe(end_sub)
        log.info("Shutting down", restarts=registry.restart_counts)
        await registry.stop_all()
        await bus.stop()
        timeline.close()
        summary = {"metrics": metrics.snapshot(), "bus": bus.snapshot_stats()}
        print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ARIA-POC app shell")
    parser.add_argument("--config", default="configs/laptop.yaml", help="config profile path")
    parser.add_argument("--duration", type=float, default=None, help="seconds to run (default: profile)")
    parser.add_argument("--log-level", default=None, help="override profile log level")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    repo_root = pathlib.Path(__file__).resolve().parent
    log_dir = pathlib.Path(cfg.get("logging.dir", "logs"))
    if not log_dir.is_absolute():
        log_dir = repo_root / log_dir  # anchor relative paths to the repo, not the CWD
    setup_logging(
        level=args.log_level or cfg.get("logging.level", "INFO"),
        console=cfg.get("logging.console", "pretty"),
        dir_name=str(log_dir),
        max_bytes_mb=int(cfg.get("logging.max_bytes_mb", 5)),
        backups=int(cfg.get("logging.backups", 3)),
        levels=cfg.get("logging.levels", {}),
    )
    log = get_logger("aria.main")
    log.info("Booting ARIA-POC", config=cfg.source)

    duration = args.duration if args.duration is not None else float(cfg.get("runtime.duration", 0))
    try:
        return asyncio.run(run(cfg, duration, log, log_dir))
    except KeyboardInterrupt:  # belt and braces around asyncio.run teardown
        return 0


if __name__ == "__main__":
    sys.exit(main())
