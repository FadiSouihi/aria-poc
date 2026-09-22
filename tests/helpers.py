"""Shared test helpers: async runner, event collectors, hermetic services."""
from __future__ import annotations

import asyncio
import time
from typing import List

from aria.core.config import Config
from aria.core.event_bus import EventBus
from aria.core.service import Service
from aria.core.telemetry.metrics import Metrics


def run(coro):
    """Run an async test scenario to completion."""
    return asyncio.run(coro)


class Collector:
    """Async subscriber that records every event it receives."""

    def __init__(self) -> None:
        self.events: List = []

    async def __call__(self, event) -> None:
        self.events.append(event)

    async def wait_for(self, count: int, timeout: float = 2.0) -> List:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.events) >= count:
                return self.events
            await asyncio.sleep(0.001)
        raise TimeoutError(f"Collector got {len(self.events)} events, wanted {count}")


def make_bus(maxsize: int = 16, metrics: Metrics | None = None) -> EventBus:
    return EventBus(metrics=metrics or Metrics(), maxsize=maxsize)


def attach(svc: Service, bus: EventBus, config: dict | None = None) -> Service:
    svc.attach(bus, Config(config or {}, "<test>"))
    return svc


class CrashOnceService(Service):
    """Crashes exactly once on start; every later start runs steady."""

    name = "test.crash_once"
    produces = ("SceneTick",)

    def __init__(self) -> None:
        super().__init__()
        self._boomed = False

    async def on_start(self) -> None:
        if not self._boomed:
            self._boomed = True
            self.spawn(self._boom(), "boom")
        else:
            self.spawn(self._steady(), "steady")

    async def _boom(self) -> None:
        raise RuntimeError("one-shot crash")

    async def _steady(self) -> None:
        while True:
            self.last_heartbeat = time.monotonic()
            await asyncio.sleep(0.02)


class HeartbeatStopper(Service):
    """Owns its heartbeat (overrides the base loop): beats only while
    ``self.beat`` is true — set it False to simulate a hang for the watchdog."""

    name = "test.heartbeat_stopper"
    config_schema = {"heartbeat_interval": (0.05, (int, float))}

    def __init__(self) -> None:
        super().__init__()
        self.beat = True

    async def _heartbeat_loop(self) -> None:
        interval = float(self.config.get("heartbeat_interval", 0.05))
        while True:
            await asyncio.sleep(interval)
            if self.beat:
                self.last_heartbeat = time.monotonic()
