"""Service base class: lifecycle, health, heartbeat, crash containment.

Every capability in the system is a Service. The base class owns:
- lifecycle state machine: new → initialized → starting → running → stopping
  → stopped (or crashed from anywhere);
- a heartbeat task so the watchdog can detect hangs (§6.1);
- declared config schema: defaults are filled, unknown keys are dropped with a
  warning, wrong types are rejected early;
- crash containment: a failing spawned task flips the service to ``crashed``
  and publishes ``ServiceCrashed`` — the app itself keeps running.

Subclasses implement ``on_start``/``on_stop`` (and optionally ``init``) and
spawn background loops with ``self.spawn(coro, name)``.
"""
from __future__ import annotations

import asyncio
import time
import typing
from typing import Any, ClassVar, Coroutine, Dict, Optional, Tuple

from aria.core.config import Config
from aria.core.events import Event
from aria.core.telemetry.logging import get_logger
from aria.core.telemetry.metrics import Metrics

NEW = "new"
INITIALIZED = "initialized"
STARTING = "starting"
RUNNING = "running"
STOPPING = "stopping"
STOPPED = "stopped"
CRASHED = "crashed"


class Health(typing.NamedTuple):
    status: str  # "ok" | "degraded" | "down"
    detail: str


class Service:
    name: ClassVar[str] = "service"
    produces: ClassVar[Tuple[str, ...]] = ()
    consumes: ClassVar[Tuple[str, ...]] = ()
    # key -> (default, allowed types). None in the types tuple means optional.
    config_schema: ClassVar[Dict[str, Tuple[Any, Tuple[type, ...]]]] = {}

    def __init__(self) -> None:
        self.bus: Optional["object"] = None  # EventBus; typed loosely to avoid a cycle
        self.metrics = Metrics()  # replaced by the bus's shared Metrics at attach
        self.config: Optional[Config] = None
        self.state: str = NEW
        self.last_heartbeat: float = 0.0
        self._tasks: list = []
        self._hb_task: Optional[asyncio.Task] = None
        self._hb_interval: float = 5.0
        self.log = get_logger(f"aria.unattached.{type(self).__name__}")

    # -- wiring ----------------------------------------------------------
    def attach(self, bus, config: Config) -> None:
        self.bus = bus
        self.metrics = getattr(bus, "metrics", None) or Metrics()
        self.config = Config(self._apply_schema(config.raw), source=config.source)
        self._hb_interval = float(self.config.get("heartbeat_interval", 5.0))
        self.log = get_logger(f"aria.{self.name}")

    def _apply_schema(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in raw.items():
            if key not in self.config_schema:
                self.log.warning("Unknown config key dropped", key=key, service=self.name)
                continue
            out[key] = value
        for key, (default, allowed) in self.config_schema.items():
            if key in out:
                value = out[key]
                if allowed and not isinstance(value, allowed):
                    raise TypeError(
                        f"Config key '{key}' for service '{self.name}' must be "
                        f"{allowed}, got {type(value).__name__}"
                    )
            else:
                out[key] = default
        return out

    # -- lifecycle -------------------------------------------------------
    async def init(self) -> None:
        """One-time setup hook (called on every (re)start)."""

    async def start(self) -> None:
        if self.state == RUNNING:
            return
        if self.state in (NEW, CRASHED, STOPPED):
            await self.init()
            self.state = INITIALIZED
        self.state = STARTING
        self.last_heartbeat = time.monotonic()  # healthy until proven otherwise
        self._hb_task = self.spawn(self._heartbeat_loop(), "heartbeat")
        await self.on_start()
        self.state = RUNNING
        self.log.info("Service started", service=self.name)

    async def stop(self) -> None:
        if self.state in (STOPPED, NEW):
            self.state = STOPPED
            return
        self.state = STOPPING
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._hb_task = None
        try:
            await self.on_stop()
        except Exception:
            self.log.exception("on_stop failed", service=self.name)
        self.state = STOPPED
        self.log.info("Service stopped", service=self.name)

    async def on_start(self) -> None:
        """Override: spawn background loops here."""

    async def on_stop(self) -> None:
        """Override: release resources here."""

    def health(self) -> Health:
        if self.state == CRASHED:
            return Health("down", "task crashed")
        if self.state == RUNNING:
            stale_for = time.monotonic() - self.last_heartbeat
            if stale_for > self._hb_interval * 3 + 1.0:
                return Health("degraded", f"heartbeat stale for {stale_for:.1f}s")
            return Health("ok", "")
        return Health("degraded", f"state={self.state}")

    # -- helpers for subclasses ------------------------------------------
    def spawn(self, coro: Coroutine, name: str) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro, name=f"{self.name}:{name}")
        self._tasks.append(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task) -> None:
        try:
            self._tasks.remove(task)
        except ValueError:
            pass
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        self.state = CRASHED
        self.log.exception("Service task crashed", service=self.name, task=task.get_name())
        if self.bus is not None:
            try:
                asyncio.get_running_loop().create_task(
                    self.bus.publish(
                        Event(
                            "ServiceCrashed",
                            {"service": self.name, "reason": "task_crashed", "error": repr(exc)},
                        )
                    )
                )
            except RuntimeError:  # loop already closing
                pass

    async def _heartbeat_loop(self) -> None:
        self.last_heartbeat = time.monotonic()
        while True:
            await asyncio.sleep(self._hb_interval)
            self.last_heartbeat = time.monotonic()
