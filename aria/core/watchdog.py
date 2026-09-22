"""Watchdog: supervises services and restarts crashed/hung ones in place.

Fixes the current robot's failure mode (NFR-03 Step 6: "camera crash needs a
full restart of main.py"): a crashed service publishes ``ServiceCrashed``,
the watchdog restarts it after a bounded backoff, and ``ServiceRestarted`` is
published. Restarts per service are capped by ``max_restarts``; giving up
raises a generic ``Anomaly`` (which triggers a timeline dump).

Note: the watchdog supervises every registry service except itself — if the
watchdog process itself dies, the app shell surfaces it at shutdown.
"""
from __future__ import annotations

import asyncio
import time
from typing import Tuple

from aria.core.events import Event
from aria.core.service import CRASHED, RUNNING, Service
from aria.core.telemetry.logging import get_logger

_SCHEMA = {
    "interval": (1.0, (int, float)),
    "stale_after": (5.0, (int, float)),
    "stale_safety_factor": (3.0, (int, float)),
    "max_restarts": (5, (int,)),
    "backoff": (0.5, (int, float)),
    "heartbeat_interval": (5.0, (int, float)),
}


class WatchdogService(Service):
    name = "core.watchdog"
    produces: Tuple[str, ...] = ("ServiceCrashed", "ServiceRestarted", "Anomaly")
    consumes: Tuple[str, ...] = ()
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._registry = None

    def set_registry(self, registry) -> None:
        self._registry = registry

    async def on_start(self) -> None:
        if self._registry is None:
            raise RuntimeError("WatchdogService requires set_registry(registry) before start")
        self.spawn(self._supervise_loop(), "supervise")

    async def _supervise_loop(self) -> None:
        interval = float(self.config.get("interval", 1.0))
        while True:
            await asyncio.sleep(interval)
            try:
                await self._check_all()
            except Exception:
                self.log.exception("Watchdog check cycle failed")

    async def _check_all(self) -> None:
        stale_after = float(self.config.get("stale_after", 5.0))
        safety = float(self.config.get("stale_safety_factor", 3.0))
        max_restarts = int(self.config.get("max_restarts", 5))
        backoff = float(self.config.get("backoff", 0.5))
        now = time.monotonic()

        for name, svc in list(self._registry.services.items()):
            if svc is self:
                continue
            crashed = svc.state == CRASHED
            # A service is only "stale" well past its own heartbeat period.
            # Using the global threshold alone meant a 5 s heartbeat against a
            # 5 s threshold — zero margin, so heavy work elsewhere (Whisper
            # saturating the GPU) caused spurious restarts.
            period = float(getattr(svc, "_hb_interval", 0.0) or 0.0)
            threshold = max(stale_after, period * safety) if period > 0 else stale_after
            delay = now - svc.last_heartbeat
            stale = svc.state == RUNNING and delay > threshold
            if not (crashed or stale):
                continue

            restarts = self._registry.restart_counts.get(name, 0)
            if restarts >= max_restarts:
                self.log.error("Restart budget exhausted; giving up on service", service=name)
                await self.bus.publish(
                    Event("Anomaly", {"what": "restart_budget_exhausted", "service": name})
                )
                continue

            reason = "task_crashed" if crashed else "heartbeat_stale"
            self.log.warning("Service unhealthy; restarting", service=name, reason=reason,
                             delayed_s=round(delay, 2), threshold_s=round(threshold, 2))
            await self.bus.publish(Event("ServiceCrashed", {"service": name, "reason": reason}))
            await asyncio.sleep(backoff)
            await self._registry.restart(name, reason=reason)
            await self.bus.publish(Event("ServiceRestarted", {"service": name, "reason": reason}))
