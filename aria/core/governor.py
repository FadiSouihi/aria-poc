"""Resource governor: samples memory and publishes pacing decisions.

Phase-0 scope (ROADMAP §4.2): process RSS sampling with hysteresis. When RSS
crosses ``ram_throttle_mb`` the level flips to ``throttled`` and a
``GovernorThrottled`` event is published; it returns to ``normal`` once RSS
stays under ``ram_release_mb``. Consumers (camera pacing, model placement in
later phases) subscribe to that event — the governor itself stays decoupled.

VRAM sampling (NVML) lands in Phase 1 with the first GPU models.
"""
from __future__ import annotations

import asyncio
import time
from typing import Tuple

import psutil

from aria.core.events import Event
from aria.core.service import Service
from aria.core.telemetry.logging import get_logger

MB = 1024 * 1024

_SCHEMA = {
    "interval": (5.0, (int, float)),
    "ram_throttle_mb": (3500.0, (int, float)),
    "ram_release_mb": (2800.0, (int, float)),
    "heartbeat_interval": (5.0, (int, float)),
}


class GovernorService(Service):
    name = "core.governor"
    produces: Tuple[str, ...] = ("GovernorThrottled",)
    consumes: Tuple[str, ...] = ()
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self.level: str = "normal"

    async def on_start(self) -> None:
        throttle = float(self.config.get("ram_throttle_mb", 3500.0))
        release = float(self.config.get("ram_release_mb", 2800.0))
        if release >= throttle:
            self.log.warning(
                "ram_release_mb >= ram_throttle_mb; swapping to keep hysteresis sane",
                throttle=throttle,
                release=release,
            )
            throttle, release = release, throttle
        self.spawn(self._sample_loop(throttle, release), "sample")

    async def _sample_loop(self, throttle: float, release: float) -> None:
        interval = float(self.config.get("interval", 5.0))
        while True:
            await asyncio.sleep(interval)
            sample = self._sample()
            self.metrics.observe("gov.rss_mb", sample["rss_mb"])
            self.metrics.observe("gov.sample_latency_s", sample["latency_s"])
            level = self.level
            if level == "normal" and sample["rss_mb"] >= throttle:
                level = "throttled"
            elif level == "throttled" and sample["rss_mb"] <= release:
                level = "normal"
            if level != self.level:
                self.level = level
                self.log.info(
                    "Governor level changed", level=level, rss_mb=round(sample["rss_mb"], 1)
                )
                await self.bus.publish(
                    Event(
                        "GovernorThrottled",
                        {"level": level, "rss_mb": round(sample["rss_mb"], 1)},
                    )
                )

    def _sample(self) -> dict:
        start = time.perf_counter()
        rss_mb = psutil.Process().memory_info().rss / MB
        return {
            "rss_mb": rss_mb,
            "system_percent": psutil.virtual_memory().percent,
            "latency_s": time.perf_counter() - start,
        }
