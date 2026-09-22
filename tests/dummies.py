"""Importable dummy services for registry/contract tests."""
import asyncio

from aria.core.service import Service


class SimpleService(Service):
    name = "test.simple"
    produces = ("SceneTick",)
    config_schema = {"speed": (1, (int,)), "heartbeat_interval": (5.0, (int, float))}

    async def on_start(self) -> None:
        self.spawn(self._loop(), "tick")

    async def _loop(self) -> None:
        interval = float(self.config.get("heartbeat_interval", 0.05))
        while True:
            self.metrics.inc("dummies.ticks")
            await asyncio.sleep(interval)
