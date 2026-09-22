"""Contract tests: Governor hysteresis decisions (sampling is injected)."""
import pytest

from aria.core.service import Service
from aria.core.config import Config
from helpers import Collector, attach, make_bus, run


class BurstyService(Service):
    """Minimal consumer to prove the app survives a crash it observes."""

    name = "test.bursty"
    config_schema = {"heartbeat_interval": (5.0, (int, float))}


def test_governor_transitions_with_hysteresis():
    from aria.core.governor import GovernorService

    async def scenario():
        bus = make_bus(maxsize=16)
        await bus.start()
        gov = attach(
            GovernorService(),
            bus,
            {"interval": 0.01, "ram_throttle_mb": 50, "ram_release_mb": 30, "heartbeat_interval": 0.05},
        )
        rss = [200.0, 200.0, 200.0, 20.0, 20.0, 20.0]  # start high, then low
        gov._sample = lambda: {
            "rss_mb": rss.pop(0) if rss else 20.0,
            "system_percent": 10.0,
            "latency_s": 0.0001,
        }
        levels = Collector()
        bus.subscribe("GovernorThrottled", levels)

        await gov.start()
        await levels.wait_for(2)
        assert levels.events[0].payload["level"] == "throttled"
        assert levels.events[1].payload["level"] == "normal"
        assert gov.level == "normal"
        await gov.stop()
        await bus.stop()
    run(scenario())


def test_governor_swaps_bad_thresholds():
    from aria.core.governor import GovernorService

    async def scenario():
        bus = make_bus()
        gov = attach(
            GovernorService(),
            bus,
            {"ram_throttle_mb": 30, "ram_release_mb": 50, "heartbeat_interval": 0.05},
        )
        await gov.start()  # must not crash on misconfigured thresholds
        await gov.stop()
    run(scenario())
