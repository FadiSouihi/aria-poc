"""Contract tests: Watchdog restarts crashed and hung services in place."""
import asyncio

from aria.core.config import Config
from aria.core.registry import Registry
from helpers import Collector, CrashOnceService, HeartbeatStopper, attach, make_bus, run

WD_CFG = {
    "interval": 0.02,
    "stale_after": 0.1,
    "max_restarts": 3,
    "backoff": 0.0,
    "heartbeat_interval": 0.05,
}


def _build(bus, cfg, with_crasher=True):
    registry = Registry(bus, Config({"services": {}}, "<test>"))
    dog = attach(WatchdogForTest(), bus, dict(WD_CFG))
    dog.set_registry(registry)
    registry.services["watchdog"] = dog
    if with_crasher:
        crasher = attach(CrashOnceService(), bus, {"heartbeat_interval": 0.05})
        registry.services["crasher"] = crasher
    else:
        crasher = None
    stopper = attach(HeartbeatStopper(), bus, {"heartbeat_interval": 0.02})
    registry.services["stopper"] = stopper
    return registry, crasher, stopper


from aria.core.watchdog import WatchdogService as WatchdogForTest  # noqa: E402


def test_watchdog_restarts_crashed_service_once():
    async def scenario():
        bus = make_bus(maxsize=32)
        await bus.start()
        registry, crasher, stopper = _build(bus, WD_CFG)
        crashes, restarts = Collector(), Collector()
        bus.subscribe("ServiceCrashed", crashes)
        bus.subscribe("ServiceRestarted", restarts)

        await registry.start_all()
        await crashes.wait_for(1)           # crash event from the failing task
        await restarts.wait_for(1, timeout=3)  # watchdog restarts it
        assert crasher.state == "running"
        assert registry.restart_counts["crasher"] == 1

        await asyncio.sleep(0.3)            # steady: no further restarts
        assert registry.restart_counts["crasher"] == 1
        await registry.stop_all()
        await bus.stop()
    run(scenario())


def test_watchdog_respects_each_service_heartbeat_period():
    """Regression: a 5 s heartbeat against a 5 s threshold meant zero margin, so
    heavy GPU work elsewhere (Whisper) caused spurious restarts. A service may
    only be called stale well past its own heartbeat period."""
    import time

    from dummies import SimpleService

    async def scenario():
        bus = make_bus(maxsize=32)
        await bus.start()
        registry = Registry(bus, Config({"services": {}}, "<test>"))
        dog = attach(WatchdogForTest(), bus, dict(WD_CFG))        # stale_after 0.1
        dog.set_registry(registry)
        registry.services["watchdog"] = dog
        slow = attach(SimpleService(), bus, {"heartbeat_interval": 5.0})
        registry.services["slow"] = slow
        await registry.start_all()

        slow.last_heartbeat = time.monotonic() - 6.0              # 6 s delay < 3x5 s
        await dog._check_all()
        assert registry.restart_counts.get("slow", 0) == 0

        slow.last_heartbeat = time.monotonic() - 20.0             # beyond 3x5 s → stale
        await dog._check_all()
        assert registry.restart_counts.get("slow", 0) == 1

        await registry.stop_all()
        await bus.stop()
    run(scenario())


def test_watchdog_catches_hung_heartbeat_and_recovers():
    async def scenario():
        bus = make_bus(maxsize=32)
        await bus.start()
        registry, crasher, stopper = _build(bus, WD_CFG, with_crasher=False)
        restarts = Collector()
        bus.subscribe("ServiceRestarted", restarts)

        await registry.start_all()
        await asyncio.sleep(0.05)  # healthy phase
        assert stopper.state == "running"

        stopper.beat = False                 # simulate a hang
        await restarts.wait_for(1, timeout=3)
        stopper.beat = True                  # recover for real
        await asyncio.sleep(0.3)
        assert 1 <= registry.restart_counts["stopper"] <= 2
        assert stopper.state == "running"
        assert stopper.health().status == "ok"
        await registry.stop_all()
        await bus.stop()
    run(scenario())
