"""Contract tests: Service lifecycle, schema validation, crash containment."""
import asyncio

import pytest

from aria.core.config import Config
from aria.core.service import CRASHED, RUNNING, STOPPED, Service
from helpers import Collector, CrashOnceService, HeartbeatStopper, attach, make_bus, run


class SimpleService(Service):
    name = "test.simple"
    config_schema = {"speed": (1, (int,)), "heartbeat_interval": (5.0, (int, float))}


def test_lifecycle_transitions_and_health():
    async def scenario():
        bus = make_bus()
        await bus.start()
        svc = attach(SimpleService(), bus, {"speed": 3})
        assert svc.config.get("speed") == 3
        assert svc.state == "new"
        await svc.start()
        assert svc.state == RUNNING
        assert svc.health().status == "ok"
        await svc.stop()
        assert svc.state == STOPPED
        await svc.start()  # restart works
        assert svc.state == RUNNING
        await bus.stop()
    run(scenario())


def test_unknown_config_keys_dropped_and_defaults_filled():
    async def scenario():
        bus = make_bus()
        svc = attach(SimpleService(), bus, {"junk": True})
        assert "junk" not in svc.config.raw
        assert svc.config.get("speed") == 1  # schema default
    run(scenario())


def test_wrong_config_type_rejected_early():
    with pytest.raises(TypeError):
        SimpleService().attach(None, Config({"speed": "fast"}, "<bad>"))


def test_crash_containment_publishes_and_app_stays_alive():
    async def scenario():
        bus = make_bus(maxsize=16)
        await bus.start()
        crashes = Collector()
        bus.subscribe("ServiceCrashed", crashes)

        crasher = attach(CrashOnceService(), bus, {"heartbeat_interval": 0.05})
        survivor = attach(SimpleService(), bus, {"heartbeat_interval": 0.05})
        await crasher.start()
        await survivor.start()

        await crashes.wait_for(1)
        assert crasher.state == CRASHED
        assert survivor.state == RUNNING  # the app keeps running
        assert crashes.events[0].payload["service"] == "test.crash_once"

        await crasher.start()  # recovery path: next start is steady
        assert crasher.state == RUNNING
        await bus.stop()
    run(scenario())


def test_heartbeat_updates_while_running():
    async def scenario():
        bus = make_bus()
        await bus.start()
        svc = attach(HeartbeatStopper(), bus, {"heartbeat_interval": 0.02})
        before = svc.last_heartbeat
        await svc.start()
        await asyncio.sleep(0.08)
        assert svc.last_heartbeat > before
        await bus.stop()
    run(scenario())
