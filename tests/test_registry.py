"""Contract tests: config-driven Registry (build, disable, restart, errors)."""
from aria.core.config import Config
from aria.core.service import RUNNING, STOPPED
from aria.core.registry import Registry
from helpers import make_bus, run


def _profile(disabled_b=False, bad_class=False):
    services = {
        "a": {
            "class": "tests.dummies:SimpleService",
            "enabled": True,
            "config": {"heartbeat_interval": 0.05},
        }
    }
    if not bad_class:
        services["b"] = {
            "class": "aria.perception.scene:SceneManagerService",
            "enabled": not disabled_b,
            "config": {},
        }
    else:
        services["broken"] = {"class": "tests.dummies:DoesNotExist"}
    return Config({"services": services}, "<profile>")


def test_build_skips_disabled_and_instantiates_enabled():
    async def scenario():
        bus = make_bus()
        await bus.start()
        registry = Registry(bus, _profile(disabled_b=True))
        registry.build()
        assert registry.names() == ["a"]
        assert "b" in registry.disabled
        await registry.start_all()
        assert registry.get("a").state == RUNNING
        await registry.stop_all()
        assert registry.get("a").state == STOPPED
        await bus.stop()
    run(scenario())


def test_restart_in_place_counts_and_recovers():
    async def scenario():
        bus = make_bus()
        await bus.start()
        registry = Registry(bus, _profile())
        registry.build()
        await registry.start_all()
        await registry.restart("a", reason="test")
        assert registry.restart_counts["a"] == 1
        assert registry.get("a").state == RUNNING
        await registry.stop_all()
        await bus.stop()
    run(scenario())


def test_unbuildable_service_is_skipped_not_fatal():
    async def scenario():
        bus = make_bus()
        registry = Registry(bus, _profile(bad_class=True))
        registry.build()
        assert registry.names() == ["a"]  # broken entry skipped, good one built
    run(scenario())
