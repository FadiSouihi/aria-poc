"""Contract tests: SceneManager per-track FSM (the unified 0/1/N logic).

Drives ``_on_scenetick`` directly with synthetic track events — no ML needed.
Timers are shrunk via config so the test runs in milliseconds.
"""
import asyncio

from aria.core.config import Config
from aria.core.events import Event
from aria.perception.scene import SceneManagerService
from helpers import Collector, attach, make_bus, run

CFG = {
    "near_area_frac": 0.06,
    "central_frac": 0.60,
    "near_dwell_s": 0.1,
    "engage_dwell_s": 0.1,
    "departed_after_s": 0.3,
    "heartbeat_interval": 5.0,
}
NEAR_BBOX = [220, 180, 420, 300]   # ~8% of a 480x640 frame, dead center
FAR_BBOX = [10, 10, 60, 50]        # tiny corner box


def _tick(track_id: int, bbox) -> Event:
    return Event("SceneTick", {"tracks": [{"id": track_id, "bbox": bbox, "conf": 0.9}],
                               "frame_id": 1, "latency_ms": 1.0, "detect_hz": 15.0})


def _states(collector):
    return {t["id"]: t["state"] for e in collector.events for t in e.payload["tracks"]}


def test_presence_then_near_then_engaged():
    async def scenario():
        bus = make_bus()
        await bus.start()
        out = Collector()
        bus.subscribe("TrackStates", out, policy="block", maxsize=32)
        svc = attach(SceneManagerService(), bus, CFG)
        await svc.init()
        await svc.start()

        await svc._on_scenetick(_tick(1, NEAR_BBOX))
        await out.wait_for(1)
        assert _states(out)[1] == "PRESENCE"
        await asyncio.sleep(0.15)
        await svc._on_scenetick(_tick(1, NEAR_BBOX))
        await out.wait_for(2)
        assert _states(out)[1] == "NEAR"
        await asyncio.sleep(0.15)
        await svc._on_scenetick(_tick(1, NEAR_BBOX))
        await out.wait_for(3)
        assert _states(out)[1] == "ENGAGED"
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_far_track_never_engages():
    async def scenario():
        bus = make_bus()
        await bus.start()
        out = Collector()
        bus.subscribe("TrackStates", out, policy="block", maxsize=32)
        svc = attach(SceneManagerService(), bus, CFG)
        await svc.init()
        await svc.start()
        for _ in range(4):
            await svc._on_scenetick(_tick(7, FAR_BBOX))
            await asyncio.sleep(0.05)
        assert _states(out)[7] == "PRESENCE"
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_departure_cleanup_after_absence():
    async def scenario():
        bus = make_bus()
        await bus.start()
        out = Collector()
        bus.subscribe("TrackStates", out, policy="block", maxsize=32)
        svc = attach(SceneManagerService(), bus, CFG)
        await svc.init()
        await svc.start()
        await svc._on_scenetick(_tick(3, NEAR_BBOX))
        await asyncio.sleep(0.4)  # > departed_after_s with no more ticks
        await svc._on_scenetick(_tick(9, NEAR_BBOX))  # next tick triggers cleanup
        assert 3 not in svc.tracks
        assert 9 in svc.tracks
        await svc.stop()
        await bus.stop()
    run(scenario())


def test_face_match_updates_identity():
    async def scenario():
        bus = make_bus()
        await bus.start()
        out = Collector()
        bus.subscribe("TrackStates", out, policy="block", maxsize=32)
        svc = attach(SceneManagerService(), bus, CFG)
        await svc.init()
        await svc.start()
        await svc._on_scenetick(_tick(5, NEAR_BBOX))
        await svc._on_face_matched(Event("FaceMatched", {"track_id": 5, "identity": "fedi",
                                                         "similarity": 0.72, "votes": 3, "reason": "vote"}))
        assert svc.tracks[5]["identity"] == "fedi"
        await svc.stop()
        await bus.stop()
    run(scenario())
