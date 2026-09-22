"""Contract tests: TamperService (FUNC-05 Step 13 fix — camera tamper awareness).

The logic is a pure class (no camera needed) plus an end-to-end service test
driven through a patched shared FrameStore.
"""
import numpy as np
import pytest

from aria.core.config import Config
from aria.core.events import Event
from aria.perception.tamper import TamperLogic, TamperService
from helpers import Collector, attach, make_bus, run


def test_tamper_logic_uniform_frames_trip_after_consecutive():
    logic = TamperLogic(std_threshold=6.0, dark_threshold=18.0, consecutive=3)
    black = np.zeros((10, 10), dtype=np.uint8)
    assert logic.update(black) is None
    assert logic.update(black) is None
    assert logic.update(black) == "tampered"  # 3rd consecutive sample trips it


def test_tamper_logic_noise_frames_stay_normal():
    logic = TamperLogic(std_threshold=6.0, dark_threshold=18.0, consecutive=3)
    rng = np.random.default_rng(7)
    for _ in range(10):
        assert logic.update(rng.integers(0, 255, (40, 40), dtype=np.uint8)) is None
    assert logic.state == "normal"


def test_tamper_logic_recovery_publishes_cleared():
    logic = TamperLogic(consecutive=2)
    logic.update(np.zeros((10, 10), dtype=np.uint8))
    assert logic.update(np.zeros((10, 10), dtype=np.uint8)) == "tampered"
    rng = np.random.default_rng(3)
    assert logic.update(rng.integers(0, 255, (40, 40), dtype=np.uint8)) == "cleared"
    assert logic.state == "normal"


def test_tamper_service_end_to_end(monkeypatch):
    async def scenario():
        import aria.perception.framestore as fs_mod

        store = fs_mod.FrameStore(capacity=4)
        monkeypatch.setattr(fs_mod, "DEFAULT_STORE", store)

        bus = make_bus()
        await bus.start()
        detected, cleared = Collector(), Collector()
        bus.subscribe("TamperDetected", detected, policy="block", maxsize=8)
        bus.subscribe("TamperCleared", cleared, policy="block", maxsize=8)

        svc = attach(TamperService(), bus, {"every": 1, "consecutive": 2, "width_downscale": 64})
        await svc.init()
        await svc.start()

        # tampered: uniform black frames (2 consecutive samples trip it)
        for fid in range(1, 4):
            store.put(fid, np.zeros((40, 40, 3), dtype=np.uint8))
            await svc._on_frame(Event("Frame", {"frame_id": fid}))
        # recovery: textured frames
        rng = np.random.default_rng(11)
        for fid in range(4, 7):
            store.put(fid, rng.integers(0, 255, (40, 40, 3), dtype=np.uint8))
            await svc._on_frame(Event("Frame", {"frame_id": fid}))

        await detected.wait_for(1)
        await cleared.wait_for(1)
        await svc.stop()
        await bus.stop()
    run(scenario())
