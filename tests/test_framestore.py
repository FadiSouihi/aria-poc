"""Contract tests: FrameStore (bounded in-process pixel exchange)."""
import numpy as np

from aria.perception.framestore import FrameStore


def test_put_and_latest():
    store = FrameStore(capacity=4)
    store.put(1, np.zeros((2, 2), dtype=np.uint8))
    store.put(2, np.ones((2, 2), dtype=np.uint8))
    fid, frame = store.latest()
    assert fid == 2
    assert frame[0, 0] == 1
    assert len(store) == 2


def test_capacity_evicts_oldest():
    store = FrameStore(capacity=2)
    for i in range(5):
        store.put(i, np.full((1, 1), i, dtype=np.uint8))
    fid, frame = store.latest()
    assert fid == 4 and frame[0, 0] == 4
    assert len(store) == 2
    assert store.get(0) is None
    assert store.get(3) is not None


def test_empty_latest_is_none():
    store = FrameStore()
    assert store.latest() == (None, None)
    assert store.get(42) is None


def test_repeated_put_same_id_keeps_latest_value():
    store = FrameStore()
    store.put(1, np.zeros((1, 1), dtype=np.uint8))
    store.put(1, np.full((1, 1), 9, dtype=np.uint8))
    assert len(store) == 1
    fid, frame = store.latest()
    assert fid == 1 and frame[0, 0] == 9
