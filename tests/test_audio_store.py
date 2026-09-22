"""Contract tests: AudioStore + WAV helpers (Phase 2 audio twin of FrameStore)."""
import numpy as np

from aria.audio.store import (AudioStore, dbfs, read_wav, resample_linear, rms, write_wav)


def test_append_and_read_window():
    store = AudioStore(sample_rate=1000, retained_s=2.0)
    store.append(np.ones(500, dtype=np.float32))
    store.append(np.full(500, 2.0, dtype=np.float32))
    assert store.total_samples() == 1000
    window = store.read_window(1.0)
    assert window.size == 1000          # 1 s at 1 kHz
    assert window[-1] == 2.0


def test_read_window_shorter_than_requested_at_start():
    store = AudioStore(sample_rate=1000, retained_s=2.0)
    store.append(np.ones(200, dtype=np.float32))
    assert store.read_window(1.0).size == 200


def test_retention_evicts_oldest():
    store = AudioStore(sample_rate=100, retained_s=1.0)   # keeps 100 samples
    store.append(np.zeros(80, dtype=np.float32))
    store.append(np.ones(80, dtype=np.float32))
    assert store.total_samples() == 160
    stats = store.stats()
    assert stats["dropped_samples"] > 0
    window = store.read_window(10.0)
    assert window.size <= 100


def test_read_since_returns_tail():
    store = AudioStore(sample_rate=1000, retained_s=5.0)
    store.append(np.arange(0, 100, dtype=np.float32))
    store.append(np.arange(100, 200, dtype=np.float32))
    tail = store.read_since(150)
    assert tail.size == 50
    assert tail[0] == 150.0


def test_read_since_before_retained_returns_available():
    store = AudioStore(sample_rate=100, retained_s=1.0)
    store.append(np.ones(300, dtype=np.float32))
    tail = store.read_since(0)
    assert 0 < tail.size <= 100


def test_rms_and_dbfs():
    silence = np.zeros(100, dtype=np.float32)
    assert rms(silence) == 0.0
    assert dbfs(silence) == -120.0
    loud = np.full(100, 0.5, dtype=np.float32)
    assert abs(rms(loud) - 0.5) < 1e-6
    assert -6.5 < dbfs(loud) < -5.5      # ~ -6 dBFS


def test_wav_roundtrip(tmp_path):
    rate = 16000
    original = (np.sin(np.linspace(0, 40 * np.pi, rate)) * 0.5).astype(np.float32)
    path = tmp_path / "tone.wav"
    write_wav(path, original, rate)
    loaded, loaded_rate = read_wav(path)
    assert loaded_rate == rate
    assert loaded.size == original.size
    assert np.max(np.abs(loaded - original)) < 1e-3


def test_resample_linear_changes_rate():
    source = np.sin(np.linspace(0, 20 * np.pi, 8000)).astype(np.float32)
    out = resample_linear(source, 8000, 16000)
    assert abs(out.size - 16000) <= 1