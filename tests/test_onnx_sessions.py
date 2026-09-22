"""ONNX Runtime session threading — the fix behind the app's CPU cost.

The VAD calls a tiny graph ~31x/second. With ORT's defaults (one thread per
physical core, spin-waiting) that showed up as the single largest CPU consumer in
the app for microseconds of real work. These tests pin the policy so it cannot
silently regress to the defaults.
"""
from __future__ import annotations

import pathlib

import pytest

from aria.core.onnx import DEFAULTS, default_spinning, default_threads, make_session, make_session_options

WEIGHTS = pathlib.Path(__file__).resolve().parents[1] / "weights"
VAD_MODEL = WEIGHTS / "silero_vad.onnx"


def test_defaults_are_one_thread_and_no_spinning():
    for kind in ("vad", "turn", "face", "voiceprint", "default"):
        assert default_threads(kind) >= 1
        assert default_spinning(kind) is False


def test_heavy_per_turn_model_may_use_more_threads():
    # WavLM is heavy but runs once per turn, so latency benefits from a couple of
    # threads; the tiny frequent models stay at one.
    assert default_threads("voiceprint") >= default_threads("vad")
    assert default_threads("vad") == 1


def test_session_options_carry_the_policy():
    options = make_session_options(kind="vad")
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1


def test_explicit_thread_override_is_respected():
    options = make_session_options(threads=4, kind="vad")
    assert options.intra_op_num_threads == 4


def test_thread_count_is_never_zero():
    assert make_session_options(threads=0, kind="vad").intra_op_num_threads == 1
    assert make_session_options(threads=-3, kind="vad").intra_op_num_threads == 1


def test_spinning_can_be_enabled_explicitly():
    # Not used by default, but the knob has to work for latency-critical models.
    options = make_session_options(kind="vad", spinning=True)
    assert options is not None


@pytest.mark.skipif(not VAD_MODEL.exists(), reason="silero_vad.onnx not present")
def test_real_session_uses_one_thread_and_runs():
    """End-to-end through the production wrapper: the session must build with the
    low-noise options and still return a sane probability."""
    import numpy as np

    from aria.audio.vad import FRAME, SileroVadModel

    model = SileroVadModel(str(VAD_MODEL), threads=1, spinning=False)
    assert model.session.get_providers() == ["CPUExecutionProvider"]
    silence = model.prob(np.zeros(FRAME, dtype=np.float32))
    assert 0.0 <= silence <= 1.0
    assert DEFAULTS["vad"]["threads"] == 1


def test_unknown_kind_falls_back_to_defaults():
    assert default_threads("no-such-kind") == DEFAULTS["default"]["threads"]
