"""Contract tests: three-tier end-of-turn (FUNC-14 hard-pause / noise fixes).

Pure decision logic + the numpy log-mel used by the Smart Turn model.
"""
import numpy as np
import pytest

from aria.audio.turn import (EouTracker, HeuristicEouClassifier, _log_mel_whisper,
                             mel_filterbank_slaney)


# -- decision logic ------------------------------------------------------------
def test_semantic_complete_when_model_says_done():
    tracker = EouTracker(threshold=0.5)
    assert tracker.on_speech_end(now=10.0, segment_started=8.0, p_turn=0.9,
                                 duration_s=2.0) == "complete"


def test_semantic_incomplete_waits_for_more_speech():
    """p_turn low + long enough utterance → keep listening (the 'only pausing' case)."""
    tracker = EouTracker(threshold=0.5, pending_timeout_s=1.5)
    assert tracker.on_speech_end(now=10.0, segment_started=8.0, p_turn=0.2,
                                 duration_s=2.0) == "wait"


def test_short_low_confidence_is_discarded_as_noise():
    """A cough/click must not open a turn (FUNC-14 false responses)."""
    tracker = EouTracker(threshold=0.5, min_utterance_s=0.6)
    assert tracker.on_speech_end(now=10.0, segment_started=9.8, p_turn=0.1,
                                 duration_s=0.2) == "discard"


def test_hard_timeout_always_completes():
    tracker = EouTracker(threshold=0.99)
    assert tracker.on_speech_end(now=10.0, segment_started=0.0, p_turn=0.0,
                                 duration_s=10.0, reason="hard_timeout") == "complete"


def test_max_utterance_cap_completes():
    tracker = EouTracker(threshold=0.99, max_utterance_s=20.0)
    assert tracker.on_speech_end(now=30.0, segment_started=0.0, p_turn=0.1,
                                 duration_s=21.0) == "complete"


def test_pending_expiry_finishes_long_speech_and_drops_noise():
    tracker = EouTracker(threshold=0.5, min_utterance_s=0.6)
    assert tracker.on_pending_expiry(now=12.0, segment_started=9.0, duration_s=3.0) == "complete"
    assert tracker.on_pending_expiry(now=12.0, segment_started=11.7, duration_s=0.3) == "discard"


# -- heuristic fallback classifier ---------------------------------------------
def test_heuristic_classifier_rates_finished_speech_higher():
    classifier = HeuristicEouClassifier(rate=1000, long_utterance_s=1.0, trailing_silence_s=0.2)
    speech = np.ones(2000, dtype=np.float32) * 0.3
    speech[-200:] = 0.0                                  # quiet tail
    complete = classifier.prob(speech)
    clipped = classifier.prob(np.ones(400, dtype=np.float32) * 0.3)   # short blip
    assert complete > clipped
    assert complete >= 0.5


# -- log-mel for Smart Turn ------------------------------------------------------
def test_mel_filterbank_shape_and_normalisation():
    bank = mel_filterbank_slaney()
    assert bank.shape == (80, 201)                       # 80 mels × (n_fft/2 + 1)
    assert np.all(bank >= 0.0)
    assert np.all(bank.sum(axis=1) > 0.0)                # every filter has weight


def test_log_mel_shape_and_range():
    audio = np.sin(np.linspace(0, 100 * np.pi, 16000)).astype(np.float32) * 0.4
    features = _log_mel_whisper(audio)
    assert features.shape == (1, 80, 800)
    assert np.isfinite(features).all()
    assert features.min() >= -1.5 and features.max() <= 1.5   # whisper normalisation range


def test_log_mel_pads_short_audio():
    features = _log_mel_whisper(np.zeros(1600, dtype=np.float32))
    assert features.shape == (1, 80, 800)