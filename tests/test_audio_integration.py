"""Integration tests: the real audio models on real speech fixtures.

These validate the empirical behaviour of each downloaded model (VAD
probabilities, Smart Turn discrimination, Whisper transcription, speaker
embeddings) instead of assuming the docs are right. Every test skips cleanly
when its model or fixture is missing, so the hermetic suite stays green.
"""
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "fixtures" / "audio"
VAD_ONNX = ROOT / "weights" / "silero_vad.onnx"
SMART_TURN_ONNX = ROOT / "weights" / "smart-turn-v3.2-cpu.onnx"
WHISPER_DIR = ROOT / "weights" / "whisper-turbo-ct2"


def _fixture(name: str):
    path = AUDIO_DIR / f"{name}.wav"
    if not path.exists():
        pytest.skip(f"audio fixture missing: {path}")
    from aria.audio.store import read_wav

    audio, rate = read_wav(path)
    return audio.astype(np.float32), rate


@pytest.mark.parametrize("name", ["greeting", "question"])
def test_vad_probabilities_separate_speech_from_silence(name):
    if not VAD_ONNX.exists():
        pytest.skip("silero_vad.onnx not present")
    from aria.audio.vad import FRAME, SileroVadModel

    audio, rate = _fixture(name)
    model = SileroVadModel(str(VAD_ONNX), rate=rate, frame=FRAME)
    probs = []
    for i in range(0, audio.size - FRAME, FRAME):
        probs.append(model.prob(audio[i:i + FRAME]))
    probs = np.asarray(probs)
    assert probs.size > 10
    speech = probs.max()
    trailing = probs[-3:].mean()          # fixture ends with silence
    assert speech > 0.5, f"VAD never fired on real speech (max={speech:.3f})"
    assert trailing < 0.5, f"VAD stayed hot in trailing silence (mean={trailing:.3f})"


def test_vad_stays_quiet_on_noise_fixture():
    if not VAD_ONNX.exists():
        pytest.skip("silero_vad.onnx not present")
    from aria.audio.vad import FRAME, SileroVadModel

    audio, rate = _fixture("noise_only")
    model = SileroVadModel(str(VAD_ONNX), rate=rate, frame=FRAME)
    probs = [model.prob(audio[i:i + FRAME]) for i in range(0, audio.size - FRAME, FRAME)]
    assert float(np.mean(probs)) < 0.5, "stationary noise must not read as speech"


def test_smart_turn_prefers_complete_over_clipped_speech():
    """End-of-turn model sanity: a finished sentence must score higher than a
    mid-sentence cut of the same audio."""
    if not SMART_TURN_ONNX.exists():
        pytest.skip("smart-turn model not present")
    from aria.audio.turn import SmartTurnClassifier

    audio, rate = _fixture("greeting")
    classifier = SmartTurnClassifier(str(SMART_TURN_ONNX), rate=rate)
    complete = classifier.prob(audio)
    clipped = classifier.prob(audio[: int(audio.size * 0.55)])   # cut mid-sentence
    assert 0.0 <= complete <= 1.0 and 0.0 <= clipped <= 1.0
    assert complete > clipped, f"complete={complete:.3f} should exceed clipped={clipped:.3f}"


def test_whisper_transcribes_the_fixture():
    if not WHISPER_DIR.exists():
        pytest.skip("whisper model not downloaded")
    from aria.audio.stt import WhisperTranscriber

    audio, _rate = _fixture("greeting")
    transcriber = WhisperTranscriber(str(WHISPER_DIR), device="cpu", compute_type="int8", language="auto")
    text, elapsed, language, probability = transcriber.transcribe(audio)
    assert len(text) > 5, f"expected a transcript, got {text!r}"
    assert "library" in text.lower(), f"expected 'library' in transcript, got {text!r}"
    assert language == "en", f"expected detected language 'en', got {language!r}"
    assert probability is None or probability > 0.5
    # Correctness is the contract here; throughput is *measured* (not gated) by
    # tools/bench_audio.py. This ceiling only catches a pathological stall. Note
    # the real number: large-v3-turbo on CPU int8 runs ~9x realtime (RTF ~9), so a
    # 4.7 s clip takes ~45 s — which is why the GPU path exists and why a smaller
    # model is the right choice for the CPU fallback.
    assert elapsed < 120.0, f"transcription took {elapsed:.1f}s — suspected stall"


@pytest.mark.parametrize("name,expected_language,script", [
    ("fr_library", "fr", "latin"),
    ("ar_greeting", "ar", "arabic"),
])
def test_non_english_speech_is_transcribed_not_translated(name, expected_language, script):
    """Regression for "it translates what I say": with language=auto and
    task=transcribe, French/Arabic must come back in their own language and
    script — never as English text."""
    if not WHISPER_DIR.exists():
        pytest.skip("whisper model not downloaded")
    from aria.audio.stt import WhisperTranscriber

    audio, _rate = _fixture(name)
    transcriber = WhisperTranscriber(str(WHISPER_DIR), device="cpu", compute_type="int8", language="auto")
    text, _elapsed, language, _probability = transcriber.transcribe(audio)
    assert language == expected_language, f"detected {language!r}, expected {expected_language!r}"
    if script == "arabic":
        assert any("\u0600" <= ch <= "\u06ff" for ch in text), \
            f"expected Arabic script, got {text!r} (translated?)"
    else:
        assert any(ch.isalpha() and ord(ch) < 0x250 for ch in text), \
            f"expected Latin script, got {text!r}"
        assert "the library" not in text.lower(), f"looks translated: {text!r}"


def test_edge_provider_speaks_in_the_voice_it_is_given():
    """Regression: the service reported fr-FR-DeniseNeural while edge-tts kept
    using its constructor voice, so ARIA always sounded English. Two different
    voices for the same text must produce different audio."""
    try:
        import edge_tts  # noqa: F401
    except Exception as exc:
        pytest.skip(f"edge-tts unavailable: {exc}")
    from aria.audio.tts import EdgeTtsProvider

    provider = EdgeTtsProvider(voice="en-US-AriaNeural")
    text = "Bonjour, comment allez-vous ?"
    try:
        english, rate = provider.synth(text, "en", "en-US-AriaNeural")
        french, _ = provider.synth(text, "fr", "fr-FR-DeniseNeural")
    except Exception as exc:                 # offline
        pytest.skip(f"edge-tts request failed: {exc}")
    assert english.size > 0 and french.size > 0
    assert abs(english.size - french.size) > 0.05 * english.size, \
        "same audio length for two voices — the voice argument was ignored"


def test_voiceprint_embedder_separates_speakers_and_matches_itself():
    from aria.audio.voiceprint import build_embedder

    try:
        embedder, _threshold = build_embedder("auto")
    except Exception as exc:      # provider unavailable on this machine
        pytest.skip(f"voiceprint provider unavailable: {exc}")

    same_a, _ = _fixture("greeting")
    same_b, _ = _fixture("greeting2") if (AUDIO_DIR / "greeting2.wav").exists() else (None, None)
    other, _ = _fixture("question")

    emb_a = embedder.embed(same_a)
    emb_other = embedder.embed(other)
    assert emb_a.shape == emb_other.shape and emb_a.size > 50
    norm = lambda v: v / max(float(np.linalg.norm(v)), 1e-9)
    cross = float(np.dot(norm(emb_a), norm(emb_other)))
    self_sim = float(np.dot(norm(emb_a), norm(emb_a)))
    assert abs(self_sim - 1.0) < 1e-3
    if same_b is not None:
        emb_b = embedder.embed(same_b)
        same = float(np.dot(norm(emb_a), norm(emb_b)))
        assert same > cross, f"same speaker {same:.3f} should beat different speaker {cross:.3f}"