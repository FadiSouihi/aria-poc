"""Synthesize speech fixtures — the audio twin of record_fixture.py.

    python tools/say.py --name hello_aria --text "Hello ARIA, where is the library?"
    python tools/say.py --name other_speaker --text "..." --voice en-US-GuyNeural
    python tools/say.py --name offline_probe --text "..." --provider sapi

Writes ``fixtures/audio/<name>.wav`` (16 kHz mono) so STT/VAD/turn tests and
benches replay real speech with no microphone involved.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aria.audio.store import SAMPLE_RATE, resample_linear, write_wav  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesize a speech fixture")
    ap.add_argument("--name", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--provider", default="auto", choices=["auto", "edge", "sapi", "fake"])
    ap.add_argument("--voice", default="en-US-AriaNeural")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--pause-after-s", type=float, default=0.6,
                    help="trailing silence (lets VAD close the turn)")
    ap.add_argument("--noise-only", action="store_true", help="write synthetic noise instead (negative fixture)")
    args = ap.parse_args()

    out_dir = ROOT / "fixtures" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.wav"

    if args.noise_only:
        rng = np.random.default_rng(11)
        audio = (rng.standard_normal(int(SAMPLE_RATE * 1.5)) * 0.02).astype(np.float32)
        write_wav(out, audio, SAMPLE_RATE)
        print(f"OK (noise fixture): {out} ({audio.size / SAMPLE_RATE:.2f}s)")
        return 0

    from aria.audio.tts import EdgeTtsProvider, FakeTtsProvider, SapiTtsProvider

    providers = []
    if args.provider in ("auto", "edge"):
        providers.append(EdgeTtsProvider(voice=args.voice, rate=args.rate))
    if args.provider in ("auto", "sapi"):
        providers.append(SapiTtsProvider(rate=args.rate))
    if args.provider == "fake":
        providers.append(FakeTtsProvider())

    last_error = None
    for provider in providers:
        try:
            audio, rate = provider.synth(args.text)
            audio = resample_linear(np.asarray(audio, dtype=np.float32), rate, SAMPLE_RATE)
            if args.pause_after_s > 0:
                audio = np.concatenate([audio, np.zeros(int(args.pause_after_s * SAMPLE_RATE), dtype=np.float32)])
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0:
                audio = (audio / peak * 0.8).astype(np.float32)
            write_wav(out, audio, SAMPLE_RATE)
            print(f"OK: {out} ({audio.size / SAMPLE_RATE:.2f}s via {provider.name})")
            return 0
        except Exception as exc:
            last_error = exc
            print(f"provider {provider.name} failed: {exc}", file=sys.stderr)
    print(f"ERROR: no TTS provider produced audio: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())