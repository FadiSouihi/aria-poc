"""Enroll a speaker voice-print.

    python tools/enroll_voice.py --name fedi --seconds 8          # live mic
    python tools/enroll_voice.py --name visitor --wav fixtures/audio/other.wav

Records (or reads) speech, embeds it with the ECAPA model and writes
``data/voices/<name>.npz`` for the voice-print gate.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aria.audio.store import SAMPLE_RATE, dbfs, read_wav, resample_linear, write_wav  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Enroll a speaker voice-print")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--wav", default="", help="use an existing WAV instead of the mic")
    ap.add_argument("--model", default="weights/wavlm-base-plus-sv.onnx")
    ap.add_argument("--provider", default="auto", choices=["auto", "wavlm", "resemblyzer", "speechbrain"])
    ap.add_argument("--output", default="embeddings", choices=["embeddings", "logits"])
    ap.add_argument("--keep-wav", action="store_true", help="also save the recorded audio fixture")
    args = ap.parse_args()

    if args.wav:
        audio, rate = read_wav(args.wav)
        audio = resample_linear(audio, rate, SAMPLE_RATE)
    else:
        import sounddevice as sd

        print(f"Recording {args.seconds}s — please speak naturally…")
        audio = np.asarray(
            sd.rec(int(args.seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", device=None if args.device < 0 else args.device),
            dtype=np.float32,
        ).reshape(-1)
        sd.wait()

    if audio.size < SAMPLE_RATE:
        print("ERROR: need at least 1 s of speech", file=sys.stderr)
        return 1
    print(f"Audio: {audio.size / SAMPLE_RATE:.2f}s, mean {dbfs(audio):.1f} dBFS")

    if args.keep_wav:
        out_dir = ROOT / "fixtures" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_wav(out_dir / f"enroll_{args.name}.wav", audio, SAMPLE_RATE)

    from aria.audio.voiceprint import build_embedder

    embedder, threshold = build_embedder(args.provider, args.model, args.output)
    emb = np.asarray(embedder.embed(audio), dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    emb = emb / norm if norm > 0 else emb

    out = ROOT / "data" / "voices" / f"{args.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, emb=emb, seconds=audio.size / SAMPLE_RATE, provider=embedder.name)
    print(f"Enrolled voice '{args.name}' → {out} ({emb.size}-d via {embedder.name}, "
          f"match threshold {threshold})")
    return 0


if __name__ == "__main__":
    sys.exit(main())