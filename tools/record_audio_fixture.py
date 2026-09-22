"""Record a microphone fixture (WAV + meta) for replay tests and benches.

    python tools/record_audio_fixture.py --name my_voice --seconds 6
    python tools/record_audio_fixture.py --name room_noise --seconds 10 --device 1

Writes ``fixtures/audio/<name>.wav`` at 16 kHz mono plus a ``meta.yaml``.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aria.audio.store import SAMPLE_RATE, dbfs, write_wav  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Record a mic fixture")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--device", type=int, default=-1, help="-1 = system default")
    ap.add_argument("--rate", type=int, default=SAMPLE_RATE)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    import sounddevice as sd

    frames = int(args.seconds * args.rate)
    print(f"Recording {args.seconds}s from device {args.device}… speak now")
    audio = sd.rec(frames, samplerate=args.rate, channels=1, dtype="float32",
                   device=None if args.device < 0 else args.device)
    sd.wait()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    out_dir = ROOT / "fixtures" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{args.name}.wav"
    write_wav(wav_path, audio, args.rate)
    meta = {
        "name": args.name,
        "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "microphone",
        "device": args.device,
        "rate": args.rate,
        "seconds": round(audio.size / args.rate, 2),
        "dbfs_mean": round(dbfs(audio), 1),
        "notes": args.notes,
    }
    (out_dir / f"{args.name}.meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"OK: {wav_path} ({meta['seconds']}s, mean {meta['dbfs_mean']} dBFS)")
    return 0


if __name__ == "__main__":
    sys.exit(main())