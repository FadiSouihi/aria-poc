"""Fetch the model weights this project needs (they are not stored in git).

    python tools/fetch_models.py               # everything missing
    python tools/fetch_models.py --list        # what, where from, licence
    python tools/fetch_models.py --only vad whisper
    python tools/fetch_models.py --verify      # check existing files, download nothing
    python tools/fetch_models.py --force       # re-download even if present

Every entry pins a SHA-256, so a fresh machine gets the *same* bytes the
benchmarks in ROADMAP.md were measured with. A mismatch fails loudly rather than
silently changing behaviour (that is how a different WavLM export could shift the
voice-print threshold).

Total download is ~1.7 GB, dominated by Whisper `large-v3-turbo` (1.5 GB). On a
metered connection use `--only vad turn face vision` first to get the perception
stack running (about 55 MB) and add `whisper` later.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
UA = {"User-Agent": "aria-poc-fetch-models/1.0 (+https://github.com/FadiSouihi)"}


@dataclass
class Model:
    key: str
    dest: str                      # path relative to the repo root
    what: str                      # what it is used for
    license: str
    size: int                      # expected bytes (0 = unknown)
    sha256: Optional[str] = None   # None = warn on mismatch instead of failing
    url: Optional[str] = None      # direct download
    hf_repo: Optional[str] = None  # huggingface file download
    hf_file: Optional[str] = None
    hf_snapshot: List[str] = field(default_factory=list)  # multiple files into dest/
    ultralytics: Optional[str] = None                     # downloaded by ultralytics

    @property
    def path(self) -> pathlib.Path:
        return ROOT / self.dest


MODELS: List[Model] = [
    Model(
        key="vision",
        dest="weights/yolo26n.pt",
        what="YOLO26n person detector + ByteTrack (Ultralytics)",
        license="AGPL-3.0 (Ultralytics)",
        size=5_544_453,
        sha256="9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
        ultralytics="yolo26n.pt",
    ),
    Model(
        key="vad",
        dest="weights/silero_vad.onnx",
        what="Silero VAD v5 speech/silence detection (run on ONNX, not torchaudio)",
        license="MIT",
        size=2_327_524,
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        url="https://raw.githubusercontent.com/snakers4/silero-vad/master/"
            "src/silero_vad/data/silero_vad.onnx",
    ),
    Model(
        key="turn",
        dest="weights/smart-turn-v3.2-cpu.onnx",
        what="Smart Turn v3.2 semantic end-of-turn classifier (EOU tier 2)",
        license="BSD-2-Clause",
        size=8_679_182,
        sha256="2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
        hf_repo="pipecat-ai/smart-turn-v3",
        hf_file="smart-turn-v3.2-cpu.onnx",
    ),
    Model(
        key="face",
        dest="weights/face_detection_yunet_2023mar.onnx",
        what="YuNet face detection + 5 landmarks (OpenCV Zoo)",
        license="Apache-2.0",
        size=232_589,
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        # NOTE: raw.githubusercontent.com serves the Git-LFS *pointer* (131 bytes)
        # for opencv_zoo; the media host serves the real file.
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    Model(
        key="face",
        dest="weights/face_recognition_sface_2021dec.onnx",
        what="SFace 128-d face embedding (OpenCV Zoo)",
        license="Apache-2.0",
        size=38_696_353,
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
    Model(
        key="voiceprint",
        dest="weights/wavlm-base-plus-sv.onnx",
        what="WavLM x-vector speaker embedding (speaker gate). This is the "
             "int8-quantised export: the fp32 file in the same repo is 402 MB.",
        license="MIT",
        size=101_683_453,
        sha256="576bf6017796bdd179824d801b3a355a1dda2451559c6a398562175e33589f68",
        hf_repo="Xenova/wavlm-base-plus-sv",
        hf_file="onnx/model_quantized.onnx",
    ),
    Model(
        key="whisper",
        dest="weights/whisper-turbo-ct2/model.bin",
        what="faster-whisper large-v3-turbo, CTranslate2 int8 (speech-to-text)",
        license="MIT",
        size=1_617_884_929,
        hf_repo="deepdml/faster-whisper-large-v3-turbo-ct2",
        hf_snapshot=["config.json", "model.bin", "preprocessor_config.json",
                     "tokenizer.json", "vocabulary.json"],
    ),
]


def sha256_of(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


def download(url: str, dest: pathlib.Path, size: int = 0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or size or 0)
        seen = 0
        step = max(total // 10, 1) if total else 0
        mark = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            seen += len(block)
            if step and seen >= mark + step:
                mark = seen
                print(f"      {human(seen)} / {human(total)}", flush=True)
    tmp.replace(dest)


def fetch(model: Model) -> bool:
    dest = model.path
    # A fresh clone has no weights/ directory at all.
    dest.parent.mkdir(parents=True, exist_ok=True)
    if model.hf_snapshot:
        from huggingface_hub import snapshot_download

        snapshot_download(model.hf_repo, local_dir=str(dest.parent),
                          allow_patterns=model.hf_snapshot)
        return True
    if model.ultralytics:
        # Let Ultralytics resolve its own asset URL: hard-coding a release tag
        # breaks as soon as they publish a new one.
        import os

        from ultralytics import YOLO

        previous = os.getcwd()
        try:
            os.chdir(dest.parent)
            YOLO(model.ultralytics)
        finally:
            os.chdir(previous)
        return dest.exists()
    if model.hf_repo and model.hf_file:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(model.hf_repo, model.hf_file)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        return True
    download(model.url, dest, model.size)
    return True


def check(model: Model) -> str:
    """'ok' | 'missing' | 'wrong-size' | 'wrong-hash'"""
    path = model.path
    if not path.exists():
        return "missing"
    if model.size and path.stat().st_size != model.size:
        return "wrong-size"
    if model.sha256 and sha256_of(path) != model.sha256:
        return "wrong-hash"
    return "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch ARIA-POC model weights")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--only", nargs="+", metavar="KEY", help="subset: vision vad turn face voiceprint whisper")
    ap.add_argument("--verify", action="store_true", help="check existing files, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download even when present")
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'key':<11} {'file':<46} {'size':>9}  licence")
        for m in MODELS:
            print(f"{m.key:<11} {m.dest:<46} {human(m.size):>9}  {m.license}")
        print("\nSources: Ultralytics assets · snakers4/silero-vad · "
              "pipecat-ai/smart-turn-v3 · opencv/opencv_zoo · "
              "Xenova/wavlm-base-plus-sv · deepdml/faster-whisper-large-v3-turbo-ct2")
        return 0

    wanted = MODELS
    if args.only:
        keys = {k.lower() for k in args.only}
        wanted = [m for m in MODELS if m.key in keys]
        unknown = keys - {m.key for m in MODELS}
        if unknown:
            print(f"unknown key(s): {', '.join(sorted(unknown))}")
            return 2

    problems = 0
    for m in wanted:
        state = check(m)
        if args.verify:
            print(f"{state:<11} {m.dest}")
            problems += state != "ok"
            continue
        if state == "ok" and not args.force:
            print(f"present    {m.dest}")
            continue
        print(f"fetching   {m.dest}  ({human(m.size)}) — {m.what}")
        try:
            fetch(m)
        except (urllib.error.URLError, OSError, RuntimeError, ImportError) as exc:
            print(f"           FAILED: {exc}")
            print("           If this is a TLS/certificate error behind a proxy or "
                  "antivirus, download the file manually in a browser and place it "
                  f"at {m.dest}")
            problems += 1
            continue
        state = check(m)
        if state == "ok":
            print("           ok (hash verified)" if m.sha256 else "           ok")
        elif state == "wrong-hash" and m.sha256 is None:
            print("           WARNING: size fine, checksum differs from what this "
                  "project was measured with")
        else:
            print(f"           {state.upper()} — expected {human(m.size)} at {m.dest}")
            problems += 1

    if problems:
        print(f"\n{problems} model(s) missing or wrong. Run again, or fetch manually.")
        return 1
    print("\nAll requested models present.")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())