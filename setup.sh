#!/usr/bin/env bash
# ARIA-POC one-shot setup for Linux and NVIDIA Jetson.
#
#   ./setup.sh                  # venv + deps + CUDA torch + models + selfcheck
#   ./setup.sh --jetson         # Jetson Orin (NVIDIA torch wheels, not PyPI)
#   ./setup.sh --cpu-only       # no NVIDIA GPU
#   ./setup.sh --skip-models    # code only, fetch the 1.7 GB later
#   ./setup.sh --dev            # + pytest and optional extras
#   ./setup.sh --sys-deps       # also apt-install portaudio/libgl/espeak/ffmpeg
#
# Every step is idempotent: re-run it any time.
set -euo pipefail

VENV=".venv"
PYTHON="${PYTHON:-python3}"
JETSON=0
CPU_ONLY=0
SKIP_MODELS=0
DEV=0
SYS_DEPS=0

while [ $# -gt 0 ]; do
  case "$1" in
    --jetson) JETSON=1 ;;
    --cpu-only) CPU_ONLY=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --dev) DEV=1 ;;
    --sys-deps) SYS_DEPS=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "$0")"
echo "== ARIA-POC setup =="

if [ "$SYS_DEPS" = "1" ]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "-- system libraries (audio, GUI, offline TTS)"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
      libportaudio2 portaudio19-dev libgl1 libglib2.0-0 espeak-ng ffmpeg
  else
    echo "-- apt-get not found; install PortAudio/libGL manually if needed"
  fi
fi

if [ ! -d "$VENV" ]; then
  echo "-- creating virtualenv ($PYTHON)"
  "$PYTHON" -m venv "$VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip wheel >/dev/null

echo "-- runtime dependencies"
"$VPY" -m pip install -r requirements.txt
if [ "$DEV" = "1" ]; then
  "$VPY" -m pip install -r requirements-dev.txt
fi

if [ "$CPU_ONLY" = "1" ]; then
  echo "-- PyTorch (CPU wheels)"
  "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
elif [ "$JETSON" = "1" ]; then
  # JetPack versions its wheels separately from PyPI. v61 = JetPack 6.1; check
  # with:  cat /etc/nv_tegra_release
  JP="${JETPACK_INDEX:-v61}"
  echo "-- PyTorch (NVIDIA Jetson wheels, index jp/$JP)"
  "$VPY" -m pip install torch torchvision \
    --index-url "https://developer.download.nvidia.com/compute/redist/jp/$JP/pytorch"
  echo "-- NOTE: on Jetson the ONNX models intentionally stay on the CPU provider"
  echo "--       (they are tiny), so onnxruntime from PyPI is enough."
else
  echo "-- PyTorch (CUDA 12.6 wheels, works on RTX/GTX with a recent driver)"
  "$VPY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

if [ "$SKIP_MODELS" = "1" ]; then
  echo "-- skipping model download (run: $VPY tools/fetch_models.py)"
else
  echo "-- model weights (1.7 GB, hash-verified)"
  "$VPY" tools/fetch_models.py
fi

echo "-- quick self-check"
"$VPY" tools/selfcheck.py || true

cat <<EOF

Done. Next:
  $VPY main.py --config configs/laptop.yaml          # camera + mic + HUD + voice
  $VPY main.py --config configs/audio_only.yaml      # conversation only
  $VPY tools/selfcheck.py --full                     # full verification
  $VPY -m pytest -q tests                            # 137 tests

On a robot (no monitor) set display.enabled: false in the profile you use.
EOF