# Installing ARIA-POC on another machine

Three supported targets. Windows and Linux x86-64 are the development path;
Jetson Orin Nano is the deployment target and its differences are called out
explicitly (including the bits not yet verified on hardware).

| Target | Status | Notes |
|---|---|---|
| Windows 11 + NVIDIA laptop GPU | ✅ verified (this machine) | CUDA Whisper, DirectShow camera, SAPI fallback TTS |
| Linux x86-64 + NVIDIA GPU | ✅ expected to work | untested here; same wheels as Windows for everything else |
| **Jetson Orin Nano 8 GB** (JetPack 6.x) | ⚠️ prepared, not yet verified on hardware | PyTorch comes from NVIDIA, not PyPI; see the Jetson section |

Disk: ~5.5 GB with the virtualenv and all models (Whisper is 1.5 GB of that).
RAM: 16 GB on the laptop; the Jetson's 8 GB unified memory is the binding
constraint, which is why the resource work in `ARCHITECTURE.md` matters.

---

## 0. Prerequisites

- **Python 3.12** (3.10–3.12 all work; 3.13+ has no wheels for some deps yet)
- **Git**
- ~10 minutes and a network connection (models are 1.7 GB)

## 1. Windows (laptop / desktop)

```powershell
git clone https://github.com/FadiSouihi/aria-poc.git
cd aria-poc
.\setup.ps1                 # venv + deps + CUDA torch + models + selfcheck
```

Useful switches:

```powershell
.\setup.ps1 -SkipModels     # code only; fetch models later
.\setup.ps1 -CpuOnly        # no NVIDIA GPU: CPU torch (vision/STT will be slow)
.\setup.ps1 -Dev            # also install pytest and the optional extras
.\setup.ps1 -Python "py -3.12"
```

Then talk to it:

```powershell
.venv\Scripts\python.exe main.py --config configs\laptop.yaml
```

## 2. Linux x86-64

```bash
git clone https://github.com/FadiSouihi/aria-poc.git
cd aria-poc
./setup.sh                  # venv + deps + CUDA torch + models + selfcheck
./setup.sh --cpu-only       # no NVIDIA GPU
./setup.sh --skip-models    # code only
```

`setup.sh --sys-deps` also installs the system libraries the audio/GUI stack
needs (`libportaudio2`, `libgl1`, `espeak-ng`, `ffmpeg`) via `apt` — do it once if
you hit "PortAudio library not found" or OpenCV GUI errors.

## 3. Jetson Orin Nano (JetPack 6.x)

The differences that actually bite, in order:

1. **PyTorch does not come from PyPI.** NVIDIA ships aarch64 builds per JetPack
   release. `setup.sh --jetson` uses:
   ```bash
   pip install torch torchvision \
       --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch
   ```
   (`v61` = JetPack 6.1 — check your release with `cat /etc/nv_tegra_release` and
   adjust, e.g. `v60` for JetPack 6.0.) The exact laptop pin
   (`torch==2.14.0+cu126`) does **not** exist for aarch64; the code does not
   depend on that version.
2. **The ONNX models run on the CPU anyway.** Silero VAD, Smart Turn, YuNet,
   SFace and WavLM are all created with `CPUExecutionProvider` on purpose (they
   are tiny; GPU round-trips cost more than they save), so plain `onnxruntime`
   from PyPI is enough — you do not need `onnxruntime-gpu` unless you later move
   YOLO to ONNX.
3. **CTranslate2** ships aarch64 wheels for recent versions. If `pip` cannot find
   one for your Python, either use a supported Python or build from source
   (`pip install ctranslate2 --no-binary :all:`, needs cmake); the fallback is
   NVIDIA's NGC container.
4. **Run headless.** The debug HUD needs a display; on a robot set
   `display.enabled: false` in the profile (or use `configs/audio_only.yaml` for
   the lowest-latency conversation loop). Use `opencv-python-headless` if you
   never want GUI code linked in.
5. **Audio.** `sudo apt install libportaudio2` for `sounddevice`. PortAudio on
   Jetson works with ALSA/PulseAudio; list devices with
   `python -c "import sounddevice; print(sounddevice.query_devices())"` and pin
   `mic.device` / `tts.device` if the default is wrong.
6. **TTS.** The default provider is `edge-tts` (cloud, best quality, needs
   internet). Offline, install `espeak-ng` and switch the provider to `pyttsx3`,
   or keep `fake` for benchmarks. Windows-only SAPI5 is not available on Linux —
   that is exactly why the TTS provider is a config swap.
7. **Performance expectations.** Whisper `large-v3-turbo` int8 on GPU is the
   conversational path. On CPU it runs at **RTF ≈ 9** (a 4.7 s sentence takes
   ~45 s), so a CPU-only fallback needs a smaller model — a config change
   (`stt.model`), not a code change. YOLO is the other compute sink; TensorRT
   INT8 export is the Phase 7 plan.

```bash
git clone https://github.com/FadiSouihi/aria-poc.git
cd aria-poc
./setup.sh --jetson --sys-deps
```

## 4. Model weights (any platform)

Weights are **not** in git (1.7 GB, and Ultralytics is AGPL — keeping them out
also keeps the repo's licensing simple). `setup.ps1` / `setup.sh` run this for
you; run it by hand any time:

```bash
python tools/fetch_models.py            # everything missing, hash-verified
python tools/fetch_models.py --list     # what, from where, licence, size
python tools/fetch_models.py --only vision vad turn face    # ~55 MB, no Whisper yet
python tools/fetch_models.py --verify   # check what you have; downloads nothing
```

Each file has a pinned SHA-256, so every machine gets the same bytes the
benchmarks were measured with. If a download fails behind a proxy or TLS-
inspecting antivirus, the script says which file to fetch manually and where to
put it.

## 5. Verify the install

```bash
python tools/selfcheck.py            # 7 quick checks
python tools/selfcheck.py --full     # + real vision and audio benchmarks on fixtures
python -m pytest -q tests            # 137 tests (a few skip if models are absent)
```

`selfcheck --full` ending in `ALL CHECKS PASSED (9/9)` means the camera/mic,
event flow, telemetry, models and both benchmarks are working on that machine.

## 6. Run it

```bash
python main.py --config configs/laptop.yaml        # camera + mic + HUD + voice
python main.py --config configs/audio_only.yaml    # conversation only (lowest latency)
python main.py --config configs/laptop.yaml --duration 30
python tools/measure_load.py --config configs/laptop.yaml   # CPU/RSS/GPU cost
python tools/bench_audio.py --manifest fixtures/audio/manifest.yaml --assert
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Library cublas64_12.dll is not found` | CUDA runtime missing. The app pre-loads torch's CUDA libs and falls back to CPU int8; install the CUDA 12 runtime for GPU Whisper. |
| `PortAudio library not found` | `sudo apt install libportaudio2` (Linux) — bundled on Windows. |
| `qt.qpa.plugin: could not load the Qt platform plugin` | Headless machine: set `display.enabled: false`, or `apt install libgl1 libglib2.0-0`. |
| `operator torchvision::nms does not exist` | torch and torchvision versions disagree. Reinstall both from the same index in one command. |
| Nothing happens when I talk | First ~12 s are warm-up (Whisper load). Watch `logs/aria.log` for `STT ready`. |
| `Could not establish trust relationship` during download | TLS-inspecting proxy/antivirus. Download the file in a browser and place it at the path the script prints. |
| Console shows `?` instead of Arabic | Console codepage; logs are UTF-8 on disk (`logs/aria.log` is fine). |
| Camera stretched / mirrored wrong | `display.flip` (`none`/`horizontal`/`vertical`/`both`) and `display.fit`; press **f** in the window to cycle flip live. |

## What is deliberately not automated

- **Face/voice enrolment** — `python tools/enroll.py --name <person>` and
  `tools/enroll_voice.py` write to `data/`, which is gitignored because it is
  biometric data.
- **TensorRT / INT8 export** (Phase 7) and **LLM routing** (Phase 3) do not
  exist yet, so nothing to install for them.
- **Jetson hardware verification** — the steps above are the documented path;
  this project has not yet been run on an Orin Nano. Expect the model *loading*
  and `onnxruntime` details to be the friction points, not the app itself.