# ARIA-POC — Quickstart & Status

Prototype of a unified-perception, session-aware security/receptionist robot
(POC on this laptop, deployment target Jetson Orin Nano 8 GB).
The full plan lives in `ROADMAP.md` (and the rendered PDF report); how the
code is organized is in `ARCHITECTURE.md`.

## Quickstart

**Windows**

```powershell
git clone https://github.com/FadiSouihi/aria-poc.git
cd aria-poc
.\setup.ps1                                # venv + deps + CUDA torch + models + selfcheck

.venv\Scripts\python.exe main.py --config configs\laptop.yaml --duration 60
.venv\Scripts\python.exe run_tests.py      # contract suite (147 tests, all green)
```

**Linux / Jetson Orin Nano**

```bash
git clone https://github.com/FadiSouihi/aria-poc.git
cd aria-poc
./setup.sh                 # add --jetson on an Orin Nano, --cpu-only without a GPU
```

Full detail per platform — including the Jetson gotchas (PyTorch comes from
NVIDIA, headless display, PortAudio, offline TTS) and a troubleshooting table —
is in [`INSTALL.md`](INSTALL.md). Model weights are **not** in git (1.7 GB);
`setup.ps1` / `setup.sh` fetch and hash-verify them, or run
`python tools/fetch_models.py` yourself.

Talk to it: the mic → Silero VAD → end-of-turn → Whisper → gate → TTS loop runs
in the same app. ARIA only answers when someone is engaged (see Phase 2 below).
`requirements.txt` explains why torchaudio is deliberately absent.

**Resource cost, measured** (`python tools/measure_load.py --config ...`). Steady
state, this laptop, nothing else running:

| Profile | CPU before | CPU now |
|---|---|---|
| `laptop.yaml` — full stack, idle | 9.64 cores (80%) | **0.54 cores (4%)** |
| `bench_talk.yaml` — full stack, conversing | — | **0.70 cores (6%)** |
| `audio_only.yaml` — conversation only, idle | 6.93 cores (58%) | **0.14 cores (1%)** |

The big win was not doing less work — it was stopping ONNX Runtime and the HUD
from *spinning*. Details in `ARCHITECTURE.md` ("Resource budget"); every lever is
a YAML key (`ort_threads`, `busy_detect_hz`, `timeline.sample`, `display.target_fps`),
so it is tunable or revertible without touching a code path.

**It speaks your language, it does not translate.** Speech is transcribed with
language auto-detection (`task="transcribe"`), so French stays French and Arabic
stays Arabic — and the reply uses a voice for that language (`fr` →
`fr-FR-DeniseNeural`, `ar` → `ar-MA-MounaNeural`, ...). Add or change languages
in `configs/laptop.yaml` under `tts.voices` and `responder.templates`.

**Where the time goes** (live, French fixture, measured with
`tools/report_latency.py` + `logs/aria.log`): VAD silence 320 ms → end-of-turn
62 ms → Whisper 610 ms (of which ~300 ms is language detection) → gate 0 ms →
first audio **0.0 ms** (reply prefix pre-synthesized) → 0.9 ms gap before the
rest of the reply. Total ≈ **0.9 s** from "you stop talking" to ARIA speaking.
With the full vision profile it is ~1.4 s because vision shares the GPU — use
`--config configs\audio_only.yaml` when testing conversation itself. Pin
`stt.language` to skip detection and save ~300 ms; set `tts.warm_phrases` for
your own canned lines.

**First ~8 seconds after launch are warm-up** (Whisper loads + warms in the
background). Speech during warm-up is dropped rather than answered late; set
`mic.config.wait_for_ready: ["stt"]` if you would rather capture start only once
the transcriber is warm.

## Verify everything yourself

One command checks the whole stack (tests, boot, event flow, telemetry, models):

```powershell
python tools/selfcheck.py
```

Expect seven PASS lines and exit code 0 — or add `--full` for two more checks
that benchmark the real vision **and** audio stacks on bundled fixtures:

```powershell
python tools/selfcheck.py --full       # + VRAM/latency + EOU/false-response benchmarks
```

Or run the pieces manually:

```powershell
python run_tests.py                                            # 147 contract/integration tests
python main.py --config configs/laptop.yaml --duration 60      # webcam + YOLO + HUD + voice (q closes window)
python tools\bench_vision.py --source file --path fixtures/bus.jpg --seconds 8 --assert
python tools\bench_audio.py --manifest fixtures\audio\manifest.yaml --assert
python tools\measure_load.py --config configs\laptop.yaml      # CPU cores busy / RSS / GPU, steady state
python tools\record_fixture.py --name group --seconds 30       # capture a camera fixture
python tools\record_audio_fixture.py --name my_voice --seconds 6   # capture a mic fixture
python tools\say.py --name demo --text "Where is the library?"     # synthesize a speech fixture
python tools\enroll.py --name fedi                             # enroll a face identity
python tools\enroll_voice.py --name fedi --seconds 8           # enroll a voice-print
python tools\calibrate_voice.py                                # same/cross-speaker similarity check
python tools\report_latency.py --runs 3                        # per-utterance pipeline timings
Get-Content logs\aria.log -Tail 5                              # JSON log lines
Get-Content logs\timeline_full.jsonl -Tail 5                   # flight-recorder events
```

`configs\quiet.yaml` is the same laptop stack with a silent console, so a script
gets nothing but the JSON metrics summary on stdout (used by `measure_load.py`).
`configs\bench_talk.yaml` loops a speech fixture through the mic so every turn is
real while vision stays live — that is how the "conversing" load number above is
measured, and it is the profile to use when checking for performance regressions.

The debug window (`configs\laptop.yaml` → `display`) mirrors the view like a
selfie camera (`flip: horizontal`; press **f** to cycle none/horizontal/vertical/
both, **q** to quit) and keeps the camera's aspect ratio instead of stretching a
4:3 feed into a 16:9 window (`fit: letterbox`). Mirroring is display-only: the
pipeline, track boxes and face galleries keep using raw frames. Set
`display.enabled: false` for headless runs.

`report_latency.py` is the tool for "why did it feel slow/buffered": it prints
speech-end → transcript → decision → speech for each utterance and flags
`QUEUED` (you spoke again while the previous turn was still being transcribed)
and `STALE` (a transcript arrived long after the speech ended).

Replay a fixture instead of the webcam: set `camera.source: file` and
`camera.path: fixtures/single/video.mp4` in the profile (loop by default).
Replay speech instead of the mic: set `mic.source: file` and
`mic.path: fixtures/audio/greeting.wav`.

## Layout

```
main.py             app shell: config → telemetry → bus → registry → run
run_tests.py        pytest wrapper
tools/selfcheck.py  one-command health check (--full adds vision + audio benches)
tools/record_fixture.py  capture replayable camera fixtures
tools/record_audio_fixture.py  capture replayable mic fixtures
tools/say.py        synthesize speech fixtures with the TTS providers
tools/enroll.py     enroll face identities into data/faces/
tools/enroll_voice.py    enroll voice-prints into data/voices/
tools/bench_vision.py    vision latency/VRAM benchmark with budget assert
tools/bench_audio.py     EOU latency + false-response-rate + language-preservation benchmark
tools/calibrate_voice.py same/cross-speaker similarity calibration
tools/report_latency.py  per-utterance pipeline timings from logs/aria.log
tools/fetch_models.py    download + hash-verify the model weights (not in git)
tools/measure_load.py    CPU cores busy / RSS / GPU utilisation of a profile
configs/            laptop.yaml (runtime) · audio_only.yaml · quiet.yaml (bench)
                    bench_talk.yaml (looping conversation) · test.yaml (hermetic)
aria/core/          events · context · config · event_bus · service · registry
                    watchdog · governor · readiness · onnx (session threading)
                    telemetry (logging/timeline/metrics)
aria/perception/    framestore · camera (fake/webcam/file) · vision (YOLO+tracker)
                    face (YuNet+SFace+voting) · scene (per-track FSM) · tamper
                    display (HUD) · transform (flip/letterbox)
aria/audio/         store (ring buffer) · mic (device/file/fake) · vad (Silero v5 ONNX)
                    turn (3-tier EOU + Smart Turn) · stt (faster-whisper) · gate
                    (addressed speech) · voiceprint (WavLM x-vectors) · tts (edge/SAPI)
                    responder (Phase 3 stub)
tests/              contract suites + hermetic fakes (helpers, dummies)
weights/            NOT in git — python tools/fetch_models.py (1.7 GB, hash-pinned)
fixtures/           replayable fixtures (bus.jpg, single/, audio/) — committed
logs/               structured JSON logs + timeline_full.jsonl (flight recorder)
docs/               rendered roadmap report (PDF/HTML)
setup.ps1 / setup.sh / INSTALL.md   one-shot install per platform (Windows/Linux/Jetson)
```

## Phase 2 exit criteria — all demonstrated

Measured on this laptop (RTX 4050, `tools/bench_audio.py --assert` + a live run):

- [x] **Three-tier end-of-turn** — VAD silence → Smart Turn v3.2 semantic model
      (`p_turn` 0.99 on a finished sentence) → hard timeouts; decision logic is a
      pure `EouTracker` class with 8 unit tests
- [x] **EOU latency**: classify p50 **47–63 ms** (bench) / 83–94 ms (live), budget 800 ms
- [x] **False-response rate 0.00** (0/2 fixtures) — stationary noise and a short
      click produce **no turn at all**, not a rejected answer
- [x] **Miss rate 0.00** (4/4 real questions transcribed and accepted)
- [x] **STT**: faster-whisper `large-v3-turbo` (CTranslate2, int8_float16 on GPU),
      RTF **0.24–0.31** on 3–5 s utterances; CPU demotion on `GovernorThrottled`
- [x] **Speech is never translated** — `language: auto` + `task="transcribe"`;
      bench asserts transcript language *and* script per fixture:
      EN "Hello Aria, where is the library please?", FR "Bonjour Aéria, où se
      trouve la bibliothèque", AR "مرحبا، أين تقع المكتبة من فضلك؟"
- [x] **Language-matched voices** — `tts.voices` maps `en/fr/ar/es` to
      edge-tts voices; the bench asserts the reply voice (`fr-FR-DeniseNeural`,
      `ar-MA-MounaNeural`) and a localised reply wrapper
- [x] **Addressed-speech gate** — rejects empty/too-short speech, bystanders and
      speech with nobody engaged (`no_one_engaged` was the live-run rejection)
- [x] **Voice-prints** — WavLM x-vectors (512-d, raw-waveform ONNX), calibrated
      same-speaker 0.92 vs other-speaker 0.45–0.49, gallery in `data/voices/*.npz`
- [x] **TTS with barge-in** — edge-tts default, offline SAPI5 fallback, clause
      pipelining (next clause synthesized while the current one plays, LRU cache),
      playback stops on a *genuine* interruption; half-duplex ducking plus an
      echo-tail guard keeps ARIA from hearing her own speaker, and a barge-in must
      clear a loudness floor (`barge_in_min_dbfs`) so her own quieter echo is
      ignored instead of stopping the reply (`tts.barge_in_ignored_quiet`)
- [x] **No reply backlog, and no self-conversation** — STT is **single-flight**:
      a turn that arrives while Whisper is busy is *dropped* (`stt.dropped_busy`)
      rather than queued, so noise with pauses cannot build a backlog, and a turn
      older than `max_stale_s` is dropped too (checked at admission *and* after the
      readiness wait); live `speech_end→transcript` p50 **880 ms** (was 1144–4883 ms)
- [x] **Reply starts instantly, with no gap mid-reply** — the reply prefix is
      pre-synthesized at start-up (`tts.warm_phrases`) so the first clause costs
      **0.0 ms**, clauses are played through **one** low-latency output stream
      (opening one per clause was audible dead air: `tts.clause_gap_ms` **0.9 ms**
      now), and the gate no longer waits for a voice-print that cannot exist
      (`transcript→decision` 83–145 ms → **0 ms**)
- [x] **147/147 tests green** (138 hermetic contract suites + 9 real-model
      integration)
- [x] Live run: 487 mic → 487 VAD chunks, 2 speech events → 2 turns → 2
      transcripts → 2 accepted → 2 replies → 3 spoken clauses, **0 restarts**,
      935 events, **0 dropped**

### Deliberate dependency decisions (Phase 2)

`torchaudio` is not installed and the `silero-vad` package is not used: torchaudio
has no build for the pinned torch 2.14, and it downgrades torch when installed,
which breaks torchvision's NMS ops. Silero VAD therefore runs directly on its ONNX
graph (with the 64-sample context Silero v5 requires) and Smart Turn's whisper
log-mel is computed in numpy. Voice-prints use a raw-waveform WavLM ONNX model
because SpeechBrain hard-imports torchaudio and Resemblyzer needs `webrtcvad`
(MSVC build tools). All four swaps sit behind the same interfaces, so they can be
reverted by config once a compatible wheel exists.

## Phase 1 exit criteria — all demonstrated

- [x] 0/1/N people through ONE code path (per-track set diffing, no mode flags)
- [x] Track IDs persist across frames (`persist=True`) with a 3s occlusion
      buffer (`track_buffer: 45` @ 15fps) in `configs/tracker_bytetrack.yaml`
- [x] Identity = multi-frame voting (3 consistent matches + margin) with
      periodic re-verification (5s) and `data/faces/*.npz` galleries
- [x] Per-track FSM `PRESENCE → NEAR → ENGAGED → departed` (dwell timers,
      centrality+proximity), transitions logged
- [x] Tamper detection (`TamperDetected`/`TamperCleared`) + camera hot
      reconnect (`DeviceLost` → backoff retries → `DeviceRestored`)
- [x] bench_vision within budget: p50 ≈ 17–32 ms detect latency, ≈ 41 MB
      torch VRAM on the RTX 4050 (budget 2600 MB)
- [x] 44/44 tests green (hermetic contract suites + fixture integration)

Earlier phases: Phase 0 (core skeleton — event bus, service lifecycle,
registry, watchdog, governor, telemetry) exit criteria are recorded in
`ROADMAP.md` §10; those capabilities are now part of the base stack.

## Scope decision (project, 2026-09-19; revised 2026-10-05 after live testing)

**Revised.** Live testing showed that locking the pipeline to English was the wrong
call: the robot already *heard* French and Arabic correctly, but the old STT config
forced `language="en"`, so Whisper translated on the way out. The pipeline now
transcribes in the spoken language (`language: auto`, `task="transcribe"`) and
replies with a language-matched voice — nothing is ever translated, and the text
that comes back equals the words that were spoken. Verified by
`tools/bench_audio.py` for EN/FR/AR with script assertions.

Still out of scope: **dialogue design** stays English-first (the interaction text,
intents and prompts are English), and Darija-specific tuning is not attempted —
Moroccan Arabic is handled as standard Arabic, with `ar-MA-*` voices available.
Adding a language stays a configuration change (`tts.voices`,
`responder.templates`), not a code change. See ROADMAP §5.2.

## Next phase

Phase 3 — Cognition: dialogue FSM on the session layer, `AttentionPolicy` for
groups (who to answer when several people are present), LLM router with
guardrails/PII masking, and replacing `aria/audio/responder.py` (the Phase 2
stub) with the real reply source. Memory cards (Phase 4) plug into the same
`UtteranceAccepted` → reply path, so the recall feature ("did you find the
library last time?") lands as a new subscriber, not a rewrite.
