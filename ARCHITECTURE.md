# ARIA-POC — Architecture (Phases 0–2)

Companion to `ROADMAP.md`. This document explains how the running system is
wired and the rules that keep features swappable and testable.

## Runtime flow

```
load_config(profile)
  → setup_logging(dir, level, console)          aria.core.telemetry.logging
  → Metrics()                                   aria.core.telemetry.metrics
  → EventBus(metrics, maxsize)                  aria.core.event_bus
  → TimelineRecorder(dir).install(bus)          aria.core.telemetry.timeline
  → Registry(bus, config).build()               aria.core.registry
      for each enabled service in config:
        import class → attach(bus, config) → schema-validate → set_registry (watchdog)
  → await bus.start() → await registry.start_all()
  → run until duration / Ctrl+C
  → registry.stop_all() → bus.stop() → print metrics + bus summary
```

## Module map

| Module | Responsibility |
|---|---|
| `core/events.py` | `Event` dataclass + documented event registry (`REGISTRY`); unknown names are allowed but counted/warned |
| `core/context.py` | Correlation context (`session_id`, `track_id`, `utterance_id`) via contextvars; stamped onto every event and log line |
| `core/config.py` | YAML profiles with `extends` inheritance and dotted access |
| `core/event_bus.py` | Typed pub/sub; per-subscriber bounded queues with `drop_new` / `drop_oldest` / `block` overflow policies; taps; failing subscribers never break publishers |
| `core/service.py` | `Service` base: lifecycle state machine, heartbeat, config schema, crash containment (crashed task → `ServiceCrashed`, app keeps running) |
| `core/registry.py` | Plugin registry: `module:Class` + `enabled` + `config` per service; in-place `restart()` |
| `core/watchdog.py` | Heartbeat staleness + crash detection → bounded restarts with backoff → `ServiceRestarted`; gives up loudly via `Anomaly` |
| `core/governor.py` | RSS sampling with hysteresis → `GovernorThrottled`; consumers subscribe (decoupled) |
| `core/readiness.py` | Tiny in-process readiness registry (`mark_ready` / `wait_ready`) so a replay source can hold off until a slow service (Whisper: load + warmup ≈ 8 s) is actually able to consume |
| `core/onnx.py` | One place that builds every ONNX Runtime session with low-noise threading (`intra_op_num_threads`, `inter_op=1`, sequential, `allow_spinning=0`) and caps OpenCV's parallel loops; per-model policy in `DEFAULTS`, overridable per service (`ort_threads`, `ort_spinning`, `opencv_threads`) |
| `core/telemetry/logging.py` | Structured JSON logs (one logger per service) + ID masking + rotating file sink |
| `core/telemetry/timeline.py` | Full JSONL event tap (one buffered handle, flushed on interesting events/anomalies/close) + in-memory ring buffer + auto-dump on anomalies; `sample` keeps 1-in-N of high-rate metadata or drops it |
| `core/telemetry/metrics.py` | Counters + histograms (nearest-rank percentiles) |
| `main.py` | App shell: config → telemetry → bus (+ timeline) → registry → run → graceful shutdown with a metrics summary. `runtime.exit_on_source_end` lets a replay run stop when the fixture has been *processed* (plus a grace period) instead of after a guessed sleep — a fixed budget used to cut the run short before the turn completed |
| `perception/framestore.py` | In-process pixel exchange: the bus carries only metadata; consumers grab the *latest* pixels from a bounded store at their own pace (single writer, many readers) |
| `perception/camera.py` | Frame source: `fake` (hermetic) / `webcam` / `file` (fixtures) with a threaded grabber; read failures → `DeviceLost` + bounded hot reconnect → `DeviceRestored` (no app restarts) |
| `perception/vision.py` | YOLO + ByteTrack via Ultralytics — ONE code path for 0/1/N people (person class only); diffs track sets per cycle → `TrackAppeared`/`TrackLost` + `SceneTick`; model/tracker are config (swap ByteTrack→BoT-SORT by editing YAML) |
| `perception/face.py` | YuNet (landmarks) + ONNX embedder (default SFace 128-d; ArcFace r50 is a config swap) + gallery voting: identity only after N consistent matches with a margin; periodic re-verification; galleries in `data/faces/*.npz` |
| `perception/scene.py` | SceneManager: per-track FSM `PRESENCE → NEAR → ENGAGED → (absent) → departed`; proximity+centrality dwell timers; consumes `SceneTick`+`FaceMatched`, publishes `TrackStates` |
| `perception/tamper.py` | Cheap frame statistics (uniformity/darkness streaks) → `TamperDetected`/`TamperCleared`; pure logic class, unit-testable without a camera |
| `perception/display.py` | Debug HUD window (laptop profile only): boxes+states+identities+tamper banner; `flip` mirrors/flips the *view* (default `horizontal`, press **f** to cycle) while the pipeline keeps raw frames — so face galleries stay valid; `fit: letterbox` scales into the window preserving aspect ratio instead of stretching (a 4:3 camera in a 16:9 window was visibly distorted); redraws only when the source frame id changes (`target_fps` defaults to the camera rate); degrades to a logged no-op without a GUI |
| `perception/transform.py` | Pure view transforms: `apply_flip` (none/horizontal/vertical/both), `flip_bbox` (map boxes into the flipped view), `fit_to_window` (aspect-preserving letterbox); no camera or GUI needed, so they are unit tested |
| `audio/store.py` | Audio twin of `FrameStore`: bounded ring buffer of 16 kHz float32 samples with *exact* retention (head chunk is trimmed, not just dropped), absolute sample indices so any consumer can request `read_since(idx)` |
| `audio/mic.py` | Capture source `device` / `file` / `fake` with a threaded grabber; publishes `AudioChunk` metadata only; hot reconnect on device loss; optional `wait_for_ready: ["stt"]` gates a replay until dependencies are warm; a non-looping fixture announces `AudioSourceFinished` so replay runs (benchmarks, regression tests) end deterministically |
| `audio/vad.py` | Silero VAD v5 **directly on ONNX** (no torch/torchaudio) with the 64-sample context v5 requires; pure `VadStateMachine` (threshold, start/silence frames, hard timeout) publishes `SpeechStarted`/`SpeechEnded` with absolute sample indices; **half-duplex ducking** ignores mic audio while TTS plays (config `duck_while_speaking`) |
| `audio/turn.py` | Three-tier end-of-turn: pure `EouTracker` (`complete` / `wait` / `discard`) driven by a semantic `EouClassifier` — `SmartTurnClassifier` (ONNX + numpy whisper log-mel, right-aligned window) with `HeuristicEouClassifier` as config fallback; publishes `TurnCompleted` |
| `audio/stt.py` | faster-whisper (CTranslate2), `task="transcribe"` with `language: auto` — **it never translates**; subscribes *before* loading its model (background load + warmup) so no turn is lost, then serves a bounded turn queue, dropping turns that are already stale (`max_stale_s`, re-checked at dequeue); publishes `UtteranceHeard` with `language`/`language_probability`/`queue_wait_s`; demotes to CPU on `GovernorThrottled` |
| `audio/gate.py` | Pure `AddressedSpeechGate` + service: accepts only speech that is long enough, from an engaged person, and not a known *other* speaker; publishes `UtteranceAccepted` / `UtteranceRejected` with a machine-readable reason (the false-response metric) |
| `audio/voiceprint.py` | `VoiceGallery` (cosine match against `data/voices/*.npz`) + provider chain `wavlm` (raw-waveform ONNX x-vectors) → `resemblyzer` → `speechbrain`; publishes `VoiceprintIdentified`; stays idle (gate degrades to engagement-only) if no provider loads |
| `audio/tts.py` | TTS providers `edge` (default) / `sapi` (offline) / `fake` (hermetic) behind one interface, each taking `(text, language, voice)`; `tts.voices` maps a language code → voice (config-only language support, missing code → default + `tts.no_voice.<code>`); long replies are split into clauses and **pipelined** (clause *n+1* is synthesized while *n* plays) with an LRU cache, and all clauses play through **one** low-latency output stream (reopening per clause was audible dead air, tracked as `tts.clause_gap_ms`); `warm_phrases` pre-synthesizes canned lines at start-up so a reply's first clause is instant; one `SpeechSynthesized` per clause keeps ducking/barge-in accurate; a failed primary falls back offline for a **cooldown**, never permanently (an offline voice cannot speak Arabic/French) |
| `audio/responder.py` | Phase 3 placeholder: `UtteranceAccepted` → templated `SpeakRequest`, so the audio loop is demonstrable end-to-end before the dialogue layer exists; templates are per-language (`templates: {fr: ...}`) and the language travels with the reply, so ARIA answers in the language she was spoken to |

## Event registry (Phase 0–2 vocabulary)

Perception: `Frame`, `TrackAppeared`, `TrackLost`, `FaceMatched`,
`TamperDetected`, `TamperCleared`, `SceneTick` (raw vision snapshot),
`TrackStates` (SceneManager's enriched snapshot) · Audio in: `AudioChunk`
(metadata; samples live in `AudioStore`), `SpeechStarted`, `SpeechEnded`,
`TurnCompleted`, `UtteranceHeard` · Audio decisions: `UtteranceAccepted`,
`UtteranceRejected`, `VoiceprintIdentified`, `SpeakRequest`,
`SpeechSynthesized`, `BargeIn` · Session: `SessionOpened`, `SessionClosed` ·
Supervision: `DeviceLost`, `DeviceRestored`, `AudioSourceFinished`, `ServiceCrashed`,
`ServiceRestarted`, `GovernorThrottled`, `Anomaly`.

Rules: publishing an unregistered name works but is warned/counted once —
adding an event is a one-line registry change (keeps the vocabulary
documented and reviewable). Services must unsubscribe in `on_stop` so a
watchdog restart does not duplicate subscriptions.

## Audio pipeline (Phase 2)

```
mic ──AudioChunk──▶ AudioStore (ring buffer)
                      │
        VadService ───┘ walks each 512-sample frame once (cursor), 64-sample
          │            context per call, ducked while TTS plays
          ├──SpeechStarted──▶ (barge-in path → TtsService stops playback)
          └──SpeechEnded ──▶ TurnDetectorService
                               tier 1: VAD silence already elapsed
                               tier 2: Smart Turn v3.2 p_turn ≥ threshold → complete
                               tier 3: pending_timeout / max_utterance → complete
                                       short + low confidence → discard (noise)
                               └──TurnCompleted──▶ SttService (turn queue)
                                                    │  drop if already stale
                                                    │  (max_stale_s, at enqueue
                                                    │   and again at dequeue)
                                                    └─UtteranceHeard(language)──▶
                                                    VoiceprintService ──▶ Gate
Gate ──UtteranceAccepted──▶ Responder(stub, language-aware) ──SpeakRequest──▶ TtsService
     └─UtteranceRejected(reason) → metric, no reply
```

Why this shape: the *decision* that a turn ended is separated from *detecting*
silence, because "the user paused to think" and "the user finished" look
identical to a VAD. That is the FUNC-14 failure class, and it is why the
false-response rate is measured (`tools/bench_audio.py`) rather than assumed.

Three rules keep the loop from drifting out of sync with the person speaking:

1. **The turn queue is bounded and staleness-checked.** If transcription falls
   behind (or the model is still warming up), a turn whose speech ended more than
   `max_stale_s` ago is *dropped* rather than answered late. Answering an old
   question after a new one is what "the voice gets buffered" felt like
   (`stt.dropped_stale*`, `QUEUED`/`STALE` in `tools/report_latency.py`).
2. **Text is never translated.** `language: auto` + `task="transcribe"`; the
   detected language rides along on `UtteranceHeard` and `SpeakRequest`, where it
   selects a voice (`tts.voices`) and a reply template — a *voice* choice, not a
   translation step. Text spoken equals text received.
3. **Replies are pipelined, not monolithic.** Clause *n+1* is synthesized while
   *n* plays, so time-to-first-audio is one clause rather than the whole reply,
   and each clause publishes `SpeechSynthesized` just before it plays (which is
   also what extends the VAD duck window and what barge-in interrupts).

## Service contract

- Lifecycle: `new → initialized → starting → running → stopping → stopped`
  (or `crashed` from anywhere). `start()` is idempotent; `stop()` then
  `start()` is the recovery path the watchdog uses.
- Config schema declared per class (`key: (default, types)`): defaults are
  filled, unknown keys are dropped with a warning, wrong types rejected at
  attach time. Every service accepts `heartbeat_interval`.
- Health: `ok` while running with a fresh heartbeat, `degraded` when stale,
  `down` when crashed. `last_heartbeat` is set at start so health is honest
  immediately.
- Crash containment: a failing spawned task flips state to `crashed` and
  publishes `ServiceCrashed` — nothing else in the app is affected.

## Policies

- **Backpressure:** under overload the bus drops *frames* (`drop_new` /
  `drop_oldest`), never `block`-policy events; drops are counted per
  subscription and surfaced in metrics and the shutdown summary.
- **Telemetry:** every log line is one JSON object with `ts / level /
  service / msg / fields / context`; long digit runs are masked
  (`[REDACTED-ID]`) so logs never become a data-leak channel. The timeline
  taps every event; anomalies (crash, throttle, tamper, device loss) auto-dump
  the last ~5 minutes of events to `logs/timeline_<ts>_<reason>.jsonl`.
  Fidelity is configurable (`timeline.sample`): the tap runs *inline on the
  publishing task*, so `Frame`/`AudioChunk` (≈46 events/s) are sampled or dropped
  in the demo profile and kept at full rate in test/bench profiles. Writes are
  buffered on one open handle and flushed on a timer, on `close()`, and on any
  anomaly — the old code opened, wrote and closed the file once per event.
- **Watchdog margin:** a service is only called stale after
  `max(stale_after, heartbeat_interval * stale_safety_factor)` (default 3×), so a
  service that beats every 5 s cannot be restarted for being 5 s late — heavy GPU
  work elsewhere (Whisper) used to trip exactly that. `Service.health()` uses the
  same 3× convention, so health and restarts agree.
- **Startup order & readiness:** a slow model load must never cost an event.
  `SttService` subscribes in `on_start` and loads/warms its model in a background
  task (`core/readiness.mark_ready("stt")` when done); its config block is listed
  *first* so the load overlaps vision startup. Replay sources can additionally
  wait (`mic.wait_for_ready: ["stt"]`). Anything spoken before warm-up is dropped
  as stale — never answered minutes later.
- **Testing:** every service has a contract suite; fakes live in
  `tests/helpers.py` / `tests/dummies.py`; `configs/test.yaml` runs the whole
  stack hermetically (fast timers, tiny queues, quiet console). Policy suites
  (`test_stt_logic.py`, `test_tts.py`, `test_watchdog.py`) inject a fake engine
  and assert the *behaviour* — stale drops, language pass-through, voice
  selection, clause pipelining, cache hits, restart margins — without models.

## Resource budget (measured, not assumed)

`tools/measure_load.py` runs a profile, skips the start-up burst, then reports
CPU-seconds per wall second ("cores busy"), RSS and GPU utilisation. Measured on
the dev laptop (Core 5 210H, 12 logical CPUs, RTX 4050 6 GB):

| Profile | before | after | what it is |
|---|---|---|---|
| `laptop.yaml` idle | 9.64 cores (80%) | **0.54 cores (4%)** | camera + YOLO + face + HUD + audio, nobody around |
| `laptop.yaml` talking (`bench_talk.yaml`) | — | **0.70 cores (6%)** | vision live while turns loop through VAD→STT→TTS |
| `audio_only.yaml` idle | 6.93 cores (58%) | **0.14 cores (1%)** | conversation loop alone |

The dominant cost was never the models — it was **spinning**:

- **ONNX Runtime session threading** (`aria/core/onnx.py`). ORT defaults to one
  thread per physical core *plus* spin-waiting. Silero VAD calls a tiny graph
  ~31×/second, so it was waking and spinning 8 threads for microseconds of math,
  continuously, forever. Now every session is built with
  `intra_op_num_threads` (1 for the tiny/frequent models, 2 for per-turn WavLM),
  `inter_op_num_threads=1`, sequential execution and `allow_spinning=0`. Per-model
  policy lives in `DEFAULTS`; services override it with `ort_threads` /
  `ort_spinning`. OpenCV's own parallel loops are capped the same way
  (`opencv_threads`, default 2).
- **The debug HUD** redrew at 30 fps from a 15 fps camera — copying, drawing,
  letterboxing and blitting twice per delivered frame. It now redraws only when
  the frame id changes and defaults to `target_fps: 15`.
- **Vision yields while the machine is needed elsewhere**
  (`busy_detect_hz: 5`): detection drops to 5 Hz while a turn is being transcribed
  and while a reply is playing, then returns to 15 Hz. The triggers are
  `TurnCompleted` and `SpeechSynthesized` — deliberately **not** `SpeechStarted`,
  because that is the *listening* phase where vision feeds engagement tracking
  (throttling there suppressed the signal the voice gate needs).
- **The detect loop paces against a schedule** instead of sleeping the interval
  after each inference; previously inference time silently added to the period
  and a "15 Hz" loop behaved like 4 Hz.
- **No polling**: the STT worker waits on an `asyncio.Queue` instead of waking
  50×/second to check for work.

Honest caveats: `nvidia-smi` utilisation is **system-wide**, not per-process, so
treat GPU numbers as indicative (CPU numbers are exact — they come from the
process tree's own CPU time). Lower overall load also lets the GPU drop to idle
clocks, so per-inference latency can look worse while total energy is lower
(observed p50 16 ms under load vs 31 ms when the machine is otherwise idle).

## Phase mapping (what lands where)

Phase 1 adds real perception (YOLO26n + tracker + face pipeline) behind
`perception/`; Phase 2 adds `audio/` (VAD + Smart Turn + faster-whisper + gate +
voice-prints + TTS) behind the same bus contracts; Phase 3 adds `cognition/`
(dialogue FSM, `AttentionPolicy`, LLM router, guardrails) and replaces
`audio/responder.py`; Phase 4 `memory/` (profiles, memory cards, recall);
Phase 5 hardens supervision; Phase 6 validation. Interfaces stay as declared
here — later phases extend, not reshape.

### Swappability ledger (Phase 2)

| Slot | Default | Swapped because | Swap cost |
|---|---|---|---|
| VAD | Silero v5 via `silero-vad` package | package hard-imports torchaudio (no build for pinned torch) | ONNX graph called directly, same model file |
| Smart Turn features | torchaudio mel | torchaudio unavailable | numpy whisper log-mel, validated on fixtures |
| Voice-prints | SpeechBrain ECAPA | hard-imports torchaudio | raw-waveform WavLM ONNX, threshold recalibrated |
| Offline TTS | Kokoro (planned) | no ONNX/offline wheel verified here | SAPI5 via pyttsx3 (no download, works offline) |

Each row is one config key or one provider class — reverting any of them is a
YAML edit, not a refactor. The rationale lives in `requirements.txt`.

## Scope decision (revised after live testing)

Language is **preserved, not translated**: `language: auto` on STT plus
`task="transcribe"` means French in → French text out, Arabic in → Arabic text
out, and the detected language selects the reply voice (`tts.voices`) and the
reply template (`responder.templates`). Dialogue *design* stays English-first and
Darija is not specially tuned (Moroccan Arabic is handled as standard Arabic).
Adding a language remains a YAML edit. Original decision (English-only,
2026-09-19) was revised 2026-10-05 — see README and ROADMAP §5.2.
