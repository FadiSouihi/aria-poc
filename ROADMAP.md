# ARIA-POC — Roadmap (condensed, v1.2)

Authoritative plan for the prototype. The full formatted report is
`docs/ARIA_POC_Roadmap.pdf` (v1.2). This file is the working copy that lives
with the code.

## 1. Goal

Replace the current robot's fragile dual code paths (single-human vs group
detection, evaluated once per session) with **one dynamic perception
pipeline** where 0/1/N people are the same code path, and make **sessions,
identity, and memory first-class, continuously verified concepts** — built
modularly so every capability is a swappable service behind configuration.

## 2. Scope decision (project, 2026-09-19; **revised 2026-10-05**)

**Original:** English-only interaction; multi-language switching (Arabic, French,
Derja) out of scope.

**Revised after live testing.** Locking the pipeline to English was the wrong
call, because the model was already hearing other languages correctly — the
forced `language="en"` made Whisper *translate* the transcript on the way out,
which is what the user objected to ("it knows what I'm saying but when relaying
it, it translates it"). The pipeline now:

- transcribes in the spoken language (`language: auto`, `task="transcribe"`) —
  text out equals words in, never a translation;
- picks a voice for that language (`tts.voices`: `en`, `fr`, `ar`, `es`, each a
  one-line config addition) and a localised reply wrapper
  (`responder.templates`), so ARIA answers *in* the language she was addressed in.

Verified per language by `tools/bench_audio.py` (transcript language + script +
reply voice assertions) and by `tests/test_audio_integration.py`. What stays out
of scope: dialogue *design* is English-first, and **Darija is not specially
tuned** — Moroccan Arabic is handled as standard Arabic (`ar-MA-*` voices are
available). Adding a language remains config, not code.

## 3. Targets

- **POC (this machine):** ASUS Vivobook 16 — Core 5 210H (8C/12T),
  RTX 4050 Laptop 6 GB VRAM, 16 GB RAM, Windows 11.
- **Deployment:** Jetson Orin Nano 8 GB (JetPack 6.x, unified memory).

## 4. Model budget (POC)

| Component | Selection | Budget |
|---|---|---|
| Person detection | YOLO26n @ 640 (person class) | ~0.5 GB VRAM |
| Face detection | SCRFD-500M (alt: YuNet) | ~0.1 GB |
| Face embedding | ArcFace r50 (InsightFace buffalo_l); **Phase 1 default: SFace 128-d ONNX (opencv_zoo)** — the InsightFace HF zoo is gated, ArcFace stays a config swap | ~0.05–0.3 GB |
| VAD + turn detection | Silero VAD v5 + Smart Turn v3.2 | ~0.1 GB |
| STT | faster-whisper `large-v3-turbo` int8_float16 | ~1.2 GB (or CPU) |
| LLM | cloud (default) · Qwen3-4B-Instruct-2507 Q4 (fallback) | ~2.5 GB on demand |
| TTS | edge-tts · Kokoro-82M (offline EN) · Piper1-GPL (capability) | 0 VRAM |

Steady-state ≈ 2.2 GB, peak (local LLM) ≈ 4.4 GB — inside 6 GB with headroom;
the governor demotes STT to CPU under pressure.

## 4b. Resource budget (measured 2026-10-06)

The prototype was consuming **9.6 of 12 logical cores (80%)** on this laptop. It
was not the models — it was spinning (ONNX Runtime's default thread pool
spin-waits; a tiny VAD graph called 31×/second was waking 8 threads forever),
plus a 30 fps HUD on a 15 fps source and a flight recorder that opened, wrote and
closed its file once per event. After `aria/core/onnx.py` + the vision yield
policy + buffered/sampled telemetry + schedule-paced detect loop:

| Profile | before | after |
|---|---|---|
| `laptop.yaml` (vision + audio + HUD), idle | 9.64 cores (80%) | **0.54 cores (4%)** |
| `laptop.yaml` while conversing (`bench_talk.yaml`) | — | **0.70 cores (6%)** |
| `audio_only.yaml` (conversation only), idle | 6.93 cores (58%) | **0.14 cores (1%)** |

Measured with `tools/measure_load.py` (steady state, warm-up excluded). This is
what makes the Jetson target plausible: an Orin Nano has 6 weaker cores, so the
old numbers had no room for Phase 3 (dialogue/LLM) or Phase 4 (memory) at all.

Open item from the same measurement: the CPU *fallback* for STT runs
large-v3-turbo int8 at **RTF ≈ 9** (a 4.7 s clip takes ~45 s), which is not
conversational. A smaller CT2 model for the CPU path is the fix (config knob,
model download pending).

## 5. Key selections (swappable via config)

- **Detection/tracking:** YOLO26n (Jan 2026, NMS-free end-to-end) + ByteTrack;
  BoT-SORT / OC-SORT / Deep OC-SORT / TrackTrack are YAML alternates.
- **Identity:** SCRFD + ArcFace 512-d with multi-frame voting (≥N consistent
  matches + margin) and per-track embedding galleries. Phase 1 ships the
  voting + gallery logic with SFace 128-d (YuNet landmarks); swapping in
  ArcFace later is a config change behind the same interface.
- **End-of-turn (three tiers):** Silero VAD v5 → Smart Turn v3.2 (8 MB int8 CPU,
  23 languages, ~10–100 ms) → configurable threshold fallback. Silero runs on
  its ONNX graph directly (the pip package hard-imports torchaudio, which has no
  build for the pinned torch — see `requirements.txt`); the log-mel Smart Turn
  needs is computed in numpy.
- **STT:** faster-whisper `large-v3-turbo` int8_float16 (English-first).
  Measured RTF 0.21–0.27 on 4 s utterances (RTX 4050).
- **Voice-prints:** WavLM x-vectors (512-d, raw-waveform ONNX) — chosen over
  SpeechBrain ECAPA (hard-imports torchaudio) and Resemblyzer (needs `webrtcvad`
  → MSVC build tools). Calibrated: same speaker 0.92 vs other 0.45–0.49.
- **TTS:** edge-tts (laptop default) / SAPI5 offline fallback (shipped) /
  Kokoro-82M (offline EN, still the preferred upgrade once a wheel is verified) /
  Piper1-GPL (multilingual capability retained, not wired in).
- **LLM:** cloud completion API with Ollama Qwen3-4B fallback router.
- **Memory:** SQLite (WAL) + optional sqlite-vec later.

## 6. Architecture (built in Phase 0)

EventBus (typed, bounded, tapped) ← Service base (lifecycle, heartbeat,
schema, crash containment) ← Registry (config-driven, restart in place) ←
Watchdog (staleness/crash → bounded restarts) ← Governor (hysteresis pacing)
← Telemetry (structured logs, JSONL timeline flight-recorder, metrics).
Details: `ARCHITECTURE.md`.

## 7. Unified detection (Phase 1–3 design)

1. Sense once: detect → track → N persistent `Track`s, re-evaluated every
   cycle (no modes).
2. Per-track FSM: `PRESENCE → NEAR → ENGAGED → DEPARTED`.
3. Identity binding with **ID-lock** + periodic re-verification + suspicion
   triggers; context never transfers on an A→B swap.
4. Multimodal departure: face-loss AND no voice AND no interaction.
5. Group awareness: SceneManager + AttentionPolicy (addressee selection,
   bystander acknowledgment, A→B handoff).
6. Tamper detection via frame statistics; device hot-reconnect.

## 8. Sessions & memory (Phase 3–4)

Rolling FIFO (default 6) + running summary; MemoryCards extracted on close;
RecallComposer injects recent/unresolved cards on re-engage ("Did you find
the library last time?"); sensitive-ID masking at the policy layer; wipe on
confirmed departure; stranger profiles auto-expire.

## 9. Observability & testing (Phase 0 — done)

- Structured JSON logs with correlation ids; ID masking; rotating files.
- Event timeline: full tap + ring buffer + anomaly-triggered dumps.
- Metrics: counters/histograms (fps, EOU latency, RTF, TTFT, RSS...).
- Test-first: contract suites per interface, hermetic fakes, fixture
  replay (camera + audio fixtures), CSV-derived scenario tests,
  benchmarks-as-tests. **145/145 tests green** (Phase 0–2).

## 10. Phases

| Phase | Scope | Exit criteria |
|---|---|---|
| 0 — Skeleton ✅ | primitives + observability + test scaffolding | services start/stop/restart from config; watchdog restarts in place; backpressure tested; structured logs + correlation ids; contract suite green |
| 1 — Perception ✅ | real camera sources, YOLO26n + tracker, SceneManager, face pipeline, tamper, fixture recorder | 0/1/N via one code path; 3 s occlusion survives; identity stable; `bench_vision` in budget; scenario tests on fixtures |
| 2 — Audio ✅ | VAD + Smart Turn, faster-whisper, voice-prints, TTS abstraction, barge-in v1 | FUNC-14 8/10/11/12 fixed in replay tests; measured EOU latency + false-response rate → **p50 47–62 ms; false-response 0.00; miss 0.00; RTF ≤0.31; EN/FR/AR transcribed in-language with language-matched voices; speech_end→transcript 880 ms; first audio 0.0 ms; clause gap 0.9 ms; 145/145 tests; idle CPU 0.54 cores** |
| 3 — Cognition | dialogue FSM, AttentionPolicy, LLM router, guardrails + masking | full conversation; NFR-06 probes mitigated; network cut → local model |
| 4 — Memory | profiles, MemoryCards, recall, retention/wipes | FUNC-10 recall passes (same/next day); masking regressions pass |
| 5 — Robustness | watchdog self-heal, hot reconnect, degradation modes, soak tests | camera crash heals in place; RSS returns to baseline; replug detected live |
| 6 — Validation | CSV test matrix, dashboards, demo mode | all mapped rows fixed or explicitly deferred; end-to-end demo |
| 7 — Jetson | TensorRT INT8, deployment guide, mic-array, diarization, anti-spoof, LiDAR | on-device validation when hardware access is granted |

## 11. Defect → feature → phase (from the manual test CSV)

Hard pause timeout → three-tier turn detection (P2 ✅ shipped: VAD silence →
Smart Turn v3.2 semantic model → hard timeouts) · noise/overheard speech
→ VAD + engagement gating + voice-prints (P2 ✅ shipped: false-response rate
0.00 on the noise/click fixtures; `no_one_engaged` rejection in the live run)
· no recall → MemoryCards (P4)
· language switching → **shipped as language *preservation*** (P2 ✅: speech is
transcribed in the spoken language and answered with a voice for that language;
nothing is translated — see §2) · reply backlog / "buffered voice" → bounded turn
queue + `max_stale_s` + clause pipelining (P2 ✅: the 4.9 s tail is gone — max
`speech_end→transcript` 4883 → 2362 ms at p50 ≈ 1.27 s, queue wait p50
64–96 ms, and superseded turns are dropped instead of answered out of order;
see §12.6)
· overlapping speakers →
one-at-a-time prompt now, diarization research later (P2/7) · face
false-positive → multi-frame voting (P1) · spoof accepted → passive liveness
research (P7) · gaze-down = "left" → body-track presence + multimodal
departure (P1) · no tamper awareness → frame-statistics tamper (P1) · no
group capability → SceneManager + AttentionPolicy (P1/3) · no ID-lock →
identity binding + re-verification (P1/3) · camera crash needs restart →
watchdog + DeviceMonitor (P5) · "technical difficulties" on network loss →
local LLM router (P3) · memory residue → governor + teardown + soak test
(P0/5) · prompt injection → no-secrets design + screening + canary (P3) ·
masking edge cases → policy-layer masking with regression tests (P3/4).

## 12. Assumptions & risks

Internet available on the laptop (cloud defaults, offline by config);
webcam + headset mic stand in for the robot's CSI camera + USB mic; 6 GB
VRAM is the binding constraint (governor built-in); newer-model churn
(YOLO26/Smart Turn/Qwen3) is mitigated by version pinning + contract tests +
golden-file diffs.

### Phase 2 findings (measured, not assumed)

1. **PyTorch wheel triangle.** `torchaudio` has no 2.14 build; installing it
   downgrades torch and breaks `torchvision.ops.nms` (verified failure:
   `operator torchvision::nms does not exist`). The stack is therefore pinned
   to torch 2.14.0 + torchvision 0.29.0 with **no torchaudio**, and every audio
   component that would have needed it was re-implemented (ONNX VAD, numpy
   log-mel, WavLM voice-prints). Re-check when a matching torchaudio ships.
2. **Silero v5 needs 64 samples of context** prepended to each 512-sample call.
   Feeding bare frames returns ~0.003 probability on real speech — silent,
   plausible-looking failure; caught by a fixture test that asserts the VAD
   actually fires on speech.
3. **Echo path.** With a speaker and mic, ARIA hears itself. Half-duplex ducking
   (`vad.duck_while_speaking`, default on) prevents self-triggered barge-in;
   turn it off for headset use where true barge-in is wanted. Real full-duplex
   needs AEC (Phase 5/7).
4. **Voice-print threshold** was calibrated on synthetic TTS voices (an easy
   case). Re-run `tools/calibrate_voice.py` after enrolling real voices and
   adjust `voiceprint.threshold`; the gate degrades to engagement-only when the
   score is below threshold rather than guessing.
5. **EOU pending window.** `pending_timeout_s` (1.0 s in the laptop profile) is the
   knob that trades "waits for you to continue" against "replies too eagerly"; it
   is a config value and a bench manifest entry, so the trade-off is measured per
   profile.
6. **The "buffered voice" was a queue, not the microphone.** Live measurement
   (`tools/report_latency.py`) showed `speech_end→transcript` p50 1144 ms but
   **max 4883 ms** with transcripts arriving after the *next* utterance — STT
   transcribed inline on the bus callback, so a second utterance queued behind the
   first and the reply order looked scrambled. Fixed by a bounded turn queue with
   a worker, `max_stale_s` (drop superseded turns, checked at enqueue *and*
   dequeue) and a warm-up pass. Result: max `speech_end→transcript`
   **4883 → 2362 ms** (p50 ≈ 1.27 s), queue wait p50 **64–96 ms**,
   `transcript→decision` p50 **83 ms**, and no reply is ever spoken for an older
   utterance than the one before it. The remaining p50 is Whisper decode +
   320 ms of VAD silence + EOU, i.e. the honest floor for this model on this
   laptop; a smaller multilingual model (`stt.model`) or a local TTS is the next
   lever if it must be lower.
7. **A slow model load silently ate turns.** `SttService` subscribed to
   `TurnCompleted` *after* loading Whisper (~8 s), so anything spoken in that
   window was never seen by STT at all (found because a fixture bench reported
   "no response" for a file that clearly contained a question). Fixed: subscribe
   first, load in a background task, `core/readiness` signal, STT listed first in
   the profile so its load overlaps vision startup, and `mic.wait_for_ready`
   for replay sources.
8. **Watchdog had zero margin.** `stale_after` (5 s) equalled the camera
   heartbeat interval (5 s), so a heartbeat delayed by Whisper's GPU work caused
   `Service unhealthy; restarting {service: camera}` in a live log. Fixed: stale
   only after `max(stale_after, heartbeat_interval × stale_safety_factor)`
   (default 3×) — the same 3× convention `Service.health()` already used — plus
   `camera.heartbeat_interval: 1.0`. Verified: 0 restarts across live runs.
9. **Clause-level pipelining beats a shorter reply.** Splitting the reply and
   synthesizing clause *n+1* while *n* plays cuts time-to-first-audio to one
   clause; with the LRU cache, repeated phrases cost 0 ms (observed 2 cache hits
   in a 25 s live run). Each clause publishes `SpeechSynthesized` just before it
   plays, which is what extends the VAD duck window and what barge-in interrupts.
10. **The TTS provider ignored the resolved voice.** `TtsService` resolved
    `fr` → `fr-FR-DeniseNeural` and reported it on `SpeechSynthesized`, but
    `EdgeTtsProvider.synth()` used the voice from its *constructor*, so the audio
    was always `en-US-AriaNeural`. ARIA "knew" the language and still sounded
    English — the reported voice was a lie. Fixed by passing the voice into
    `synth(text, language, voice)`; regression test asserts two voices produce
    different audio for the same text (three distinct outputs verified live).
11. **The offline fallback was a one-way door.** On any edge-tts error the service
    did `self._provider, self._fallback = self._fallback, None`, so *every* later
    reply used SAPI5 — which has no Arabic voice here and produced 0.0 s of audio
    (`audio_s: 0.0` in the log) while still reporting success. Now: fallback for a
    `fallback_cooldown_s` window only, SAPI **raises** when the language has no
    installed voice, empty audio is an error, and `tts.failed.<lang>` counts it.
12. **The pause inside the reply was a reopened audio stream.** `_play_blocking`
    opened and closed an `sd.OutputStream` **per clause** (~100–300 ms on WASAPI),
    which is the gap heard between "I heard you say:" and the words that followed.
    One low-latency stream per reply plus a warm prefix cache
    (`tts.warm_phrases`) gives `first_audio_ms` **0.0** and `clause_gap_ms`
    **0.9 ms** (was seconds of perceived latency).
13. **`min_silence_ms` has a measured floor.** Probing 220/250/280/320 ms against
    the FR and AR fixtures: below 320 ms the French sentence is *truncated*
    ("Bonjour Aéria." only) because the turn ends at the comma and Smart Turn
    scores a greeting as complete — raising `turn.threshold` does not help, since
    the model is confident. 320 ms is the smallest value that keeps every fixture
    intact; the latency win came from the reply path, not from cutting speech
    earlier. (Re-measure per language/microphone: `tools/bench_audio.py`.)
14. **Language detection costs ~300 ms** — half of Whisper's decode time
    (`language: auto` 610 ms vs pinned `fr` 312 ms on the same 4.7 s clip), because
    faster-whisper encodes the audio once to detect and again to decode. Pinning
    `stt.language` is the lever if latency must drop further, but *do not* cache
    the language blindly: forcing `fr` on the Arabic fixture produced fluent
    French nonsense ("Bonjour, où est-ce que la bibliothèque...") — detection
    stays per-utterance by default.
15. **CTranslate2 needs torch's CUDA DLLs on the loader path.** An audio-only run
    (no vision, so torch never imported) failed with
    `Library cublas64_12.dll is not found or cannot be loaded`; the vision stack
    had been masking this. The transcriber now imports torch before constructing a
    CUDA model and falls back to CPU if the DLLs are genuinely missing.
16. **Vision and Whisper share the one GPU.** Same machine, same fixture:
    `speech_end→transcript` p50 **0.88 s** with `configs/audio_only.yaml` versus
    **1.44 s** with the full profile (vision at 15 Hz + face every 3rd frame).
    Nothing is wrong — it is contention for the RTX 4050. Levers, in order of
    preference: use the audio-only profile when testing conversation; lower
    `vision.detect_hz` / raise `face.pace_every`; and (Phase 5) a "yield while
    speaking" policy so the vision cycle pauses during a turn instead of
    competing with it.
