# ARIA-POC Codebase Review

Date: 2026-09-22
Scope: lightweight inspection of the application shell, core runtime, audio path, configuration, documentation, and test entry point. No application code was changed.

## Overall assessment

This is a well-structured prototype with a clear service/event architecture, bounded queues, explicit lifecycle handling, useful fixture-oriented tests, and unusually strong performance documentation. It is not yet ready to be treated as a production security/reception system. The largest gaps are incomplete product behavior, privacy controls, reproducible deployment, and failure handling that can leave the process running without the capabilities users expect.

## Findings

### High priority

#### 1. The dialogue layer is still a stub

Evidence: [aria/audio/responder.py](../aria/audio/responder.py) implements `StubResponderService` and returns an acknowledgement such as `I heard you say: {text}`. The roadmap also places the dialogue FSM, LLM routing, guardrails, and memory in later phases.

Why it matters: the perception and audio loop may be functional, but the system is not yet a receptionist/security assistant. It cannot reliably answer requests, maintain conversational state, resolve ambiguity, or apply policy to generated responses.

Recommended solution:

- Replace the stub with a dialogue state machine that owns session state and turn transitions.
- Add an explicit response policy: allowed intents, refusal behavior, confidence thresholds, and escalation behavior.
- Put any LLM behind a narrow router with timeouts, cancellation, prompt/version tracking, PII minimization, and a local fallback.
- Add golden conversation scenarios for normal requests, bystanders, interruptions, low-confidence speech, and network loss.

#### 2. Speech content is written to logs without adequate privacy controls

Evidence: [aria/audio/stt.py](../aria/audio/stt.py) logs full transcript text, [aria/audio/responder.py](../aria/audio/responder.py) logs the generated reply, and [aria/core/telemetry/logging.py](../aria/core/telemetry/logging.py) only masks long digit runs.

Why it matters: ordinary names, addresses, access details, health information, and other sensitive speech can remain in `logs/aria.log` and event timelines. The current documentation implies stronger protection than the implementation provides.

Recommended solution:

- Default to logging metadata only: utterance ID, language, duration, confidence, decision, and latency.
- Make raw transcript/reply logging opt-in, short-lived, and visibly marked as sensitive.
- Add a policy-layer redactor for emails, phone numbers, addresses, names/identifiers, and configured secrets; test it with adversarial formats.
- Define retention, rotation, deletion, and access rules for logs, audio fixtures, face galleries, and voice galleries.
- Encrypt sensitive data at rest if this leaves a developer laptop.

#### 3. Startup can succeed while critical services are missing or broken

Evidence: [aria/core/registry.py](../aria/core/registry.py) catches build and startup exceptions, records them, logs a warning, and lets the application continue. There is no required-service validation in [main.py](../main.py).

Why it matters: ARIA can appear to be running while camera, microphone, STT, gate, or TTS is unavailable. For a safety/security-facing system, silent degradation can be worse than a clear refusal to operate.

Recommended solution:

- Declare services as `required` or `optional` in configuration.
- Fail startup, or enter an explicit degraded mode, when required capabilities are unavailable.
- Publish a readiness/health state that the UI and operators can see.
- Refuse to claim active monitoring or conversation when the relevant pipeline is not ready.
- Add tests for missing class, import failure, model-load failure, and restart-budget exhaustion.

### Medium priority

#### 4. Dependencies are not reproducible enough for deployment

Evidence: [requirements.txt](../requirements.txt) uses open-ended minimum versions, while the CUDA-specific `torch` and `torchvision` installation is documented separately and is not represented in a lock or constraints file. STT fallback model names can also trigger mutable remote downloads.

Why it matters: a fresh setup can resolve incompatible versions, behave differently from the development machine, or fail on Windows/Jetson. Model and runtime compatibility is central to this project.

Recommended solution:

- Maintain separate locked/constraint files for Windows GPU, CPU, and Jetson targets.
- Pin Python, CUDA/JetPack assumptions, PyTorch, ONNX Runtime, faster-whisper, and model revisions.
- Record model checksums and download them through a verified provisioning step.
- Add a clean-environment smoke test to CI and document the selected interpreter.
- Prefer explicit local model paths in production; make network downloads opt-in.

#### 5. The test baseline is not currently verifiable from the active environment

Evidence: `python run_tests.py` failed immediately with `ModuleNotFoundError: No module named 'pytest'`. The documented test counts also vary across [README.md](../README.md) and [ROADMAP.md](../ROADMAP.md) (133, 137, and 137/137 claims in different sections).

Why it matters: the project cannot currently substantiate its green-suite claims from this workspace, and stale test-count claims make it harder to know what is actually covered.

Recommended solution:

- Install dependencies into the selected `.venv` and run the suite there, or make the runner select/validate the project interpreter.
- Add CI that runs tests from a clean environment and publishes the count.
- Stop hard-coding test counts in prose; report the command and result instead.
- Add a small `selfcheck` prerequisite check that reports missing packages clearly.

#### 6. Configuration validation is shallow at the application boundary

Evidence: [aria/core/service.py](../aria/core/service.py) validates only service-level keys and basic Python types; [aria/core/config.py](../aria/core/config.py) loads YAML and merges dictionaries without schema validation for global sections or semantic ranges.

Why it matters: invalid values such as negative durations, impossible queue sizes, unsupported provider names, or malformed nested mappings can survive until runtime.

Recommended solution:

- Validate the complete profile before building services.
- Enforce ranges and enumerations, not only types.
- Reject unknown global keys in strict/production mode.
- Report all configuration errors together with file and key paths.
- Add config tests for inheritance, invalid types, invalid ranges, and unknown keys.

#### 7. The system depends on network services without an explicit operational contract

Evidence: [aria/audio/tts.py](../aria/audio/tts.py) uses edge TTS by default, while model fallback paths may download from remote registries. The architecture describes offline fallbacks, but there is no broader network timeout, retry, privacy, or offline-readiness policy visible at the app boundary.

Why it matters: network loss can delay or prevent replies, and speech may be sent to an external provider without an explicit consent/configuration boundary.

Recommended solution:

- Make cloud TTS opt-in for privacy-sensitive deployments.
- Define strict connect/read/total timeouts and bounded retries.
- Expose network state and provider choice in health telemetry.
- Test offline startup and mid-conversation network loss.
- Provide a verified, language-capable local TTS option before calling offline operation complete.

### Lower priority

#### 8. Deployment hardening and quality gates are missing

The repository has no visible packaging metadata, lock file, CI workflow, lint/type-check configuration, or automated security scan configuration. The code can be syntax-checked, but syntax validity is much weaker than repeatable build and runtime validation.

Recommended solution:

- Add CI jobs for compile, tests, lint, type checking, dependency audit, and a hermetic smoke run.
- Add a supported-install document for each target platform.
- Package the application or provide a pinned launch script that validates paths and environment variables.
- Add soak tests covering memory, queue growth, service restarts, device reconnects, and log rotation.

#### 9. Identity and anti-spoofing assumptions remain too weak for security use

The roadmap notes that voice-print calibration was performed on synthetic voices and that passive liveness/anti-spoofing is deferred. Face and voice matches should therefore be treated as convenience signals, not authentication factors.

Recommended solution:

- Label identity confidence and uncertainty in the policy layer.
- Require active liveness or a second factor for protected actions.
- Calibrate thresholds on representative real voices/faces and environmental conditions.
- Add replay, printed-photo, screen, and voice-recording attack tests.

## Strengths worth preserving

- The event bus has explicit backpressure policies and failure isolation.
- Services have a consistent lifecycle and watchdog integration.
- Frame/audio stores avoid pushing large payloads through the event bus.
- Fixture-based tests and pure logic components make several difficult paths testable.
- Performance tuning is measured and configurable rather than assumed.
- The documentation clearly records known trade-offs such as CPU fallback cost and lack of Darija-specific tuning.

## Suggested order of work

1. Establish a reproducible environment and make the test suite runnable in CI.
2. Add required-service/readiness enforcement so degraded operation is explicit.
3. Remove raw speech from default logs and define retention/privacy policy.
4. Replace the responder stub with a policy-backed dialogue layer and scenario tests.
5. Lock dependencies/models and test offline/network-failure behavior.
6. Add soak, security, anti-spoof, and deployment validation before hardware rollout.

## Checks performed

- Read the top-level documentation, roadmap, requirements, app shell, registry, service base, event bus, logging, STT, TTS/responder, and watchdog paths.
- `python -m compileall -q aria main.py tools tests`: completed without syntax errors.
- `python run_tests.py`: could not start because `pytest` is not installed in the active Python environment.
- Git status check: unavailable because `git` is not installed or not on the active shell PATH.
