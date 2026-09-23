"""Audio benchmark — Phase 2 exit criteria: EOU latency + false-response rate.

Runs the real audio pipeline (VAD → three-tier turn → faster-whisper → gate)
over a manifest of replay fixtures, then reports:

- **EOU latency**: classify time from the SpeechEnded candidate to the
  TurnCompleted decision, plus the end-of-speech→transcript total;
- **false-response rate**: fixtures marked ``no_response`` that nevertheless
  produced an accepted utterance (the FUNC-14 "it answered noise" metric);
- **miss rate**: fixtures marked ``accept`` that were not accepted;
- **RTF**: STT real-time factor.

    python tools/bench_audio.py --manifest fixtures/audio/manifest.yaml --assert
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def audio_seconds(path: pathlib.Path) -> float:
    sys.path.insert(0, str(ROOT))
    from aria.audio.store import read_wav

    samples, rate = read_wav(path)
    return samples.size / float(rate)


def profile_for(fixture: pathlib.Path, margin_s: float = 8.0) -> dict:
    """Headless profile: replay fixture through the audio stack, no playback.

    Engagement gating is disabled here (no camera in the bench) — it is covered
    by unit tests; this bench measures the noise/overlap class of failures.
    """
    return {
        "extends": str(ROOT / "configs" / "laptop.yaml"),
        "logging": {"level": "INFO", "dir": "logs_bench", "console": "quiet"},
        # Stop as soon as the fixture has been processed instead of after a
        # guessed sleep: STT readiness alone takes ~12-15 s, so a fixed
        # "fixture + 8 s" budget shut the app down before the turn completed.
        "runtime": {"exit_on_source_end": True},
        "services": {
            "camera": {"enabled": False},
            "vision": {"enabled": False},
            "face": {"enabled": False},
            "scene": {"enabled": False},
            "tamper": {"enabled": False},
            "display": {"enabled": False},
            "mic": {
                "enabled": True,
                # Start the replay only once the transcriber is warm, otherwise
                # the fixture's single utterance is already stale by then.
                "config": {"source": "file", "path": str(fixture), "loop": False,
                           "wait_for_ready": ["stt"], "readiness_timeout_s": 60.0},
            },
            "vad": {"enabled": True},
            "turn": {"enabled": True},
            "stt": {"enabled": True},
            "voiceprint": {"enabled": False},
            "gate": {"enabled": True, "config": {"require_engagement": False, "voiceprint_wait_s": 0.0}},
            "responder": {"enabled": True},
            "tts": {"enabled": True, "config": {"provider": "fake", "playback": False}},
        },
    }


TIMELINE = ROOT / "logs_bench" / "timeline_full.jsonl"


def timeline_offset() -> int:
    if not TIMELINE.exists():
        return 0
    return len(TIMELINE.read_text(encoding="utf-8", errors="replace").splitlines())


def read_timeline(offset: int) -> list[dict]:
    """Events recorded since ``offset`` (the timeline accumulates across runs)."""
    if not TIMELINE.exists():
        return []
    lines = TIMELINE.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < offset:          # file was truncated by a fresh run
        offset = 0
    events = []
    for line in lines[offset:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def run_fixture(fixture: pathlib.Path) -> dict:
    seconds = audio_seconds(fixture)
    offset = timeline_offset()
    with tempfile.TemporaryDirectory(prefix="aria-bench-audio-") as tmp:
        cfg = pathlib.Path(tmp) / "bench_audio.yaml"
        cfg.write_text(yaml.safe_dump(profile_for(fixture), sort_keys=False), encoding="utf-8")
        proc = subprocess.run(
            # The app exits by itself once the fixture is drained; this duration is
            # only a safety cap for a run that never finishes.
            [str(PY), str(ROOT / "main.py"), "--config", str(cfg), "--duration", str(seconds + 60.0)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
        )
    if proc.returncode != 0:
        return {"error": proc.stderr[-800:], "seconds": seconds}
    summary = json.loads(proc.stdout.strip())
    counters = summary["metrics"]["counters"]
    hist = summary["metrics"]["histograms"]
    accepted = int(counters.get("gate.accepted", 0))
    rejected = int(counters.get("gate.rejected", 0))
    turns = int(counters.get("eou.turns_completed", 0))

    events = read_timeline(offset)
    heard = [e.get("payload", {}) for e in events if e.get("name") == "UtteranceHeard"]
    synth = [e.get("payload", {}) for e in events if e.get("name") == "SpeechSynthesized"]
    return {
        "seconds": seconds,
        "accepted": accepted,
        "rejected": rejected,
        "turns": turns,
        "decision": "accept" if accepted else ("no_response" if turns >= 0 else "unknown"),
        "rtf": (hist.get("stt.rtf") or {}).get("p50"),
        "eou_classify_ms": (hist.get("eou.classify_ms") or {}).get("p50"),
        "stt_latency_s": (hist.get("stt.latency_s") or {}).get("p50"),
        "stt_queue_wait_s": (hist.get("stt.queue_wait_s") or {}).get("p50"),
        "utterances": int(counters.get("stt.utterances", 0)),
        "discarded": int(counters.get("eou.discarded", 0)),
        "dropped_stale": int(counters.get("stt.dropped_stale", 0)),
        # Turns that arrived while Whisper was busy (single-flight drop, no queue).
        "dropped_busy": int(counters.get("stt.dropped_busy", 0)),
        "text": heard[-1].get("text", "") if heard else "",
        "language": heard[-1].get("language") if heard else None,
        "voice": synth[-1].get("voice") if synth else None,
        # full_text is the whole reply; .text is only one clause of it
        "spoken": (synth[-1].get("full_text") or synth[-1].get("text")) if synth else "",
    }


def check_language(entry: dict, result: dict) -> str:
    """Return a failure reason if the transcript/reply does not match expectations."""
    if entry.get("expect") != "accept":
        return ""
    text = result.get("text") or ""
    want_lang = entry.get("expect_language")
    if want_lang and result.get("language") != want_lang:
        return f"language {result.get('language')!r} != {want_lang!r} (text={text[:60]!r})"
    script = entry.get("expect_script")
    if script == "arabic" and not any("\u0600" <= ch <= "\u06ff" for ch in text):
        return f"expected Arabic script, got {text[:60]!r} (translated?)"
    if script == "latin" and not any(ch.isalpha() and ord(ch) < 0x250 for ch in text):
        return f"expected Latin script, got {text[:60]!r}"
    tokens = entry.get("expect_any_of") or []
    if tokens and not any(t.lower() in text.lower() for t in tokens):
        return f"none of {tokens} in {text[:60]!r} (translated?)"
    voice = entry.get("expect_voice_contains")
    if voice and voice not in str(result.get("voice") or ""):
        return f"reply voice {result.get('voice')!r} does not match {voice!r}"
    spoken_tokens = entry.get("expect_spoken_any_of") or []
    spoken = result.get("spoken") or ""
    if spoken_tokens and not any(t.lower() in spoken.lower() for t in spoken_tokens):
        return f"reply not localised: {spoken[:60]!r} (wanted one of {spoken_tokens})"
    return ""


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Bench the audio frontend")
    ap.add_argument("--manifest", default="fixtures/audio/manifest.yaml")
    ap.add_argument("--assert", dest="do_assert", action="store_true")
    ap.add_argument("--max-false-response-rate", type=float, default=0.0)
    ap.add_argument("--max-eou-p95-ms", type=float, default=800.0)
    ap.add_argument("--max-rtf", type=float, default=1.0)
    args = ap.parse_args()

    manifest_path = ROOT / args.manifest if not pathlib.Path(args.manifest).is_absolute() else pathlib.Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    entries = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or []

    print("== bench_audio ==")
    rows, expect_positive, false_responses, missed = [], 0, 0, 0
    language_failures = []
    for entry in entries:
        fixture = ROOT / entry["path"] if not pathlib.Path(entry["path"]).is_absolute() else pathlib.Path(entry["path"])
        if not fixture.exists():
            print(f"  [skip] missing fixture {fixture}")
            continue
        result = run_fixture(fixture)
        if "error" in result:
            print(f"  [fail] {entry['name']}: {result['error']}")
            return 1
        expect = entry.get("expect", "accept")
        correct = (result["decision"] == expect)
        if expect == "accept":
            expect_positive += 1
            missed += 0 if correct else 1
        else:
            false_responses += 0 if correct else 1
        rows.append((entry["name"], expect, result, correct))
        language_problem = check_language(entry, result)
        if language_problem:
            language_failures.append(f"{entry['name']}: {language_problem}")
        print(f"  {entry['name']:<16} expect={expect:<12} got={result['decision']:<12} "
              f"turns={result['turns']:<2} rejected={result['rejected']:<2} "
              f"eou_p50={result['eou_classify_ms']} ms rtf={result['rtf']} "
              f"lang={result['language']} voice={result['voice']} | {str(result['text'])[:40]}")
        if language_problem:
            print(f"                   ↳ LANGUAGE FAIL: {language_problem}")

    if not rows:
        print("no fixtures ran")
        return 1

    negatives = len(rows) - expect_positive
    frr = (false_responses / negatives) if negatives else 0.0
    miss = (missed / expect_positive) if expect_positive else 0.0
    eou_samples = [r["eou_classify_ms"] for _, _, r, _ in rows if r["eou_classify_ms"] is not None]
    eou_p50 = sorted(eou_samples)[len(eou_samples) // 2] if eou_samples else None
    rtf_samples = [r["rtf"] for _, _, r, _ in rows if r["rtf"] is not None]
    rtf_max = max(rtf_samples) if rtf_samples else None

    print(f"\nfixtures           : {len(rows)} ({expect_positive} expect-accept, {negatives} expect-no-response)")
    print(f"false-response rate: {frr:.2f}  ({false_responses}/{negatives})")
    print(f"miss rate          : {miss:.2f}  ({missed}/{expect_positive})")
    if eou_p50 is not None:
        print(f"EOU classify p50   : {eou_p50:.0f} ms")
    if rtf_max is not None:
        print(f"STT RTF max        : {rtf_max:.2f}")
    waits = [r["stt_queue_wait_s"] for _, _, r, _ in rows if r.get("stt_queue_wait_s") is not None]
    if waits:
        print(f"STT pickup wait p50: {sorted(waits)[len(waits)//2] * 1000:.0f} ms")
    busy = sum(r.get("dropped_busy", 0) for _, _, r, _ in rows)
    if busy:
        print(f"STT dropped (busy) : {busy} turn(s) arrived mid-transcript (dropped, not queued)")
    if language_failures:
        print("language failures  :")
        for problem in language_failures:
            print(f"  - {problem}")

    if args.do_assert:
        failures = []
        if frr > args.max_false_response_rate:
            failures.append(f"false-response rate {frr:.2f} > {args.max_false_response_rate:.2f}")
        if eou_p50 is not None and eou_p50 > args.max_eou_p95_ms:
            failures.append(f"EOU p50 {eou_p50:.0f} ms > {args.max_eou_p95_ms:.0f} ms")
        if rtf_max is not None and rtf_max > args.max_rtf:
            failures.append(f"STT RTF {rtf_max:.2f} > {args.max_rtf:.2f}")
        failures.extend(language_failures)
        if failures:
            print("FAIL: " + "; ".join(failures))
            return 1
        print("OK: within audio budget (and language preserved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())