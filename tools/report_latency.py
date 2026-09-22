"""Per-utterance latency report from the JSON log — answers "why does it feel buffered?".

    python tools/report_latency.py                 # last run in logs/aria.log
    python tools/report_latency.py --runs 3        # last 3 runs
    python tools/report_latency.py --log logs/aria.log --json

For every utterance it prints the stage timings and flags the two failure modes
that feel like "buffering" to a user:

* **queued** — you started speaking again while the previous transcript was
  still being produced (STT is serial: the new turn waits its turn);
* **stale** — a transcript arrives long after the speech ended.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

STAGE_MSGS = {
    "Speech started": "speech_start",
    "Speech ended": "speech_end",
    "End-of-turn decision": "eou",
    "Utterance transcribed": "transcript",
    "Utterance accepted": "accepted",
    "Utterance rejected": "rejected",
    "Stub reply": "reply",
    "Speech synthesized": "synth",
    "Barge-in: playback stopped": "barge_in",
}


def load(path: pathlib.Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").strip().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def runs(records: list[dict]) -> list[list[dict]]:
    bounds = [i for i, r in enumerate(records) if r.get("msg") == "Booting ARIA-POC"]
    return [records[b:] for b in bounds[-1:]] if not bounds else [
        records[b: (bounds[i + 1] if i + 1 < len(bounds) else len(records))]
        for i, b in enumerate(bounds)
    ]


def _ts(value: str):
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)          # handles +00:00 offsets
    except ValueError:
        return datetime.strptime(value[:26], "%Y-%m-%dT%H:%M:%S.%f")


def ms_between(a: str, b: str) -> float:
    return (_ts(b) - _ts(a)).total_seconds() * 1000


def collect(run: list[dict]) -> list[dict]:
    events: dict[str, dict] = {}
    order: list[str] = []
    for rec in run:
        stage = STAGE_MSGS.get(rec.get("msg", ""))
        if not stage:
            continue
        fields = rec.get("fields") or {}
        uid = fields.get("utterance_id") or rec.get("context", {}).get("utterance_id")
        if stage == "speech_start":
            uid = uid or f"anon{len(order)}"
            events.setdefault(uid, {"utterance_id": uid})["speech_start"] = rec["ts"]
            order.append(uid)
            continue
        if uid is None:
            continue
        entry = events.setdefault(uid, {"utterance_id": uid})
        if uid not in order:
            order.append(uid)
        entry[stage] = rec["ts"]
        for key in ("duration_s", "p_turn", "classify_ms", "rtf", "text", "reason", "audio_s"):
            if fields.get(key) is not None:
                entry[key] = fields[key]
    return [events[u] for u in order]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Per-utterance latency report")
    ap.add_argument("--log", default="logs/aria.log")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    path = ROOT / args.log if not pathlib.Path(args.log).is_absolute() else pathlib.Path(args.log)
    if not path.exists():
        print(f"no log at {path}", file=sys.stderr)
        return 1
    all_runs = runs(load(path))[-args.runs:]
    stages = ["speech_end→transcript", "transcript→decision", "decision→synth"]
    summary: dict[str, list[float]] = {s: [] for s in stages}

    for index, run in enumerate(all_runs, 1):
        items = collect(run)
        print(f"== run {index}: {len(items)} utterances ==")
        print(f"{'utterance':<10} {'speech_s':>8} {'eou_ms':>7} {'end→text':>9} {'text→dec':>9} "
              f"{'dec→synth':>10} {'rtf':>5}  flags / text")
        previous_transcript_ts = None
        for item in items:
            uid = item["utterance_id"][:9]
            speech_s = item.get("duration_s", "")
            eou = item.get("classify_ms", "")
            end_to_text = ms_between(item["speech_end"], item["transcript"]) if {"speech_end", "transcript"} <= item.keys() else None
            decision_ts = item.get("accepted") or item.get("rejected")
            text_to_dec = ms_between(item["transcript"], decision_ts) if end_to_text is not None and decision_ts else None
            dec_to_synth = ms_between(decision_ts, item["synth"]) if decision_ts and item.get("synth") else None

            flags = []
            if end_to_text is not None and previous_transcript_ts is not None:
                if ms_between(previous_transcript_ts, item["speech_start"]) < 0:
                    flags.append("QUEUED")
            if end_to_text is not None and end_to_text > 1500:
                flags.append("STALE")
            if end_to_text is not None:
                summary["speech_end→transcript"].append(end_to_text)
            if text_to_dec is not None:
                summary["transcript→decision"].append(text_to_dec)
            if dec_to_synth is not None:
                summary["decision→synth"].append(dec_to_synth)
            if item.get("transcript"):
                previous_transcript_ts = item["transcript"]

            fmt = lambda v: f"{v:9.0f}" if isinstance(v, float) else f"{'':>9}"
            print(f"{uid:<10} {str(speech_s):>8} {str(eou)[:6]:>7} {fmt(end_to_text)} "
                  f"{fmt(text_to_dec)} {fmt(dec_to_synth)} {str(item.get('rtf',''))[:5]:>5}  "
                  f"{','.join(flags):<8} {str(item.get('text',''))[:60]}")

    print("\n== stage summary (ms) ==")
    for stage, values in summary.items():
        if values:
            values = sorted(values)
            print(f"{stage:<24} n={len(values):<3} p50={values[len(values)//2]:7.0f}  "
                  f"p95={values[min(len(values)-1, int(len(values)*0.95))]:7.0f}  max={values[-1]:7.0f}")
        else:
            print(f"{stage:<24} no data")
    if args.json:
        print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())