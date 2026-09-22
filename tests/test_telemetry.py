"""Contract tests: structured logging, masking, timeline, metrics."""
import json

from aria.core.context import bind
from aria.core.events import Event
from aria.core.telemetry.logging import get_logger, setup_logging
from aria.core.telemetry.metrics import Metrics
from aria.core.telemetry.timeline import TimelineRecorder
from helpers import make_bus, run


def test_json_logs_carry_context_fields_and_masking(tmp_path):
    setup_logging(level="INFO", console="quiet", dir_name=str(tmp_path))
    log = get_logger("aria.test.telemetry")
    with bind(session_id="s1", utterance_id="u9"):
        log.info("User ID 123456789 checked in", seq=3)

    lines = (tmp_path / "aria.log").read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record["msg"] == "User ID [REDACTED-ID] checked in"
    assert record["context"] == {"session_id": "s1", "utterance_id": "u9"}
    assert record["fields"] == {"seq": 3}
    assert record["service"] == "aria.test.telemetry"
    assert record["level"] == "INFO"


def test_metrics_counters_and_percentiles():
    m = Metrics()
    for i in range(1, 11):
        m.observe("latency_s", float(i))
    m.inc("frames", 5)
    snap = m.snapshot()
    hist = snap["histograms"]["latency_s"]
    assert hist["count"] == 10
    assert hist["p50"] == 6.0
    assert hist["p95"] == 10.0
    assert hist["max"] == 10.0
    assert snap["counters"]["frames"] == 5


def test_timeline_records_all_and_dumps_ring_on_anomaly(tmp_path):
    async def scenario():
        recorder = TimelineRecorder(dir_name=str(tmp_path), ring_size=3)
        bus = make_bus(maxsize=64)
        recorder.install(bus)
        await bus.start()
        for i in range(5):
            await bus.publish(Event("Frame", {"seq": i}))
        await bus.publish(Event("ServiceCrashed", {"service": "x"}))
        await bus.stop()
        recorder.close()   # writes are buffered; close (or an anomaly) makes them durable

        full = (tmp_path / "timeline_full.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(full) == 6  # every event tapped

        dumps = list(tmp_path.glob("timeline_*_ServiceCrashed.jsonl"))
        assert len(dumps) == 1
        ring = [json.loads(line) for line in dumps[0].read_text(encoding="utf-8").splitlines()]
        assert len(ring) == 3               # ring buffer capped
        assert ring[-1]["name"] == "ServiceCrashed"
    run(scenario())


def test_timeline_flushes_the_full_log_on_anomaly(tmp_path):
    """A crash must not cost the events that led up to it (no close() needed)."""
    async def scenario():
        recorder = TimelineRecorder(dir_name=str(tmp_path), ring_size=5)
        bus = make_bus(maxsize=64)
        recorder.install(bus)
        await bus.start()
        await bus.publish(Event("Frame", {"seq": 1}))
        await bus.publish(Event("ServiceCrashed", {"service": "x"}))
        await bus.stop()
        # deliberately NOT closed: the anomaly itself has to flush
        full = (tmp_path / "timeline_full.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(full) == 2
    run(scenario())


def test_timeline_sampling_trades_fidelity_for_cost(tmp_path):
    """High-rate metadata can be sampled or dropped; the ring keeps the rest."""
    async def scenario():
        recorder = TimelineRecorder(
            dir_name=str(tmp_path), ring_size=10,
            sample={"Frame": 2, "AudioChunk": 0},
        )
        bus = make_bus(maxsize=64)
        recorder.install(bus)
        await bus.start()
        for i in range(6):
            await bus.publish(Event("Frame", {"seq": i}))
            await bus.publish(Event("AudioChunk", {"seq": i}))
        await bus.publish(Event("SpeechStarted", {}))
        await bus.stop()
        recorder.close()

        names = [json.loads(line)["name"]
                 for line in (tmp_path / "timeline_full.jsonl").read_text(encoding="utf-8").splitlines()]
        assert names.count("Frame") == 3          # 1-in-2 of 6
        assert names.count("AudioChunk") == 0     # dropped entirely
        assert names.count("SpeechStarted") == 1  # untouched by the policy
        assert recorder.skipped == 9
        assert len(recorder) == 4                 # sampled-out events never reach the ring
    run(scenario())
