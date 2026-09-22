"""Contract tests: the readiness registry and the mic's readiness gate.

The gate exists because a model load takes seconds: a fixture replay (or a demo)
must be able to wait until the transcriber can actually consume audio, and a
service must be able to announce that it is warm.
"""
import asyncio
import threading

from aria.core import readiness
from aria.core.config import Config
from helpers import attach, make_bus, run

from aria.audio.mic import MicService  # noqa: E402


def test_mark_and_wait():
    async def scenario():
        readiness.reset()
        assert not readiness.is_ready("stt")
        assert await readiness.wait_ready(["stt"], timeout=0.05) is False
        readiness.mark_ready("stt")
        assert readiness.is_ready("stt")
        assert await readiness.wait_ready(["stt"], timeout=0.05) is True
        assert await readiness.wait_ready([], timeout=0.05) is True
        readiness.mark_not_ready("stt")
        assert not readiness.is_ready("stt")
        readiness.reset()
    run(scenario())


def test_wait_ready_returns_when_signalled_later():
    async def scenario():
        readiness.reset()

        async def announce():
            await asyncio.sleep(0.05)
            readiness.mark_ready("stt")

        asyncio.ensure_future(announce())
        assert await readiness.wait_ready(["stt"], timeout=1.0) is True
        readiness.reset()
    run(scenario())


def _mic(cfg):
    svc = MicService()
    svc.attach(make_bus(), Config(cfg, "<test>"))
    svc._stop_evt = threading.Event()
    return svc


def test_mic_waits_for_dependencies_when_configured():
    readiness.reset()
    svc = _mic({"wait_for_ready": ["stt"], "readiness_timeout_s": 0.1,
                "heartbeat_interval": 0.05})
    assert svc._wait_ready_blocking() is False        # times out, logs, does not hang
    readiness.mark_ready("stt")
    assert svc._wait_ready_blocking() is True
    readiness.reset()


def test_mic_does_not_wait_when_not_configured():
    readiness.reset()
    svc = _mic({"heartbeat_interval": 0.05})
    assert svc._wait_ready_blocking() is True          # default: start immediately