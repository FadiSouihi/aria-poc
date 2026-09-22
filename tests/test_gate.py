"""Contract tests: addressed-speech gate (false-response control)."""
from aria.audio.gate import AddressedSpeechGate, VoiceGateService
from aria.core.events import Event
from helpers import Collector, attach, make_bus, run


def _gate(**kwargs):
    defaults = dict(min_chars=2, min_speech_s=0.4, require_engagement=True,
                    engaged_states=("NEAR", "ENGAGED"), speaker_gate=True,
                    min_speaker_similarity=0.35)
    defaults.update(kwargs)
    return AddressedSpeechGate(**defaults)


def test_accepts_addressed_speech_from_engaged_person():
    accepted, reason = _gate().decide(text="where is the library", duration_s=2.0,
                                      engaged_present=True, engaged_identity="fedi",
                                      speaker="fedi", speaker_similarity=0.8)
    assert accepted and reason == "addressed"


def test_rejects_when_nobody_is_engaged():
    """FUNC-14: speech with no one in front of the robot must not trigger a reply."""
    accepted, reason = _gate().decide(text="did you see the match", duration_s=2.0,
                                      engaged_present=False, engaged_identity=None,
                                      speaker=None, speaker_similarity=None)
    assert not accepted and reason == "no_one_engaged"


def test_rejects_bystander_speaker():
    accepted, reason = _gate().decide(text="hello there", duration_s=2.0,
                                      engaged_present=True, engaged_identity="fedi",
                                      speaker="someone_else", speaker_similarity=0.9)
    assert not accepted and reason == "bystander_speaker"


def test_rejects_weak_voiceprint():
    accepted, reason = _gate().decide(text="hello there", duration_s=2.0,
                                      engaged_present=True, engaged_identity=None,
                                      speaker="maybe", speaker_similarity=0.1)
    assert not accepted and reason == "weak_voiceprint"


def test_rejects_empty_and_too_short():
    gate = _gate()
    assert gate.decide(text="", duration_s=2.0, engaged_present=True, engaged_identity=None,
                       speaker=None, speaker_similarity=None) == (False, "empty_transcript")
    assert gate.decide(text="ok", duration_s=0.1, engaged_present=True, engaged_identity=None,
                       speaker=None, speaker_similarity=None) == (False, "too_short")


def test_gate_service_publishes_decisions():
    async def scenario():
        bus = make_bus()
        await bus.start()
        accepted, rejected = Collector(), Collector()
        bus.subscribe("UtteranceAccepted", accepted, policy="block", maxsize=8)
        bus.subscribe("UtteranceRejected", rejected, policy="block", maxsize=8)
        svc = attach(VoiceGateService(), bus, {"voiceprint_wait_s": 0.0})
        await svc.init()
        await svc.start()

        # nobody engaged yet → rejected
        await svc._on_utterance(Event("UtteranceHeard", {
            "utterance_id": "u1", "text": "hello", "duration_s": 1.5}))
        await rejected.wait_for(1)

        # engaged track present → accepted
        await svc._on_tracks(Event("TrackStates", {"tracks": [
            {"id": 1, "bbox": [0, 0, 10, 10], "state": "ENGAGED", "identity": "fedi"}]}))
        await svc._on_utterance(Event("UtteranceHeard", {
            "utterance_id": "u2", "text": "where is the library", "duration_s": 1.5}))
        await accepted.wait_for(1)

        assert rejected.events[0].payload["gate_reason"] == "no_one_engaged"
        assert accepted.events[0].payload["gate_reason"] == "addressed"
        assert accepted.events[0].payload["engaged_identity"] == "fedi"
        await svc.stop()
        await bus.stop()
    run(scenario())