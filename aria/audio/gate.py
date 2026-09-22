"""VoiceGateService — decides whether an utterance is *addressed to ARIA*.

This is the false-response control (FUNC-14 "robot answered the TV / a
passer-by"): an utterance is only accepted when someone is actually engaged
with the robot and the speaker is not a known *other* person talking nearby.

``AddressedSpeechGate`` is pure logic so every rejection reason is
deterministic and unit-tested; the service just supplies the current scene
state and speaker evidence.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from aria.core.events import Event
from aria.core.service import Service

_SCHEMA = {
    "min_chars": (2, (int,)),
    "min_speech_s": (0.4, (int, float)),
    "require_engagement": (True, (bool,)),
    "engaged_states": (["NEAR", "ENGAGED"], (list,)),
    "speaker_gate": (True, (bool,)),
    "min_speaker_similarity": (0.35, (int, float)),
    "voiceprint_wait_s": (0.15, (int, float)),
    "heartbeat_interval": (5.0, (int, float)),
}


class AddressedSpeechGate:
    """Pure decision logic → (accepted: bool, reason: str)."""

    def __init__(self, min_chars: int = 2, min_speech_s: float = 0.4,
                 require_engagement: bool = True, engaged_states=("NEAR", "ENGAGED"),
                 speaker_gate: bool = True, min_speaker_similarity: float = 0.35) -> None:
        self.min_chars = int(min_chars)
        self.min_speech_s = float(min_speech_s)
        self.require_engagement = bool(require_engagement)
        self.engaged_states = tuple(engaged_states)
        self.speaker_gate = bool(speaker_gate)
        self.min_speaker_similarity = float(min_speaker_similarity)

    def decide(self, *, text: str, duration_s: float, engaged_present: bool,
               engaged_identity: Optional[str], speaker: Optional[str],
               speaker_similarity: Optional[float]) -> tuple[bool, str]:
        if len((text or "").strip()) < self.min_chars:
            return False, "empty_transcript"
        if duration_s < self.min_speech_s:
            return False, "too_short"
        if self.require_engagement and not engaged_present:
            return False, "no_one_engaged"
        if self.speaker_gate and speaker:
            if speaker_similarity is not None and speaker_similarity < self.min_speaker_similarity:
                return False, "weak_voiceprint"
            if engaged_identity and speaker != engaged_identity:
                return False, "bystander_speaker"
        return True, "addressed"


class VoiceGateService(Service):
    name = "audio.gate"
    produces = ("UtteranceAccepted", "UtteranceRejected")
    consumes = ("UtteranceHeard", "TrackStates", "VoiceprintIdentified")
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._subs = []
        self._gate: Optional[AddressedSpeechGate] = None
        self._tracks: list = []
        self._speakers: dict = {}
        self._voiceprints_active = False     # set by VoiceprintStatus; gates the wait

    async def init(self) -> None:
        self._gate = AddressedSpeechGate(
            min_chars=int(self.config.get("min_chars", 2)),
            min_speech_s=float(self.config.get("min_speech_s", 0.4)),
            require_engagement=bool(self.config.get("require_engagement", True)),
            engaged_states=list(self.config.get("engaged_states", ["NEAR", "ENGAGED"])),
            speaker_gate=bool(self.config.get("speaker_gate", True)),
            min_speaker_similarity=float(self.config.get("min_speaker_similarity", 0.35)),
        )

    async def on_start(self) -> None:
        self._subs.append(self.bus.subscribe("UtteranceHeard", self._on_utterance, policy="drop_new", maxsize=4))
        self._subs.append(self.bus.subscribe("TrackStates", self._on_tracks, policy="drop_oldest", maxsize=2))
        self._subs.append(self.bus.subscribe("VoiceprintIdentified", self._on_voiceprint, policy="drop_new", maxsize=8))
        self._subs.append(self.bus.subscribe("VoiceprintStatus", self._on_voiceprint_status, policy="drop_new", maxsize=2))

    async def on_stop(self) -> None:
        for sub in self._subs:
            self.bus.unsubscribe(sub)
        self._subs.clear()

    async def _on_tracks(self, event: Event) -> None:
        self._tracks = event.payload.get("tracks", [])

    async def _on_voiceprint(self, event: Event) -> None:
        utterance_id = event.payload.get("utterance_id")
        if utterance_id:
            self._speakers[utterance_id] = event.payload

    async def _on_voiceprint_status(self, event: Event) -> None:
        """Whether a voice-print gallery exists at all (empty → don't wait)."""
        self._voiceprints_active = bool(event.payload.get("active"))

    def _scene(self) -> tuple[bool, Optional[str]]:
        engaged = [t for t in self._tracks if t.get("state") in tuple(self.config.get("engaged_states", ["NEAR", "ENGAGED"]))]
        if not engaged:
            return False, None
        best = next((t for t in engaged if t.get("state") == "ENGAGED"), engaged[0])
        return True, best.get("identity")

    async def _on_utterance(self, event: Event) -> None:
        payload = dict(event.payload)
        utterance_id = payload.get("utterance_id")
        wait = float(self.config.get("voiceprint_wait_s", 0.15))
        # Only wait for a voice-print when there is a gallery to match against;
        # otherwise this is pure added latency on every reply.
        if wait > 0 and self._voiceprints_active:
            await asyncio.sleep(wait)   # let the voice-print result land first
        speaker_info = self._speakers.pop(utterance_id, None) or {}
        engaged_present, engaged_identity = self._scene()
        accepted, reason = self._gate.decide(
            text=str(payload.get("text", "")),
            duration_s=float(payload.get("duration_s", 0.0)),
            engaged_present=engaged_present,
            engaged_identity=engaged_identity,
            speaker=speaker_info.get("name") or payload.get("speaker"),
            speaker_similarity=speaker_info.get("similarity", payload.get("voiceprint_similarity")),
        )
        payload["gate_reason"] = reason
        payload["engaged_identity"] = engaged_identity
        if accepted:
            self.metrics.inc("gate.accepted")
            self.log.info("Utterance accepted", utterance_id=utterance_id, reason=reason,
                          text=payload.get("text"))
            await self.bus.publish(Event("UtteranceAccepted", payload))
        else:
            self.metrics.inc("gate.rejected")
            self.metrics.inc(f"gate.rejected.{reason}")
            self.log.info("Utterance rejected", utterance_id=utterance_id, reason=reason,
                          text=payload.get("text"))
            await self.bus.publish(Event("UtteranceRejected", payload))