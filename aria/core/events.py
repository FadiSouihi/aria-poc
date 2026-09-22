"""Event vocabulary for the ARIA-POC event bus.

Every cross-service message is an :class:`Event` carrying a name from
``REGISTRY`` plus a JSON-serializable payload. Publishing an unregistered name
is allowed but counted and logged as a warning — the registry keeps the
vocabulary documented and reviewable (one line to add an event).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict

REGISTRY: Dict[str, str] = {
    # --- Perception -------------------------------------------------------
    "Frame": "A camera frame was captured (metadata only; pixels stay local).",
    "TrackAppeared": "A person track appeared in the scene.",
    "TrackLost": "A person track disappeared from the scene.",
    "FaceMatched": "A track's face identity was matched to a profile.",
    "TamperDetected": "The camera feed looks obstructed or covered.",
    "SceneTick": "Raw scene snapshot from vision (active track ids/bboxes).",
    "TrackStates": "SceneManager's per-track state snapshot (id, bbox, state, identity).",
    # --- Audio ------------------------------------------------------------
    "AudioChunk": "A block of microphone audio was captured (samples stay local).",
    "AudioSourceFinished": "A non-looping replay source reached its end (replay runs can stop here).",
    "SpeechStarted": "VAD detected the start of speech.",
    "SpeechEnded": "VAD detected the end of a speech segment.",
    "TurnCompleted": "End-of-turn decided (three-tier EOU); the segment is ready to transcribe.",
    "UtteranceHeard": "A complete utterance was transcribed.",
    "UtteranceAccepted": "The addressed-speech gate accepted an utterance for response.",
    "UtteranceRejected": "The gate rejected an utterance (not addressed / bystander / noise).",
    "VoiceprintIdentified": "A speaker voice-print was matched to an enrolled profile.",
    "VoiceprintStatus": "Whether voice-print matching is possible (provider loaded + gallery non-empty).",
    "SpeakRequest": "A reply is ready to be spoken (text + language); consumed by TTS.",
    "SpeechSynthesized": "TTS produced audio for a response.",
    "BargeIn": "The user started speaking while the robot was speaking; playback stopped.",
    # --- Session ----------------------------------------------------------
    "SessionOpened": "A session was opened and bound to an identity.",
    "SessionClosed": "A session was closed.",
    # --- Supervision ------------------------------------------------------
    "DeviceLost": "A peripheral (camera/mic) disappeared.",
    "DeviceRestored": "A peripheral came back.",
    "ServiceCrashed": "A service task crashed or its heartbeat went stale.",
    "ServiceRestarted": "The watchdog restarted a service.",
    "GovernorThrottled": "The governor changed pacing due to resource pressure.",
    "Anomaly": "Generic anomaly marker (triggers a timeline dump).",
}

# Events whose occurrence auto-dumps the in-memory ring buffer to disk.
ANOMALY_EVENTS = frozenset(
    {
        "ServiceCrashed",
        "ServiceRestarted",
        "GovernorThrottled",
        "TamperDetected",
        "TamperCleared",
        "DeviceLost",
        "DeviceRestored",
        "Anomaly",
    }
)


@dataclass(frozen=True)
class Event:
    """A typed message on the bus.

    ``context`` is filled automatically from the correlation context at
    publish time when the publisher did not set it explicitly.
    """

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        ctx = " ".join(f"{k}={v}" for k, v in self.context.items())
        ctx = f" ({ctx})" if ctx else ""
        payload = " ".join(f"{k}={v}" for k, v in self.payload.items())
        payload = f" {payload}" if payload else ""
        return f"{self.name}{payload}{ctx}"
