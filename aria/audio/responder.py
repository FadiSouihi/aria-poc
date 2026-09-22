"""StubResponderService — a placeholder dialogue source for Phase 2.

Phase 3 brings the real dialogue FSM + LLM router. Until then this service
closes the loop so the audio frontend is demonstrable end-to-end: it takes an
accepted utterance and asks TTS to speak a templated acknowledgement. It is
deliberately trivial and clearly marked as a stub — swap it out, don't grow it.

Replies are templated *per language* (`templates`), so a French question is
answered with a French sentence spoken by a French voice. The words the person
said are never translated; only the wrapper is localised.
"""
from __future__ import annotations

from aria.core.events import Event
from aria.core.service import Service

_SCHEMA = {
    "template": ("I heard you say: {text}", (str,)),
    "templates": ({}, (dict,)),        # {"fr": "J'ai entendu : {text}", "ar": "سمعتك تقول: {text}"}
    "min_interval_s": (0.0, (int, float)),
    "heartbeat_interval": (5.0, (int, float)),
}


class StubResponderService(Service):
    name = "audio.responder"
    produces = ("SpeakRequest",)
    consumes = ("UtteranceAccepted",)
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._sub = None

    async def on_start(self) -> None:
        self._sub = self.bus.subscribe("UtteranceAccepted", self._on_accepted, policy="drop_new", maxsize=2)

    async def on_stop(self) -> None:
        if self._sub is not None:
            self.bus.unsubscribe(self._sub)
            self._sub = None

    def template_for(self, language) -> str:
        code = (str(language) if language else "")[:2].lower()
        templates = {str(k).lower()[:2]: str(v) for k, v in (self.config.get("templates") or {}).items()}
        if code and code in templates:
            return templates[code]
        return str(self.config.get("template", "I heard you say: {text}"))

    async def _on_accepted(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        language = event.payload.get("language")
        reply = self.template_for(language).format(text=text)
        self.metrics.inc("responder.replies")
        self.log.info("Stub reply", utterance_id=event.payload.get("utterance_id"),
                      language=language, reply=reply)
        await self.bus.publish(Event("SpeakRequest", {
            "text": reply,
            "utterance_id": event.payload.get("utterance_id"),
            # Language travels with the reply so TTS answers in the language the
            # person actually spoke — the text itself is never translated.
            "language": language,
        }))