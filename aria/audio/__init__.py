"""ARIA-POC audio frontend (Phase 2).

Mic capture → rolling AudioStore → VAD → three-tier end-of-turn → STT →
addressed-speech gate → TTS with barge-in. Same conventions as
``aria.perception``: metadata on the bus, samples in a local store, one
service per responsibility, everything config-swappable.
"""