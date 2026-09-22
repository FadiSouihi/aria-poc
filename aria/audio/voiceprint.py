"""VoiceprintService — speaker embeddings to suppress overheard speech.

Consumes ``TurnCompleted``, embeds the utterance, matches it against
``data/voices/<name>.npz`` galleries and publishes ``VoiceprintIdentified``.

Providers (config ``provider``): ``auto`` | ``resemblyzer`` | ``speechbrain``
| ``none``.

- **resemblyzer** (default): pure PyTorch d-vector encoder, no torchaudio
  dependency — SpeechBrain hard-imports torchaudio, and torchaudio has no
  build matching the torch version the vision stack is pinned to.
- **speechbrain**: ECAPA-TDNN, used automatically when torchaudio is present.

If no provider loads, the service logs once and stays idle: the gate then
falls back to engagement-only decisions, so the pipeline never depends on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from aria.audio.store import DEFAULT_AUDIO_STORE
from aria.core.events import Event
from aria.core.service import Service
from aria.perception.vision import resolve_repo_path

# Same-speaker cosine thresholds per embedding family. The wavlm value was
# calibrated on the fixtures in fixtures/audio (same voice 0.92, other voice
# 0.45-0.49); re-run tools/calibrate_voice.py with real enrolled voices.
PROVIDER_THRESHOLDS = {"wavlm-xvector": 0.70, "resemblyzer-dvector": 0.70, "speechbrain-ecapa": 0.35}

_SCHEMA = {
    "provider": ("auto", (str,)),            # auto | wavlm | resemblyzer | speechbrain | none
    "model": ("weights/wavlm-base-plus-sv.onnx", (str,)),
    "output": ("embeddings", (str,)),        # embeddings (better separation) | logits
    "max_seconds": (10.0, (int, float)),
    "voices_dir": ("data/voices", (str,)),
    "threshold": (0.0, (int, float)),        # 0 = provider default
    "min_audio_s": (0.6, (int, float)),
    "ort_threads": (2, (int,)),        # per-turn: a couple of threads help latency
    "ort_spinning": (False, (bool,)),
    "heartbeat_interval": (5.0, (int, float)),
}


class VoiceGallery:
    """Enrolled voice-prints: name → L2-normalized centroid (pure logic)."""

    def __init__(self) -> None:
        self.centroids: dict = {}

    def load(self, voices_dir: Path) -> "VoiceGallery":
        voices_dir = Path(voices_dir)
        if voices_dir.exists():
            for npz in sorted(voices_dir.glob("*.npz")):
                data = np.load(npz)
                emb = np.asarray(data["emb"], dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(emb))
                self.centroids[npz.stem] = emb / norm if norm > 0 else emb
        return self

    def match(self, emb: np.ndarray) -> tuple[Optional[str], float]:
        if not self.centroids:
            return None, -1.0
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        emb = emb / norm if norm > 0 else emb
        scored = sorted(
            ((name, float(np.dot(emb, centroid))) for name, centroid in self.centroids.items()),
            key=lambda pair: pair[1], reverse=True,
        )
        return scored[0]

    def __len__(self) -> int:
        return len(self.centroids)


class WavLmXVectorProvider:
    """WavLM speaker x-vectors (512-d) — raw-waveform ONNX, no feature
    engineering and no torchaudio, so it works on this pinned torch build."""

    name = "wavlm-xvector"

    def __init__(self, model_path: str, output: str = "embeddings",
                 max_seconds: float = 10.0, rate: int = 16000,
                 threads: Optional[int] = None, spinning: Optional[bool] = None) -> None:
        import onnxruntime as ort

        from aria.core.onnx import make_session

        ort.set_default_logger_severity(3)
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            raise FileNotFoundError(f"speaker model not found: {path}")
        # WavLM is heavy but runs once per turn: a couple of threads help latency,
        # spinning between turns only burns CPU.
        self.session = make_session(str(path), kind="voiceprint", threads=threads, spinning=spinning)
        self.input_name = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.output = output if output in self.outputs else self.outputs[0]
        self.max_samples = int(float(max_seconds) * rate)
        self.rate = rate

    def embed(self, audio: np.ndarray) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size > self.max_samples:
            samples = samples[: self.max_samples]
        if samples.size < self.rate // 4:
            samples = np.pad(samples, (0, self.rate // 4 - samples.size))
        outs = self.session.run(None, {self.input_name: samples.reshape(1, -1)})
        emb = np.asarray(outs[self.outputs.index(self.output)], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm > 0 else emb


class ResemblyzerProvider:
    """d-vector speaker encoder (256-d), pure PyTorch."""

    name = "resemblyzer-dvector"

    def __init__(self) -> None:
        from resemblyzer import VoiceEncoder

        self.encoder = VoiceEncoder("cpu")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        emb = self.encoder.embed_utterance(np.asarray(audio, dtype=np.float32))
        return np.asarray(emb, dtype=np.float32).reshape(-1)


class SpeechBrainEcapaProvider:
    """ECAPA-TDNN speaker embeddings (192-d) — needs torchaudio installed."""

    name = "speechbrain-ecapa"

    def __init__(self, source: str) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        self.model = EncoderClassifier.from_hparams(source=source, savedir=None, run_opts={"device": "cpu"})

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            tensor = self._torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
            emb = self.model.encode_batch(tensor).squeeze().detach().cpu().numpy()
        return np.asarray(emb, dtype=np.float32).reshape(-1)


def build_embedder(provider: str = "auto", model: str = "weights/wavlm-base-plus-sv.onnx",
                   output: str = "embeddings", threads: Optional[int] = None,
                   spinning: Optional[bool] = None):
    """Provider chain: returns (embedder, threshold) or raises."""
    order = {"auto": ["wavlm", "resemblyzer", "speechbrain"], "wavlm": ["wavlm"],
             "resemblyzer": ["resemblyzer"], "speechbrain": ["speechbrain"]}.get(provider)
    if order is None:
        raise ValueError(f"unknown voiceprint provider {provider!r}")
    errors = []
    for name in order:
        try:
            if name == "wavlm":
                return WavLmXVectorProvider(model, output=output, threads=threads,
                                            spinning=spinning), PROVIDER_THRESHOLDS["wavlm-xvector"]
            if name == "resemblyzer":
                return ResemblyzerProvider(), PROVIDER_THRESHOLDS["resemblyzer-dvector"]
            return SpeechBrainEcapaProvider(model), PROVIDER_THRESHOLDS["speechbrain-ecapa"]
        except Exception as exc:      # try the next provider
            errors.append(f"{name}: {exc}")
    raise RuntimeError("no voiceprint provider available (" + "; ".join(errors) + ")")


class VoiceprintService(Service):
    name = "audio.voiceprint"
    produces = ("VoiceprintIdentified",)
    consumes = ("TurnCompleted",)
    config_schema = _SCHEMA

    def __init__(self) -> None:
        super().__init__()
        self._sub = None
        self._embedder = None
        self._gallery: Optional[VoiceGallery] = None
        self._threshold = 0.35

    async def init(self) -> None:
        self._store = DEFAULT_AUDIO_STORE
        self._gallery = VoiceGallery().load(resolve_repo_path(str(self.config.get("voices_dir", "data/voices"))))
        provider = str(self.config.get("provider", "auto"))
        configured = float(self.config.get("threshold", 0.0))
        if provider == "none":
            self.log.info("Voice-prints disabled by config")
            return
        try:
            self._embedder, default_threshold = build_embedder(
                provider,
                str(self.config.get("model", "weights/wavlm-base-plus-sv.onnx")),
                str(self.config.get("output", "embeddings")),
                threads=self.config.get("ort_threads"),
                spinning=self.config.get("ort_spinning"),
            )
            self._threshold = configured if configured > 0 else default_threshold
            self.log.info("Voice-print model ready", provider=self._embedder.name,
                          threshold=self._threshold, gallery_size=len(self._gallery),
                          ort_threads=self.config.get("ort_threads"))
        except Exception as exc:
            if provider in ("wavlm", "resemblyzer", "speechbrain"):
                raise
            self.log.warning("Voice-print model unavailable; gate falls back to engagement only",
                             error=str(exc))

    async def on_start(self) -> None:
        self._sub = self.bus.subscribe("TurnCompleted", self._on_turn, policy="drop_new", maxsize=4)
        # Tell the gate whether matching is possible at all: with an empty gallery
        # (or no provider) its voice-print wait is pure added latency.
        active = bool(self._embedder is not None and self._gallery is not None and len(self._gallery) > 0)
        await self.bus.publish(Event("VoiceprintStatus", {
            "active": active,
            "gallery_size": len(self._gallery) if self._gallery is not None else 0,
            "provider": self._embedder.name if self._embedder is not None else None,
        }))

    async def on_stop(self) -> None:
        if self._sub is not None:
            self.bus.unsubscribe(self._sub)
            self._sub = None

    async def _on_turn(self, event: Event) -> None:
        if self._embedder is None:
            return
        payload = dict(event.payload)
        start = int(payload.get("start_index", 0))
        audio = self._store.read_since(start)
        end = payload.get("end_index")
        if end is not None:
            keep = max(0, int(end) - start)
            if audio.size > keep:
                audio = audio[:keep]
        min_samples = int(float(self.config.get("min_audio_s", 0.6)) * self._store.sample_rate)
        if audio.size < min_samples:
            self.metrics.inc("voiceprint.skipped_short")
            return
        import asyncio

        try:
            emb = await asyncio.to_thread(self._embedder.embed, audio)
        except Exception as exc:
            self.metrics.inc("voiceprint.errors")
            self.log.warning("Voice-print embedding failed", error=str(exc))
            return
        name, similarity = self._gallery.match(emb)
        known = name is not None and similarity >= self._threshold
        self.metrics.observe("voiceprint.similarity", float(similarity))
        self.log.info("Voice-print result", utterance_id=payload.get("utterance_id"),
                      name=name if known else "unknown", similarity=round(float(similarity), 3))
        await self.bus.publish(Event("VoiceprintIdentified", {
            "utterance_id": payload.get("utterance_id"),
            "name": name if known else None,
            "similarity": round(float(similarity), 3),
            "known": bool(known),
            "provider": self._embedder.name,
        }))