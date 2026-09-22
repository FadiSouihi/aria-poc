"""One-off calibration: same-speaker vs cross-speaker cosine for the WavLM x-vectors."""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aria.audio.store import read_wav
from aria.audio.voiceprint import WavLmXVectorProvider

MODEL = ROOT / "weights" / "wavlm-base-plus-sv.onnx"
paths = {n: ROOT / "fixtures" / "audio" / f"{n}.wav" for n in ("greeting", "greeting2", "question")}
audio = {n: read_wav(p)[0].astype(np.float32) for n, p in paths.items()}

for output in ("logits", "embeddings"):
    prov = WavLmXVectorProvider(str(MODEL), output=output)
    emb = {n: prov.embed(a) for n, a in audio.items()}
    same = float(np.dot(emb["greeting"], emb["greeting2"]))       # both edge voice A
    cross1 = float(np.dot(emb["greeting"], emb["question"]))      # voice B
    cross2 = float(np.dot(emb["greeting2"], emb["question"]))
    print(f"output={output:11s} dim={emb['greeting'].size} "
          f"same={same:.3f} cross={cross1:.3f}/{cross2:.3f} margin={same - max(cross1, cross2):+.3f}")
    # sanity: does a truncated version of the same clip stay close to itself?
    half = prov.embed(audio["greeting"][: audio["greeting"].size // 2])
    print(f"{'':19s} self-half={float(np.dot(emb['greeting'], half)):.3f}")