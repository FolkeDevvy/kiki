"""Who is talking: voiceprints + online assignment.

Enrolled people are persisted (data/speakers.json). Unknown voices become
session guests ("Guest-1", ...) so the conversation still has consistent
labels; a guest can later be named (milestone 3: "Kiki, I'm Anna").
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sherpa_onnx

from kiki.config import SAMPLE_RATE

MODEL = "wespeaker_en_voxceleb_resnet34_LM.onnx"
MIN_RELIABLE_S = 1.0  # shorter segments give noisy embeddings
CONFIDENT_NEW_S = 2.5  # long enough that an ambiguous match is better treated as a new voice


class Embedder:
    def __init__(self, models_dir: Path, num_threads: int = 2) -> None:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models_dir / MODEL), num_threads=num_threads, provider="cpu"
        )
        if not cfg.validate():
            raise FileNotFoundError(f"bad speaker model config (run `kiki fetch-models --server`): {cfg}")
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.dim = self._ex.dim

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x /= 32768.0
        s = self._ex.create_stream()
        s.accept_waveform(sample_rate=SAMPLE_RATE, waveform=x)
        s.input_finished()
        e = np.asarray(self._ex.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


@dataclass
class Voice:
    name: str
    centroid: np.ndarray
    n: int = 1
    enrolled: bool = False

    def update(self, e: np.ndarray, max_n: int = 50) -> None:
        # running mean, capped so a voice keeps adapting to the room
        self.n = min(self.n + 1, max_n)
        c = self.centroid + (e - self.centroid) / self.n
        self.centroid = c / (np.linalg.norm(c) + 1e-9)


@dataclass
class Registry:
    path: Path | None = None
    threshold: float = 0.68  # cosine similarity to accept a match (see Settings)
    new_voice_margin: float = 0.12  # below threshold-margin we're confident it's someone new
    voices: list[Voice] = field(default_factory=list)
    merged: dict[str, str] = field(default_factory=dict)  # old guest name -> surviving name
    _guests: int = 0

    @classmethod
    def load(cls, path: Path, **kw) -> "Registry":
        reg = cls(path=path, **kw)
        if path.exists():
            for name, v in json.loads(path.read_text()).items():
                reg.voices.append(Voice(name, np.asarray(v["centroid"], np.float32), v["n"], enrolled=True))
        return reg

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {v.name: {"centroid": v.centroid.tolist(), "n": v.n} for v in self.voices if v.enrolled}
        self.path.write_text(json.dumps(data))

    def enroll(self, name: str, e: np.ndarray) -> Voice:
        for v in self.voices:
            if v.name == name:
                v.update(e)
                v.enrolled = True
                self.save()
                return v
        v = Voice(name, e, enrolled=True)
        self.voices.append(v)
        self.save()
        return v

    def reset_guests(self) -> None:
        """Forget unnamed voices (they're only meaningful within one conversation)."""
        self.voices = [v for v in self.voices if v.enrolled]
        self.merged.clear()
        self._guests = 0

    def rename(self, old: str, new: str) -> bool:
        """Name a guest (or fix a name) and persist the voiceprint."""
        for v in self.voices:
            if v.name == old:
                v.name, v.enrolled = new, True
                self.save()
                return True
        return False

    def best(self, e: np.ndarray) -> tuple[Voice | None, float]:
        if not self.voices:
            return None, -1.0
        sims = [float(v.centroid @ e) for v in self.voices]
        i = int(np.argmax(sims))
        return self.voices[i], sims[i]

    def assign(self, e: np.ndarray, duration_s: float, fallback: str | None = None) -> tuple[str, float]:
        """Return (speaker name, similarity). Creates guests for clear new voices."""
        v, sim = self.best(e)
        reliable = duration_s >= MIN_RELIABLE_S
        if v is not None and sim >= self.threshold:
            if reliable:
                v.update(e)
                v = self._maybe_merge(v)
            return v.name, sim
        if not reliable:
            return fallback or (v.name if v is not None and sim > self.threshold - self.new_voice_margin else "?"), sim
        if v is None or sim < self.threshold - self.new_voice_margin or duration_s >= CONFIDENT_NEW_S:
            self._guests += 1
            g = Voice(f"Guest-{self._guests}", e)
            self.voices.append(g)
            return g.name, sim
        # ambiguous zone, short clip: don't create a voice, don't pollute a centroid
        return fallback or v.name, sim

    def _maybe_merge(self, v: Voice) -> Voice:
        """If a voice drifted onto another one, they were the same person: merge.
        Keeps the enrolled (or older) name. Returns the surviving voice."""
        for other in self.voices:
            if other is v or float(other.centroid @ v.centroid) < self.threshold + 0.05:
                continue
            keep, drop = (other, v) if (other.enrolled, -self.voices.index(other)) >= (v.enrolled, -self.voices.index(v)) else (v, other)
            n = keep.n + drop.n
            c = (keep.centroid * keep.n + drop.centroid * drop.n) / n
            keep.centroid, keep.n = c / (np.linalg.norm(c) + 1e-9), min(n, 50)
            self.voices.remove(drop)
            self.merged[drop.name] = keep.name
            if keep.enrolled:
                self.save()
            return keep
        return v
