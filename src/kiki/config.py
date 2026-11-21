"""Runtime settings, read from environment variables (and .env via docker compose)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # Silero VAD needs exactly 512 samples (32 ms) at 16 kHz
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    models_dir: Path = field(default_factory=lambda: Path(_env("KIKI_MODELS_DIR", "models")))

    # VAD / segmentation
    vad_threshold: float = float(_env("KIKI_VAD_THRESHOLD", "0.5"))
    min_speech_ms: int = int(_env("KIKI_MIN_SPEECH_MS", "96"))
    # Pipecat recommends ~0.2 s of VAD silence before asking Smart Turn
    pause_ms: int = int(_env("KIKI_PAUSE_MS", "200"))
    preroll_ms: int = int(_env("KIKI_PREROLL_MS", "300"))

    # Turn detection
    turn_threshold: float = float(_env("KIKI_TURN_THRESHOLD", "0.5"))
    # If Smart Turn keeps saying "incomplete", end the turn anyway after this much silence
    max_pause_ms: int = int(_env("KIKI_MAX_PAUSE_MS", "2000"))

    # Speaker ID: cosine similarity to accept a known voice. Calibrated on synthetic
    # voices (same voice >= 0.74 on >1 s clips, different voices up to ~0.8);
    # re-tune with real voices from the Pi mic.
    speaker_threshold: float = float(_env("KIKI_SPEAKER_THRESHOLD", "0.68"))

    # Audio device (sounddevice name or index); empty = system default
    input_device: str = _env("KIKI_INPUT_DEVICE", "")


def load() -> Settings:
    return Settings()
