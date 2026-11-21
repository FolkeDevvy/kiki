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

    # Turn detection. "auto" = int8 model on x86, fp32 on ARM; or a file name in models/
    smart_turn_model: str = _env("KIKI_SMART_TURN_MODEL", "auto")
    turn_threshold: float = float(_env("KIKI_TURN_THRESHOLD", "0.5"))
    # If Smart Turn keeps saying "incomplete", end the turn anyway after this much silence
    max_pause_ms: int = int(_env("KIKI_MAX_PAUSE_MS", "2000"))

    # Speaker ID: cosine similarity to accept a known voice. Calibrated on synthetic
    # voices (same voice >= 0.74 on >1 s clips, different voices up to ~0.8);
    # re-tune with real voices from the Pi mic.
    speaker_threshold: float = float(_env("KIKI_SPEAKER_THRESHOLD", "0.68"))

    # LLM (llama.cpp server, OpenAI-compatible)
    llm_url: str = _env("KIKI_LLM_URL", "http://localhost:8081")
    # Responder
    kiki_voice: str = _env("KIKI_VOICE", "af_heart")  # Kokoro voice for English
    kiki_voice_sv: str = _env("KIKI_VOICE_SV", "alma")  # Piper voice for Swedish (alma | lisa)
    stt_model: str = _env("KIKI_STT_MODEL", "v3")  # Parakeet: v3 = Swedish + English, v2 = English only
    engagement_window_s: float = float(_env("KIKI_ENGAGEMENT_WINDOW_S", "30"))
    # 0 = only when addressed, 1 = occasional chime-ins (default), 2 = eager
    proactivity: int = int(_env("KIKI_PROACTIVITY", "1"))
    history_turns: int = int(_env("KIKI_HISTORY_TURNS", "40"))

    # Diagnostics HTTP port on the server (see server/diagnostics.py)
    debug_port: int = int(_env("KIKI_DEBUG_PORT", "8766"))

    # Audio device (sounddevice name or index); empty = system default
    input_device: str = _env("KIKI_INPUT_DEVICE", "")


def load() -> Settings:
    return Settings()
