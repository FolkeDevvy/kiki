"""Local TTS: Kokoro 82M (multi-lang v1.0) via sherpa-onnx."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

from kiki.server.phonemize import Phonemizer

MODEL_DIR = "kokoro-multi-lang-v1_0"
# speaker ids in kokoro-multi-lang-v1_0 (English subset)
VOICES = {
    "af_alloy": 0, "af_aoede": 1, "af_bella": 2, "af_heart": 3, "af_jessica": 4, "af_kore": 5,
    "af_nicole": 6, "af_nova": 7, "af_river": 8, "af_sarah": 9, "af_sky": 10, "am_adam": 11,
    "am_echo": 12, "am_eric": 13, "am_fenrir": 14, "am_liam": 15, "am_michael": 16, "am_onyx": 17,
    "am_puck": 18, "bf_alice": 20, "bf_emma": 21, "bf_isabella": 22, "bf_lily": 23,
    "bm_daniel": 24, "bm_fable": 25, "bm_george": 26, "bm_lewis": 27,
}


class Kokoro:
    def __init__(self, models_dir: Path, voice: str = "af_heart", num_threads: int = 4) -> None:
        d = models_dir / MODEL_DIR
        if not d.exists():
            raise FileNotFoundError(f"{d} missing; run `kiki fetch-models --server`")
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(d / "model.onnx"),
                    voices=str(d / "voices.bin"),
                    tokens=str(d / "tokens.txt"),
                    lexicon=str(d / "lexicon-us-en.txt"),
                ),
                provider="cpu",
                num_threads=num_threads,
            ),
            max_num_sentences=1,
        )
        if not cfg.validate():
            raise ValueError(f"invalid kokoro config: {cfg}")
        self._tts = sherpa_onnx.OfflineTts(cfg)
        self.sample_rate = self._tts.sample_rate
        self.voice = voice
        self.phonemize = Phonemizer(d / "lexicon-us-en.txt", d / "tokens.txt")

    def __call__(self, text: str, voice: str | None = None, speed: float = 1.0) -> np.ndarray:
        """float32 audio at self.sample_rate."""
        g = sherpa_onnx.GenerationConfig()
        g.sid = VOICES[voice or self.voice]
        g.speed = speed
        g.phoneme_codepoints = [[ord(c) for c in self.phonemize(text)]]  # one "sentence"
        audio = self._tts.generate(text, g)
        return np.asarray(audio.samples, dtype=np.float32)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    n = int(round(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
