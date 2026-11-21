"""Swedish TTS: Piper (VITS) voices via sherpa-onnx, phonemized with espeak-ng.

sherpa-onnx >= 2.0 has no built-in espeak, but Piper voices are trained on
espeak IPA, so espeak-ng's output can be fed in directly as phoneme codepoints.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import sherpa_onnx

VOICES = {
    "alma": "vits-piper-sv_SE-alma-medium",
    "lisa": "vits-piper-sv_SE-lisa-medium",
}


class PiperSwedish:
    def __init__(self, models_dir: Path, voice: str = "alma", num_threads: int = 2) -> None:
        d = models_dir / VOICES[voice]
        if not d.exists():
            raise FileNotFoundError(f"{d} missing; run `kiki fetch-models --server`")
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(d / f"sv_SE-{voice}-medium.onnx"), tokens=str(d / "tokens.txt")
                ),
                provider="cpu",
                num_threads=num_threads,
            )
        )
        if not cfg.validate():
            raise ValueError(f"invalid piper config: {cfg}")
        self._tts = sherpa_onnx.OfflineTts(cfg)
        self.sample_rate = self._tts.sample_rate
        self._symbols = {
            line.rsplit(" ", 1)[0]
            for line in (d / "tokens.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def phonemize(self, text: str) -> str:
        ipa = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", "sv", text], capture_output=True, text=True, check=True
        ).stdout.strip().replace("\n", " ")
        ph = "".join(c for c in ipa if c in self._symbols).strip()
        end = text.rstrip()[-1:] if text.rstrip()[-1:] in "?!." else "."
        return ph + end

    def __call__(self, text: str, speed: float = 1.0) -> np.ndarray:
        g = sherpa_onnx.GenerationConfig()
        g.sid = 0
        g.speed = speed
        g.phoneme_codepoints = [[ord(c) for c in self.phonemize(text)]]
        return np.asarray(self._tts.generate(text, g).samples, dtype=np.float32)
