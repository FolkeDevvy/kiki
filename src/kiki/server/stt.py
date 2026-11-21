"""Local English ASR: NVIDIA Parakeet TDT 0.6B v2 via sherpa-onnx."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

from kiki.config import SAMPLE_RATE

MODEL_DIR = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"


class Parakeet:
    def __init__(self, models_dir: Path, num_threads: int = 4) -> None:
        d = models_dir / MODEL_DIR
        if not d.exists():
            raise FileNotFoundError(f"{d} missing; run `kiki fetch-models --server`")
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

    def __call__(self, audio: np.ndarray) -> str:
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x /= 32768.0
        s = self._rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, x)
        self._rec.decode_streams([s])
        return s.result.text.strip()
