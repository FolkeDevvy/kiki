"""Silero VAD (ONNX, no torch)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from kiki.config import FRAME_SAMPLES, SAMPLE_RATE

_CONTEXT = 64  # samples of the previous chunk prepended at 16 kHz


class SileroVad:
    def __init__(self, model_path: Path) -> None:
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample frame (int16 or float32 in [-1, 1])."""
        if frame.shape[-1] != FRAME_SAMPLES:
            raise ValueError(f"need {FRAME_SAMPLES} samples, got {frame.shape[-1]}")
        x = frame.astype(np.float32)
        if frame.dtype == np.int16:
            x /= 32768.0
        x = np.concatenate([self._context, x[None, :]], axis=1)
        out, self._state = self._sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -_CONTEXT:]
        return float(out[0][0])
