"""Pipecat Smart Turn v3: has the current speaker finished their turn?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from kiki.audio.features import log_mel
from kiki.config import SAMPLE_RATE

MAX_SECONDS = 8


def fit_to_window(audio: np.ndarray, seconds: int = MAX_SECONDS) -> np.ndarray:
    """Keep the last `seconds`, or left-pad with zeros (matches upstream)."""
    n = seconds * SAMPLE_RATE
    if len(audio) >= n:
        return audio[-n:]
    return np.pad(audio, (n - len(audio), 0))


class SmartTurn:
    def __init__(self, model_path: Path, threads: int = 0) -> None:
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        if threads > 0:
            so.intra_op_num_threads = threads  # cap the CPU burst per call (Pi power budget)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    def features(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x /= 32768.0
        return log_mel(fit_to_window(x))[None, :, :]

    def __call__(self, audio: np.ndarray) -> float:
        """P(turn complete) for the turn audio so far (int16 or float32)."""
        out = self._sess.run(None, {self._input: self.features(audio)})
        return float(np.asarray(out[0]).reshape(-1)[0])
