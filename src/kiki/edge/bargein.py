"""Barge-in: notice when someone talks over Kiki, without reacting to her own echo.

The Jabra's echo cancellation leaves a quiet residue of Kiki's voice in the mic;
a person talking at the table is much louder. So an interruption needs sustained,
confident speech that is also loud enough:

  - Silero VAD probability >= `vad_threshold` on most frames of a short window, and
  - those frames at or above `min_dbfs`.

Thresholds are deliberately conservative; the edge reports mic levels during
playback (`echo_dbfs` in the stats) so they can be tuned from real data.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from kiki.config import FRAME_MS


def frame_dbfs(frame: np.ndarray) -> float:
    x = frame.astype(np.float32) / 32768.0
    return 10 * math.log10(max(float(np.mean(x * x)), 1e-9))


class BargeIn:
    def __init__(self, vad, vad_threshold: float = 0.8, min_dbfs: float = -38.0,
                 min_speech_ms: int = 250, window_ms: int = 400) -> None:
        self.vad = vad
        self.vad_threshold = vad_threshold
        self.min_dbfs = min_dbfs
        self.need = max(1, min_speech_ms // FRAME_MS)
        self._window: deque[bool] = deque(maxlen=max(self.need, window_ms // FRAME_MS))
        self.levels: list[float] = []  # mic level while Kiki talks (for stats/tuning)

    def reset(self) -> None:
        self._window.clear()
        if hasattr(self.vad, "reset"):
            self.vad.reset()

    def update(self, frame: np.ndarray) -> bool:
        """Feed one mic frame recorded while Kiki is talking; True = interrupted."""
        db = frame_dbfs(frame)
        self.levels.append(db)
        p = self.vad(frame)
        self._window.append(p >= self.vad_threshold and db >= self.min_dbfs)
        return sum(self._window) >= self.need
