"""Edge health numbers, sent to the server every few seconds for remote diagnosis.

All levels are dBFS (0 = digital full scale, -90 = silence). Rough guide for a
speakerphone at a table: room noise -60..-45, speech -35..-15, clipping near 0.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

from kiki.audio import sources

_THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
_THROTTLED = Path("/sys/devices/platform/soc/soc:firmware/get_throttled")


def dbfs(energy: float) -> float:
    return 10 * math.log10(max(energy, 1e-9))


class EdgeStats:
    def __init__(self, vad_threshold: float) -> None:
        self.vad_threshold = vad_threshold
        self._reset()

    def _reset(self) -> None:
        self.t0 = time.monotonic()
        self.frame_db: list[float] = []
        self.speech_db: list[float] = []
        self.peak = 0
        self.clipped = 0
        self.muted = 0
        self.turn_ms: list[float] = []
        self.segments = 0
        self.barge_ins = 0
        self.echo_db: list[float] = []  # mic level while Kiki talks (echo residue + talk-over)
        self._dropped0 = sources.dropped_frames

    def frame(self, frame: np.ndarray, vad_p: float | None, muted: bool) -> None:
        if muted:
            self.muted += 1
            return
        x = frame.astype(np.float32) / 32768.0
        e = float(np.mean(x * x))
        db = dbfs(e)
        self.frame_db.append(db)
        if vad_p is not None and vad_p >= self.vad_threshold:
            self.speech_db.append(db)
        p = int(np.max(np.abs(frame)))
        self.peak = max(self.peak, p)
        if p >= 32000:
            self.clipped += 1

    def due(self, every_s: float = 5.0) -> bool:
        return time.monotonic() - self.t0 >= every_s

    def snapshot(self) -> dict:
        n = len(self.frame_db)
        fd = np.array(self.frame_db) if n else np.array([-90.0])
        snap = {
            "window_s": round(time.monotonic() - self.t0, 1),
            "frames": n,
            "level_dbfs": round(float(np.mean(fd)), 1),
            "noise_floor_dbfs": round(float(np.percentile(fd, 10)), 1),
            "loud_dbfs": round(float(np.percentile(fd, 95)), 1),
            "peak_dbfs": round(20 * math.log10(max(self.peak, 1) / 32768), 1),
            "speech_frac": round(len(self.speech_db) / n, 3) if n else 0.0,
            "speech_dbfs": round(float(np.mean(self.speech_db)), 1) if self.speech_db else None,
            "clipped_frames": self.clipped,
            "muted_frames": self.muted,  # half-duplex: mic ignored while Kiki talks
            "dropped_frames": sources.dropped_frames - self._dropped0,
            "segments": self.segments,
            "barge_ins": self.barge_ins,
            # while Kiki talks: typical and loud mic level (tune barge-in min_dbfs above echo_p90)
            "echo_dbfs_median": round(float(np.median(self.echo_db)), 1) if self.echo_db else None,
            "echo_dbfs_p90": round(float(np.percentile(self.echo_db, 90)), 1) if self.echo_db else None,
            "smart_turn_ms_max": round(max(self.turn_ms), 1) if self.turn_ms else None,
            "load_avg_1m": round(os.getloadavg()[0], 2),
            "cpu_temp_c": _read_temp(),
            "throttled": _read_throttled(),
        }
        self._reset()
        return snap


def _read_temp() -> float | None:
    try:
        return round(int(_THERMAL.read_text()) / 1000, 1)
    except (OSError, ValueError):
        return None


def _read_throttled() -> str | None:
    try:
        return _THROTTLED.read_text().strip()
    except OSError:
        return None
