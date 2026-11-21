"""Plays Kiki's voice on the edge device."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np


class Player:
    """Plays int16 PCM chunks in order on a background thread.

    `busy_until` (monotonic seconds) tells the edge when Kiki's audio will be done,
    so it can ignore the mic meanwhile (half-duplex until we have echo cancellation).
    """

    def __init__(self, enabled: bool = True, save_dir: Path | None = None, device=None) -> None:
        self.enabled = enabled
        self.save_dir = save_dir
        self.device = device
        self.busy_until = 0.0
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._saved: dict[str, list[np.ndarray]] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def start(self, say_id: str, sample_rate: int) -> None:
        self._q.put(("start", say_id, sample_rate))
        if self.save_dir:
            self._saved[say_id] = []

    def chunk(self, say_id: str, pcm: bytes, sample_rate: int) -> None:
        audio = np.frombuffer(pcm, dtype=np.int16)
        if self.enabled:
            now = time.monotonic()
            self.busy_until = max(self.busy_until, now) + len(audio) / sample_rate
        if self.save_dir and say_id in self._saved:
            self._saved[say_id].append(audio)
        self._q.put(("audio", audio, sample_rate))

    def end(self, say_id: str, sample_rate: int) -> Path | None:
        if not self.save_dir or say_id not in self._saved:
            return None
        from kiki.audio.sources import write_wav

        audio = np.concatenate(self._saved.pop(say_id) or [np.zeros(0, np.int16)])
        p = self.save_dir / f"kiki-{say_id}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        # write_wav writes 16 kHz; resample so the file plays at the right speed
        if sample_rate != 16000 and len(audio):
            n = int(len(audio) * 16000 / sample_rate)
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.int16)
        write_wav(p, audio)
        return p

    def stop(self) -> None:
        """Drop everything queued (barge-in / server 'stop')."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.busy_until = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        self._q.put(None)

    def _run(self) -> None:
        stream = None
        rate = None
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                break
            if item[0] != "audio" or not self.enabled:
                continue
            _, audio, sr = item
            try:
                if stream is None or rate != sr:
                    import sounddevice as sd

                    if stream is not None:
                        stream.close()
                    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16", device=self.device)
                    stream.start()
                    rate = sr
                stream.write(audio.reshape(-1, 1))
            except Exception as e:  # no audio device etc.; keep the edge alive
                print(f"[player] playback failed: {e!r}", flush=True)
                self.enabled = False
        if stream is not None:
            stream.close()
