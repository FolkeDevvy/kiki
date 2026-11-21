"""Plays Kiki's voice on the edge device, and can be cut off mid-sentence."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

BLOCK_S = 0.05  # write in small blocks so a barge-in stops playback within ~50 ms


class Player:
    """Plays int16 PCM chunks in order on a background thread.

    `busy_until` (monotonic seconds) tells the edge when Kiki's audio will be done.
    `stop()` cuts playback off immediately and drops the rest of that reply.
    """

    def __init__(self, enabled: bool = True, save_dir: Path | None = None, device=None,
                 simulate: bool = False) -> None:
        self.enabled = enabled or simulate
        self.simulate = simulate  # tests/replays: take real time, but play nothing
        self.save_dir = save_dir
        self.device = device
        self.busy_until = 0.0
        self.current: str | None = None  # say id being played
        self._cancelled: set[str] = set()
        self._q: queue.Queue = queue.Queue()
        self._close = threading.Event()
        self._abort = threading.Event()
        self._saved: dict[str, list[np.ndarray]] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def playing(self) -> bool:
        return time.monotonic() < self.busy_until

    def start(self, say_id: str, sample_rate: int) -> None:
        self.current = say_id
        self._abort.clear()
        self._q.put(("start", say_id, sample_rate))
        if self.save_dir:
            self._saved[say_id] = []

    def chunk(self, say_id: str, pcm: bytes, sample_rate: int) -> None:
        if say_id in self._cancelled:
            return  # the rest of a reply that was interrupted
        audio = np.frombuffer(pcm, dtype=np.int16)
        if self.enabled:
            now = time.monotonic()
            self.busy_until = max(self.busy_until, now) + len(audio) / sample_rate
        if self.save_dir and say_id in self._saved:
            self._saved[say_id].append(audio)
        self._q.put(("audio", say_id, audio, sample_rate))

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
        """Cut playback off now and ignore the rest of the current reply."""
        if self.current:
            self._cancelled.add(self.current)
        self._abort.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self.busy_until = time.monotonic()

    def close(self) -> None:
        self._close.set()
        self._q.put(None)

    def _run(self) -> None:
        stream = None
        rate = None
        while not self._close.is_set():
            item = self._q.get()
            if item is None:
                break
            if item[0] != "audio" or not self.enabled:
                continue
            _, say_id, audio, sr = item
            if say_id in self._cancelled:
                continue
            if self.simulate:
                block = max(1, int(sr * BLOCK_S))
                for i in range(0, len(audio), block):
                    if self._abort.is_set():
                        break
                    time.sleep(min(block, len(audio) - i) / sr)
                continue
            try:
                if stream is None or rate != sr:
                    import sounddevice as sd

                    if stream is not None:
                        stream.close()
                    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16", device=self.device)
                    stream.start()
                    rate = sr
                block = max(1, int(sr * BLOCK_S))
                for i in range(0, len(audio), block):
                    if self._abort.is_set():
                        break
                    stream.write(audio[i : i + block].reshape(-1, 1))
                    # write() returns once queued; the device latency is still to play
                    if not self._abort.is_set():
                        self.busy_until = max(self.busy_until, time.monotonic() + float(stream.latency))
                if self._abort.is_set():
                    stream.abort()  # drop what's buffered in the device, too
                    stream.start()
            except Exception as e:  # no audio device etc.; keep the edge alive
                print(f"[player] playback failed: {e!r}", flush=True)
                self.enabled = False
        if stream is not None:
            stream.close()
