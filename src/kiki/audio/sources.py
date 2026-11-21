"""Audio sources yielding fixed-size int16 frames at 16 kHz mono."""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from kiki.config import FRAME_SAMPLES, SAMPLE_RATE


def read_wav(p: Path) -> np.ndarray:
    """Read a 16-bit PCM WAV as int16 mono at 16 kHz (downmixes; naive resample)."""
    with wave.open(str(p), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{p}: need 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        sr, ch = w.getframerate(), w.getnchannels()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != SAMPLE_RATE:
        # Linear interpolation is fine for VAD/turn tests; real input is 16 kHz already.
        n = int(len(audio) * SAMPLE_RATE / sr)
        x = np.linspace(0, len(audio) - 1, n)
        audio = np.interp(x, np.arange(len(audio)), audio).astype(np.int16)
    return audio


def write_wav(p: Path, audio: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.astype(np.int16).tobytes())


async def file_frames(p: Path, realtime: bool = True, tail_silence_s: float = 2.5) -> AsyncIterator[np.ndarray]:
    """Replay a WAV as frames. Appends silence so trailing turns can close."""
    audio = read_wav(p)
    audio = np.concatenate([audio, np.zeros(int(tail_silence_s * SAMPLE_RATE), np.int16)])
    n = len(audio) // FRAME_SAMPLES
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    for i in range(n):
        if realtime:
            delay = t0 + i * FRAME_SAMPLES / SAMPLE_RATE - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
        elif i % 32 == 0:
            await asyncio.sleep(0)  # let other tasks run
        yield audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]


async def mic_frames(device: str | int | None = None) -> AsyncIterator[np.ndarray]:
    """Live microphone frames via PortAudio."""
    import sounddevice as sd  # imported lazily: needs libportaudio at runtime

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)

    def callback(indata, frames, time_info, status):  # runs on PortAudio's thread
        if status:
            print(f"[audio] {status}")
        frame = indata[:, 0].copy()
        loop.call_soon_threadsafe(_put_dropping, q, frame)

    if isinstance(device, str) and device.isdigit():
        device = int(device)
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        device=device or None,
        callback=callback,
    ):
        while True:
            yield await q.get()


dropped_frames = 0  # mic frames dropped because processing fell behind (edge stats)


def _put_dropping(q: asyncio.Queue, item) -> None:
    global dropped_frames
    if q.full():
        q.get_nowait()  # we're falling behind; drop the oldest frame
        dropped_frames += 1
    q.put_nowait(item)
