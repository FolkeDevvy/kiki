"""Whisper log-mel features in numpy.

Reimplements `transformers.WhisperFeatureExtractor(chunk_length=8)` with
`do_normalize=True` so we don't ship transformers to the Pi just for this.
`tests/test_features.py` checks parity against the real extractor.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

N_FFT = 400
HOP = 160
N_MELS = 80
SR = 16_000


def _hz_to_mel(f: np.ndarray) -> np.ndarray:  # Slaney scale
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


@lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    """(n_freq=201, n_mels=80) Slaney-normalized triangular filters."""
    n_freq = N_FFT // 2 + 1
    fft_freqs = np.linspace(0, SR // 2, n_freq)
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(SR / 2), N_MELS + 2)
    filt_freqs = _mel_to_hz(mel_pts)
    diff = np.diff(filt_freqs)
    slopes = filt_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0, np.minimum(down, up))
    enorm = 2.0 / (filt_freqs[2 : N_MELS + 2] - filt_freqs[:N_MELS])
    return fb * enorm[None, :]


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    n = np.arange(N_FFT)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)  # periodic Hann


def log_mel(audio: np.ndarray) -> np.ndarray:
    """float32 audio (already the exact length wanted) -> (80, n_frames) features."""
    x = audio.astype(np.float64)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # do_normalize
    x = np.pad(x, N_FFT // 2, mode="reflect")  # center=True
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * _window()[None, :]
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2  # (frames, 201)
    mel = mel_filters().T @ power.T  # (80, frames)
    log_spec = np.log10(np.maximum(mel, 1e-10))[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)
