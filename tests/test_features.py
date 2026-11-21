import numpy as np
import pytest

from kiki.audio.features import log_mel
from kiki.audio.turn import fit_to_window


def test_matches_transformers_whisper_extractor():
    transformers = pytest.importorskip("transformers")
    fe = transformers.WhisperFeatureExtractor(chunk_length=8)
    rng = np.random.default_rng(1)
    t = np.arange(16000 * 3) / 16000
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    audio = fit_to_window(audio)

    ref = fe(
        audio, sampling_rate=16000, return_tensors="np", padding="max_length",
        max_length=8 * 16000, truncation=True, do_normalize=True,
    ).input_features[0]
    ours = log_mel(audio)
    assert ours.shape == ref.shape == (80, 800)
    np.testing.assert_allclose(ours, ref, atol=2e-3)
