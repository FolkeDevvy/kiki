"""Speak with Kokoro, transcribe with Parakeet: words must survive the round trip."""

import pytest

from kiki.server.tts import resample

pytest.importorskip("sherpa_onnx")


@pytest.fixture(scope="module")
def engines(models_dir):
    if not (models_dir / "kokoro-multi-lang-v1_0").exists():
        pytest.skip("server models missing: run `kiki fetch-models --server`")
    from kiki.server.stt import Parakeet
    from kiki.server.tts import Kokoro

    return Kokoro(models_dir), Parakeet(models_dir)


def test_oov_words_are_spoken(engines):
    tts, stt = engines
    text = "Kiki, do you know if Thai restaurants usually take reservations?"
    audio = resample(tts(text), tts.sample_rate, 16000)
    heard = stt(audio).lower()
    for w in ["restaurants", "reservations"]:
        assert w in heard, heard
    assert "kiki" in heard or "kiki" in heard.replace(" ", "") or "kicky" in heard, heard


def test_phonemizer_falls_back_for_unknown_words(engines):
    tts, _ = engines
    ph = tts.phonemize("Kiki zorbulated.")
    assert len(ph.split()) == 2 and ph.endswith(".")
