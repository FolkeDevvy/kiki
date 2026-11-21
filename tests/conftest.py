import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kiki.audio.sources import read_wav

MODELS = Path(os.environ.get("KIKI_MODELS_DIR", "models"))


@pytest.fixture(scope="session")
def models_dir() -> Path:
    if not (MODELS / "silero_vad.onnx").exists():
        pytest.skip("models missing: run `kiki fetch-models`")
    return MODELS


@pytest.fixture(scope="session")
def say(tmp_path_factory):
    """Synthesize speech with espeak-ng -> int16 16 kHz array."""
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not installed")
    d = tmp_path_factory.mktemp("tts")

    def _say(text: str, name: str = "x"):
        p = d / f"{name}.wav"
        subprocess.run(["espeak-ng", "-s", "150", "-w", str(p), text], check=True)
        return read_wav(p)

    return _say
