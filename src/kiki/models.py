"""Download and locate the local models.

Edge (Pi) models are small ONNX files. Server models are larger and live on the
laptop / GPU box.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


@dataclass(frozen=True)
class Model:
    name: str  # file name, or directory name for archives
    url: str
    group: str  # edge | server
    archive: bool = False


MODELS = [
    # MIT
    Model("silero_vad.onnx", "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx", "edge"),
    # BSD-2. int8 build (default everywhere) and the fp32 build (published as "-gpu",
    # runs on CPU too; not fetched by default).
    Model("smart-turn-v3.2-cpu.onnx", "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx?download=true", "edge"),
    Model("smart-turn-v3.2-gpu.onnx", "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx?download=true", "extra"),
    # NVIDIA Parakeet TDT 0.6B v2 (English), int8, CC-BY-4.0
    Model("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8", f"{SHERPA}/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2", "server", archive=True),
    # WeSpeaker ResNet34 (VoxCeleb, large-margin fine-tuned)
    Model("wespeaker_en_voxceleb_resnet34_LM.onnx", f"{SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx", "server"),
    # Kokoro 82M, multi-lang v1.0 (has af_heart etc.), Apache-2.0
    Model("kokoro-multi-lang-v1_0", f"{SHERPA}/tts-models/kokoro-multi-lang-v1_0.tar.bz2", "server", archive=True),
]


def smart_turn_file(setting: str = "auto") -> str:
    """Which Smart Turn build to use. int8 everywhere: it ran fine on the Pi in
    practice and is the lighter load (an earlier "int8 is slow on ARM" theory
    was really the Pi browning out). The fp32 file stays available by name."""
    if setting != "auto":
        return setting
    return "smart-turn-v3.2-cpu.onnx"


def path(models_dir: Path, name: str) -> Path:
    p = models_dir / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `kiki fetch-models` first")
    return p


def fetch(models_dir: Path, groups: set[str], force: bool = False) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        if m.group not in groups:
            continue
        dest = models_dir / m.name
        if dest.exists() and not force:
            print(f"ok      {dest}")
            continue
        tmp = models_dir / (m.name + ".part")
        print(f"fetch   {m.url}", flush=True)
        urllib.request.urlretrieve(m.url, tmp)
        if m.archive:
            with tarfile.open(tmp, "r:*") as tar:
                tar.extractall(models_dir, filter="data")
            tmp.unlink()
        else:
            tmp.rename(dest)
        print(f"saved   {dest}", flush=True)
