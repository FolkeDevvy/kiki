"""Local English ASR: NVIDIA Parakeet TDT 0.6B v2 via sherpa-onnx."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sherpa_onnx

from kiki.config import SAMPLE_RATE

MODEL_DIRS = {
    "v2": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",  # English only
    "v3": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",  # 25 European languages incl. Swedish
}


class Parakeet:
    def __init__(self, models_dir: Path, num_threads: int = 4, version: str = "v3") -> None:
        d = models_dir / MODEL_DIRS[version]
        if not d.exists():
            raise FileNotFoundError(f"{d} missing; run `kiki fetch-models --server`")
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

    def decode(self, audio: np.ndarray) -> tuple[str, float]:
        """(text, mean token log-probability). Low confidence on the English-only
        model is a strong hint that the speech wasn't English."""
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x /= 32768.0
        s = self._rec.create_stream()
        s.accept_waveform(SAMPLE_RATE, x)
        self._rec.decode_streams([s])
        lp = list(s.result.ys_log_probs or [])
        return s.result.text.strip(), float(np.mean(lp)) if lp else -9.0

    def __call__(self, audio: np.ndarray) -> str:
        return self.decode(audio)[0]


class KBWhisper:
    """KBLab's Swedish Whisper (CTranslate2 via faster-whisper). Swedish only:
    it can't do English, so it's used when the English model isn't confident."""

    def __init__(self, models_dir: Path, size: str = "small", num_threads: int = 4) -> None:
        from faster_whisper import WhisperModel

        self._m = WhisperModel(f"KBLab/kb-whisper-{size}", device="cpu", compute_type="int8",
                               cpu_threads=num_threads, download_root=str(models_dir / "hf"))

    def __call__(self, audio: np.ndarray) -> str:
        x = audio.astype(np.float32)
        if audio.dtype == np.int16:
            x /= 32768.0
        segs, _ = self._m.transcribe(x, language="sv", beam_size=1, condition_on_previous_text=False,
                                     without_timestamps=True, vad_filter=False)
        return " ".join(sg.text.strip() for sg in segs).strip()


class Bilingual:
    """English first with Parakeet v2 (fast, best English); if it isn't confident,
    the speech is probably Swedish, so use KB-Whisper. (Parakeet v3's own language
    detection kept writing short Swedish phrases as English, even translating them.)"""

    def __init__(self, models_dir: Path, en_min_conf: float = -0.30, kb_size: str = "base") -> None:
        self.en = Parakeet(models_dir, version="v2")
        self.sv = KBWhisper(models_dir, size=kb_size)
        self.en_min_conf = en_min_conf
        self.last: dict = {}

    def __call__(self, audio: np.ndarray) -> str:
        text, conf = self.en.decode(audio)
        if conf >= self.en_min_conf and text:
            self.last = {"lang": "en", "conf": round(conf, 2), "en_text": text}
            return text
        sv = self.sv(audio)
        self.last = {"lang": "sv", "conf": round(conf, 2), "en_text": text}
        return sv or text
