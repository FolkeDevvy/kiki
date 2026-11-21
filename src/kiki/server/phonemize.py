"""Text -> Kokoro phonemes.

sherpa-onnx >= 2.0 dropped its built-in espeak fallback, so words missing from
Kokoro's lexicon were silently skipped ("Kiki", "restaurants"). We look words
up in the lexicon first (misaki-quality pronunciations) and fall back to
espeak-ng IPA, mapped onto Kokoro's symbol set, for everything else.
"""

from __future__ import annotations

import re
import subprocess
from functools import lru_cache
from pathlib import Path

# espeak IPA -> Kokoro/misaki symbols (diphthongs are single letters in Kokoro)
_IPA_MAP = [
    ("eɪ", "A"), ("aɪ", "I"), ("oʊ", "O"), ("aʊ", "W"), ("ɔɪ", "Y"),
    ("dʒ", "ʤ"), ("tʃ", "ʧ"), ("ɝ", "ɜɹ"), ("ɚ", "əɹ"), ("r", "ɹ"), ("g", "ɡ"),
    ("ɐ", "ə"), ("ᵻ", "ɪ"), ("ɬ", "l"),
]
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+|[.,!?;:—…\"()“”]")
_NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
              "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


class Phonemizer:
    def __init__(self, lexicon: Path, tokens: Path, voice: str = "en-us") -> None:
        self.voice = voice
        self.lex: dict[str, str] = {}
        for line in lexicon.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                self.lex.setdefault(parts[0].lower(), "".join(parts[1:]))
        self.symbols = {line.rsplit(" ", 1)[0] for line in tokens.read_text(encoding="utf-8").splitlines() if line.strip()}

    def word(self, w: str) -> str:
        key = w.lower()
        if key in self.lex:
            return self.lex[key]
        if key.endswith("'s") and key[:-2] in self.lex:
            return self.lex[key[:-2]] + "z"
        return self._espeak(key)

    @lru_cache(maxsize=4096)
    def _espeak(self, w: str) -> str:
        ipa = subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", self.voice, w],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        for a, b in _IPA_MAP:
            ipa = ipa.replace(a, b)
        return "".join(c for c in ipa if c in self.symbols)

    def __call__(self, text: str) -> str:
        out: list[str] = []
        for tok in _WORD.findall(text):
            if tok[0].isalpha():
                out.append(" " + self.word(tok))
            elif tok.isdigit():
                out.append(" " + self._espeak(_NUM_WORDS.get(tok, tok)))
            else:
                out.append(tok)
        return "".join(out).strip()
