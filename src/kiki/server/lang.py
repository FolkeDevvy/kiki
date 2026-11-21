"""Tiny Swedish/English language guess for short spoken lines.

Good enough to pick a TTS voice: Swedish letters (å, ä, ö) and function words
are very distinctive. Returns "sv" or "en".
"""

from __future__ import annotations

import re

_SV = set("""och att det som en ett är jag du vi ni de han hon inte på i för med till av om
har var vad hur varför när kan ska vill måste kommer finns också bara så men eller
den detta dessa här där min din vår er sin mycket lite nu då sen ja nej tack hej
the""".split()) - {"the"}
_EN = set("""the and that this is are was were i you we they he she it not on in for with
to of if have has what how why when can will would should there here my your our
just so but or very now then yes no thanks hello a an be do does""".split())
_WORD = re.compile(r"[a-zåäöéü']+", re.IGNORECASE)


def detect(text: str, default: str = "en") -> str:
    words = _WORD.findall(text.lower())
    if not words:
        return default
    sv = sum(w in _SV for w in words) + 2 * sum(any(c in w for c in "åäö") for w in words)
    en = sum(w in _EN for w in words)
    if sv == en:
        return default
    return "sv" if sv > en else "en"
