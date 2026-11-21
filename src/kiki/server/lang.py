"""Tiny Swedish/English language guess for short spoken lines.

Good enough to pick a TTS voice: Swedish letters (å, ä, ö) and function words
are very distinctive. Returns "sv" or "en".
"""

from __future__ import annotations

import re

_SV = set("""och att det som en ett är jag du vi ni de dem han hon man inte på i för med till av om
har hade var vad vem vilken vilket vilka hur varför när var kan kunde ska skulle vill
måste kommer kom finns fanns också bara så men eller den detta dessa här där hit dit
min mitt mina din ditt dina vår vårt våra er ert era sin sitt sina hans hennes deras
mycket lite nu då sen sedan ja nej jo tack hej hejdå okej kanske alltid aldrig ofta
vann vinner blir blev bli göra gör gjorde säga säger sa tycker tror vet visste kolla
kostar kostade heter hette idag imorgon igår ikväll senaste nästa förra året år dag
många mer mest någon något några ingen inget inga alla allt annan andra också redan
skulle borde efter innan under över mellan utan genom mot igen bra dåligt stor liten""".split())
_EN = set("""the and that this is are was were i you we they he she it not on in for with
to of if have has had what who which how why when where can could will would should
there here my your our their his her just so but or very now then yes no thanks hello
a an be do does did won win get got make made say said think know cost costs today
tomorrow yesterday tonight last next year many more most some any all other already
after before about from like really good bad big small please sorry okay maybe""".split())
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
