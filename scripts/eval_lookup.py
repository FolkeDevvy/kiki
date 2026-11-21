"""Does the responder ask for a lookup only when it should?

    docker compose run --rm --entrypoint python server scripts/eval_lookup.py
"""

import asyncio
import os

from kiki.server.brain import Brain
from kiki.server.llm import LLM
from kiki.server.research import LOOKUP_MARK

CASES = [  # (line, should look up?)
    ("[Anna]: Kiki, what's the weather going to be like in Stockholm tomorrow?", True),
    ("[Sara]: Kiki, how much is a ticket to Liseberg this year?", True),
    ("[Anna]: Kiki, when does the ICA Maxi in Lund close on Sundays?", True),
    ("[Sara]: Kiki, who won the last Eurovision?", True),
    ("[Anna]: Kiki, what's the latest news about the Artemis moon mission?", True),
    ("[Anna]: Kiki, what's the capital of Norway?", False),
    ("[Sara]: Kiki, what's twelve times fifteen?", False),
    ("[Anna]: Kiki, tell us a joke.", False),
    ("[Sara]: Kiki, what does the word 'serendipity' mean?", False),
    ("[Anna]: Kiki, how are you today?", False),
]


async def main() -> None:
    llm = LLM(os.environ.get("KIKI_LLM_URL", "http://localhost:8081"), timeout=120)
    ok = 0
    for line, want in CASES:
        b = Brain(llm=llm, gate=None)
        who, text = line[1:].split("]: ", 1)
        b.hear(who, text)
        out = "".join([d async for d in llm.stream(b.messages(), max_tokens=60, slot=0)]).strip()
        m = LOOKUP_MARK.match(out)
        got = m is not None
        ok += got == want
        print(f"{'✓' if got == want else '✗'} lookup={got!s:5} want={want!s:5} {text[:55]:55s} -> {out[:90]}")
    print(f"\n{ok}/{len(CASES)} correct")
    await llm.aclose()


asyncio.run(main())
