"""Print Kiki's replies to Swedish prompts (human judgement of fluency/facts).

    docker compose run --rm --entrypoint python server scripts/eval_swedish.py
"""

import asyncio
import os
import time

from kiki.server.brain import Brain
from kiki.server.lang import detect
from kiki.server.llm import LLM

PROMPTS = [
    ["[Folke]: Kiki, vad är huvudstaden i Norge?"],
    ["[Anna]: Kiki, hur länge ska man koka ett ägg för att det ska bli löskokt?"],
    ["[Folke]: Kiki, berätta ett kort skämt."],
    ["[Anna]: Kiki, vad tycker du att vi ska äta till middag ikväll?"],
    ["[Folke]: Kiki, vilket år föll Berlinmuren?", "[Kiki]: 1989.", "[Folke]: Och när återförenades Tyskland?"],
    ["[Anna]: Kiki, what's the capital of Denmark?"],
    ["[Folke]: Kiki, kan du säga hej till Anna på engelska?"],
]


async def main() -> None:
    llm = LLM(os.environ.get("KIKI_LLM_URL", "http://localhost:8081"), timeout=120)
    for lines in PROMPTS:
        b = Brain(llm=llm, gate=None)
        for line in lines:
            who, text = line[1:].split("]: ", 1)
            (b.said(text) if who == "Kiki" else b.hear(who, text))
        t = time.perf_counter()
        out = "".join([d async for d in llm.stream(b.messages(), max_tokens=80, slot=0)]).strip()
        print(f"{lines[-1]}\n   -> [{detect(out)}] {out}   ({time.perf_counter() - t:.1f}s)\n")
    await llm.aclose()


asyncio.run(main())
