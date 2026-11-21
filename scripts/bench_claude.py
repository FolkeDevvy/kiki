"""Compare Claude effort levels for Kiki's replies: time to first text, total time.

    docker compose run --rm --entrypoint python server scripts/bench_claude.py low medium
"""

import asyncio
import statistics
import sys
import time

from kiki.server.brain import PERSONA_CLAUDE
from kiki.server.claude_brain import ClaudeResponder

TURNS = [
    "[Anna]: Kiki, what's the capital of Denmark?\n(Reply in English.)",
    "[Sara]: Kiki, hur länge ska man koka ett ägg?\n(Svara på svenska.)",
    "[Sara]: Klick i vilket år följ billimulen.\n(Svara på svenska.)",
    "[Anna]: Kiki, what's twelve times fifteen?\n(Reply in English.)",
    "[Anna]: And thirteen times fifteen?\n(Reply in English.)",
    "[Sara]: Kiki, vad tycker du att vi ska äta ikväll?\n(Svara på svenska.)",
]


async def run(effort: str, rounds: int) -> None:
    firsts, totals = [], []
    for _ in range(rounds):
        r = ClaudeResponder(PERSONA_CLAUDE, "claude-opus-5-5", effort)
        await r.warm()
        for turn in TURNS:
            t = time.perf_counter()
            first, text = None, ""
            async for d in r.stream(turn):
                first = first or time.perf_counter() - t
                text += d
            total = time.perf_counter() - t
            firsts.append(first or total)
            totals.append(total)
            print(f"  {effort:6s} first {first or total:5.2f}s total {total:5.2f}s  {text.strip()[:70]}", flush=True)
        await r.reset()
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    print(f"{effort}: first text median {statistics.median(firsts):.2f}s, p90 {q(firsts, .9):.2f}s; "
          f"full reply median {statistics.median(totals):.2f}s\n", flush=True)


async def main() -> None:
    for effort in sys.argv[1:] or ["low", "medium"]:
        await run(effort, rounds=2)


asyncio.run(main())
