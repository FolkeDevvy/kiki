"""Compare Claude effort levels for Kiki's replies: time to first text, total time.

    docker compose run --rm --entrypoint python server scripts/bench_claude.py low medium
    docker compose run --rm --entrypoint python server scripts/bench_claude.py claude-haiku-5-5:low --rounds=1
"""

import asyncio
import statistics
import sys
import time

from kiki.server.brain import PERSONA_CLAUDE
from kiki.server.claude_brain import ClaudeResponder

TURNS = [
    "[Anna]: Kiki, what's the capital of Denmark?",
    "[Sara]: Kiki, how long should I boil an egg?",
    "[Sara]: Kick it, what year did the burlin wall fall?",
    "[Anna]: Kiki, what's twelve times fifteen?",
    "[Anna]: And thirteen times fifteen?",
    "[Sara]: Kiki, what do you think we should have for dinner?",
]


async def run(spec: str, rounds: int) -> None:
    model, _, effort = spec.rpartition(":") if ":" in spec else ("claude-opus-5-5", "", spec)
    firsts, totals = [], []
    for _ in range(rounds):
        r = ClaudeResponder(PERSONA_CLAUDE, model, effort)
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
            print(f"  {spec[:26]:26s} first {first or total:5.2f}s total {total:5.2f}s  {text.strip()[:70]}", flush=True)
        await r.reset()
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    print(f"{spec}: first text median {statistics.median(firsts):.2f}s, p90 {q(firsts, .9):.2f}s; "
          f"full reply median {statistics.median(totals):.2f}s\n", flush=True)


async def main() -> None:
    rounds = int(next((a[9:] for a in sys.argv if a.startswith("--rounds=")), "2"))
    for spec in [a for a in sys.argv[1:] if not a.startswith("--")] or ["low", "medium"]:
        await run(spec, rounds=rounds)


asyncio.run(main())
