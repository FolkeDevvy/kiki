"""Evaluate the turn-taking gate on hand-written multi-party scenarios.

    docker compose run --rm --entrypoint python server scripts/eval_gate.py

Each scenario is a transcript ending at the line to judge, plus the set of
acceptable actions. Scenarios are deliberately NOT copies of the prompt's
examples, so this measures generalization, not memorization.
"""

import asyncio
import os
import sys
import time

from kiki.server.brain import Brain, is_pass
from kiki.server.gate import Gate, Verdict
from kiki.server.llm import LLM

S = [
    # --- explicit don't-respond / quiet (the bug that started this) ---
    (["[Folke]: Kiki, don't respond to this next one.", "[Folke]: There's a lot of apples in the garden."], {"silent"}),
    (["[Folke]: Kiki, ignore what I say next.", "[Kiki]: Okay.", "[Folke]: My brother is visiting on Saturday."], {"silent"}),
    (["[Anna]: Kiki, shush for a while, we're talking.", ], {"quiet"}),
    (["[Bob]: Kiki please be quiet.", ], {"quiet"}),
    (["[Anna]: Kiki, be quiet for a while.", "[Kiki]: Sure.", "[Bob]: So the meeting moved to Tuesday.",
      "[Anna]: Kiki, okay you can talk again. What's the weather usually like in May in Rome?"], {"reply"}),
    # --- addressed by name ---
    (["[Anna]: Kiki, how long should I boil an egg?"], {"reply"}),
    (["[Guest-1]: Hey Kiki, what's the capital of Australia?"], {"reply"}),
    (["[Bob]: We were arguing about this. Kiki, is a tomato a fruit?"], {"reply"}),
    # --- engaged follow-ups ---
    (["[Folke]: Kiki, what's the capital of Sweden?", "[Kiki]: Stockholm.", "[Folke]: And Finland?"], {"reply"}),
    (["[Anna]: Kiki, any idea for dinner?", "[Kiki]: How about a quick pasta? Do you have tomatoes?", "[Anna]: Yeah, we have some."], {"reply"}),
    (["[Folke]: Kiki, what's the capital of Sweden?", "[Kiki]: Stockholm.", "[Folke]: Thanks."], {"silent", "reply"}),
    # --- engaged, but the conversation moved on ---
    (["[Folke]: Kiki, what's the capital of Sweden?", "[Kiki]: Stockholm.", "[Folke]: Nice. Anyway Anna, did you book the train?"], {"silent"}),
    (["[Anna]: Kiki, set the mood, tell a joke.", "[Kiki]: Why did the tomato blush? It saw the salad dressing.", "[Bob]: Haha. Anyway, who's driving tomorrow?"], {"silent"}),
    (["[Folke]: Kiki, do you hear me?", "[Kiki]: Yep, loud and clear!", "[Folke]: There's a lot of apples in the garden this year."], {"silent", "chime"}),
    # --- people talking to each other ---
    (["[Anna]: Did you feed the cat?", "[Bob]: Yes, this morning."], {"silent"}),
    (["[Bob]: I think the new season is better than the old one.", "[Anna]: No way, the first season was the best."], {"silent"}),
    (["[Anna]: Bob, can you pass the salt?"], {"silent"}),
    (["[Guest-1]: I'm so tired today.", "[Guest-2]: Same, I barely slept."], {"silent"}),
    # --- mentions, not addresses ---
    (["[Anna]: I asked Kiki about it yesterday and it was useful."], {"silent"}),
    (["[Bob]: My neighbor's dog is also called Kiki."], {"silent"}),
    (["[Folke]: Kiki is new, I built it this week."], {"silent"}),
    # --- proactive chances ---
    (["[Anna]: What's the tallest mountain in Africa again?", "[Bob]: No idea. Something with K?"], {"chime"}),
    (["[Bob]: How many tablespoons are in a cup?", "[Anna]: Hmm, I never remember that."], {"chime"}),
    (["[Anna]: The Eiffel Tower is in London, right?", "[Bob]: Yeah I think so."], {"chime"}),
    # --- not chime-worthy ---
    (["[Anna]: Should we get pizza or sushi tonight?", "[Bob]: Pizza, obviously."], {"silent"}),
    (["[Bob]: What time is the dentist tomorrow?", "[Anna]: Ten, I think."], {"silent"}),
    (["[Anna]: Where did I put my keys?", "[Bob]: No clue."], {"silent"}),
    (["[Bob]: Who called you earlier?", "[Anna]: Hmm, I can't remember."], {"silent"}),
    (["[Anna]: How far is it from Stockholm to Gothenburg?", "[Bob]: I don't know, far."], {"chime"}),
    (["[Bob]: What year did the Berlin wall fall?", "[Anna]: Eighty-something?"], {"chime"}),
]


def brain_for(lines: list[str], gate: Gate, llm: LLM) -> Brain:
    b = Brain(llm=llm, gate=gate)
    for line in lines:
        who, text = line[1:].split("]: ", 1)
        if who == "Kiki":
            b.said(text)
        else:
            b.hear(who, text)
    return b


async def final_action(b: Brain, lines: list[str]) -> tuple[str, str, Verdict | None, str]:
    """Full policy: gate + checks + (for chime) the responder's PASS."""
    who, text = lines[-1][1:].split("]: ", 1)
    d = await b.decide(who, text)
    said = ""
    if d.action == "chime":
        said = "".join([t async for t in b.llm.stream(b.messages(chime=True), max_tokens=60, slot=0)])
        if is_pass(said):
            return "silent", "responder: PASS", d.verdict, said
    return d.action, d.reason, d.verdict, said


async def main() -> None:
    url = os.environ.get("KIKI_LLM_URL", "http://localhost:8081")
    gate = Gate(url, timeout=120)
    llm = LLM(url, timeout=120)
    await gate.judge("[Anna]: hello")  # warm the cached system prompt
    ok, times, fails = 0, [], []
    for lines, want in S:
        b = brain_for(lines, gate, llm)
        t = time.perf_counter()
        action, reason, v, said = await final_action(b, lines)
        times.append((time.perf_counter() - t) * 1000)
        good = action in want
        ok += good
        raw = f"{v.action}:{max(v.probs.values(), default=0):.2f}" if v else "-"
        extra = f'  -> "{said.strip()[:50]}"' if said and not is_pass(said) else ""
        print(f"{'✓' if good else '✗'} {action:6s} want {'/'.join(sorted(want)):13s} gate {raw:12s} "
              f"{reason[:34]:34s} {lines[-1][:55]}{extra}")
        if not good:
            fails.append(lines[-1])
    times.sort()
    print(f"\n{ok}/{len(S)} correct; decision latency median {times[len(times)//2]:.0f} ms, max {times[-1]:.0f} ms")
    await gate.aclose()
    await llm.aclose()
    sys.exit(0 if ok == len(S) else 1)


if __name__ == "__main__":
    asyncio.run(main())
