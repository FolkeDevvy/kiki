"""The gate: should Kiki speak after this turn?

A local LLM reads the recent transcript and answers with ONE word, constrained by
a grammar so it costs ~1 generated token. The first token's log-probabilities
give a distribution over the actions, so proactive "chime" can require real
confidence instead of just winning the argmax.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import httpx

ACTIONS = ("reply", "chime", "silent", "quiet")

GATE_PROMPT = """You are the turn-taking judge for Kiki, a voice assistant sitting on a table \
while a group of people talk. You never talk yourself. After each new line of the transcript \
you decide what Kiki should do, answering with exactly one word:

reply  - the newest line is meant for Kiki: it addresses Kiki by name, or it's clearly a \
follow-up to something Kiki just said to that person (an answer to Kiki's question, "and what \
about...", "thanks, one more thing").
chime  - nobody addressed Kiki, but the people are stuck on a question of GENERAL KNOWLEDGE \
that Kiki can answer from what it knows: a factual question nobody can answer, someone trying \
to remember a well-known fact, a clear factual mistake. Rare. Never about the people's own \
lives (schedules, plans, where things are, what someone did, who said what), never for \
opinions, chit-chat, or things the people already answered themselves.
silent - everything else. People talking to each other, statements, stories, jokes, small \
talk, someone talking ABOUT Kiki rather than TO Kiki, the name "Kiki" used for something \
else, and any line Kiki was told not to respond to.
quiet  - someone asks Kiki to stop talking or be quiet for a while ("Kiki, shush", "Kiki, \
stop interrupting", "be quiet for a bit").

Rules that override everything:
- If someone told Kiki not to respond (to the next thing, to them, for a while), stay \
silent until they address Kiki by name again with a real request.
- Mentioning Kiki is not addressing Kiki. "Kiki is clever" or "I'll ask Kiki later" is silent.
- A question to a specific person ("Bob, did you...") is for that person: silent.
- After Kiki answered someone, a new topic or a line aimed at someone else is silent.
- When unsure, choose silent. Interrupting people is worse than missing a chance to talk.

People speak Swedish and English. Judge Swedish lines exactly like English ones: \
"Kiki, vad är..." addresses Kiki; "Kiki, var tyst" / "Kiki, sluta" asks for quiet; \
"svara inte på nästa" means don't respond. The transcript comes from speech recognition, \
so the name is sometimes mis-written ("Kiva", "Kik i", "Kicki", "Chiki"); at the start of a \
line that's still Kiki being addressed.

Examples (each ends at the line to judge):

[Anna]: Kiki, what's twelve times fifteen?
-> reply

[Kiki]: The capital of Sweden is Stockholm.
[Folke]: And Norway?
-> reply

[Folke]: Kiki, don't answer the next thing I say.
[Folke]: There are a lot of apples in the garden.
-> silent

[Kiki]: Stockholm is the capital.
[Folke]: Cool. Anna, are you coming tomorrow?
-> silent

[Anna]: Who was the actor in Titanic again?
[Bob]: Ugh, it's on the tip of my tongue.
-> chime

[Bob]: I think we should paint the kitchen blue.
-> silent

[Anna]: When is Bob's flight landing?
[Folke]: Around six I think.
-> silent

[Anna]: Kiki is honestly pretty smart.
-> silent

[Bob]: Kiki, could you stop interrupting us?
-> quiet

[Folke]: Kiki, hur många invånare har Göteborg?
-> reply

[Anna]: Har du köpt mjölk?
[Folke]: Nej, jag glömde.
-> silent

Transcript lines look like [Name]: text. Labels like Guest-1 are people whose names are \
unknown. [Kiki] lines are what Kiki said. A final "(state: ...)" line, if present, is \
context, not speech. Judge only the newest spoken line, using the whole conversation as \
context."""

GRAMMAR = "root ::= " + " | ".join(f'"{a}"' for a in ACTIONS)


@dataclass
class Verdict:
    action: str
    probs: dict[str, float] = field(default_factory=dict)
    ms: float = 0.0

    def p(self, action: str) -> float:
        return self.probs.get(action, 0.0)


class Gate:
    def __init__(self, base_url: str, slot: int | None = 1, timeout: float = 15.0) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self.slot = slot

    @staticmethod
    def messages(transcript: str, state: str = "") -> list[dict]:
        # state goes last so the transcript prefix stays cacheable
        content = transcript + (f"\n(state: {state})" if state else "")
        return [{"role": "system", "content": GATE_PROMPT}, {"role": "user", "content": content}]

    async def judge(self, transcript: str, state: str = "") -> Verdict:
        body = {
            "messages": self.messages(transcript, state),
            "max_tokens": 3,
            "temperature": 0.0,
            "grammar": GRAMMAR,
            "logprobs": True,
            "top_logprobs": 20,
            "cache_prompt": True,
        }
        if self.slot is not None:
            body["id_slot"] = self.slot
        t = time.perf_counter()
        r = await self._http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        action = choice["message"]["content"].strip()
        probs = _first_token_probs(choice.get("logprobs"))
        if action not in ACTIONS:
            action = "silent"
        return Verdict(action, probs, (time.perf_counter() - t) * 1000)

    async def aclose(self) -> None:
        await self._http.aclose()


def _first_token_probs(logprobs: dict | None) -> dict[str, float]:
    """Distribution over ACTIONS from the first token's top logprobs (raw model
    distribution, renormalized over our labels; labels are matched by prefix
    because a label may be split into several tokens)."""
    if not logprobs or not logprobs.get("content"):
        return {}
    probs: dict[str, float] = {}
    for alt in logprobs["content"][0].get("top_logprobs", []):
        tok = alt["token"].strip().lower()
        if not tok:
            continue
        for a in ACTIONS:
            if a.startswith(tok):
                probs[a] = probs.get(a, 0.0) + math.exp(alt["logprob"])
                break
    total = sum(probs.values())
    return {a: p / total for a, p in probs.items()} if total > 0 else {}
