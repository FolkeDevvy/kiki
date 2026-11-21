"""Deciding when Kiki talks, and what it says.

Floor policy per finished turn:
  1. The gate (local LLM, one-word verdict + probabilities) judges the newest line.
  2. Code checks the verdict against things it can verify:
     - "reply" without Kiki's name only counts if Kiki spoke in the last few lines
       and the gate is confident;
     - "chime" needs proactivity on, enough probability, a cooldown, no quiet mode,
       and then the responder itself may still decline with PASS;
     - "quiet" mutes chime-ins and follow-ups for a while (the name still works).
  3. If the gate is unavailable, fall back to: reply only when named.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from kiki.server.gate import Gate, Verdict
from kiki.server.lang import detect
from kiki.server.llm import LLM

# STT spellings of "Kiki" seen in practice (Parakeet mostly gets it right in English)
_NAME_EXACT = re.compile(r"\b(kiki|kikki|kiky|keeki|kikí|kicky|kiki's)\b", re.IGNORECASE)
_WORDS = re.compile(r"[a-zåäöéí']+", re.IGNORECASE)


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def named(text: str) -> bool:
    """Kiki's name, spelled the way STT usually writes it."""
    return bool(_NAME_EXACT.search(text))


def maybe_named(text: str) -> bool:
    """A near-miss of the name at the start of the line ("Kiva", "Kikes", "Kik i",
    "Klick i" from Swedish speech). Weak evidence: also matches "Kids...", so the
    policy only trusts it together with a confident gate."""
    if named(text):
        return True
    w = [x.lower() for x in _WORDS.findall(text)[:2]]
    candidates = w[:1] + (["".join(w)] if len(w) == 2 and len(w[1]) <= 2 else [])
    return any(c.startswith("k") and 3 <= len(c) <= 6 and _edit_distance(c, "kiki") <= 2 for c in candidates)


class _NameMatcher:
    """Keeps `NAME.search(text)` working for existing callers (exact names only)."""

    def search(self, text: str) -> bool:
        return named(text)


NAME = _NameMatcher()

PERSONA = """You are Kiki, a small voice assistant sitting on a table among a group of people. \
You hear the conversation through a microphone; the transcript shows who said what as \
[Name]: text. Your own past lines are marked [Kiki].

How you talk:
- You are spoken aloud by a text-to-speech voice. Write plain spoken English: no markdown, \
no lists, no emoji, no stage directions, never prefix your reply with your name.
- The people speak Swedish and English, sometimes mixed. Answer in the language the person \
spoke to you in: Swedish (natural, everyday Swedish) if they spoke Swedish, English if they \
spoke English. Never mix languages within a sentence.
- Write numbers as digits ("180", "1989"), not words.
- Be brief: usually one short sentence, at most two. People are mid-conversation; don't lecture.
- Be warm, a little playful, and direct. Answer the question first.
- Address people by name when you know it. Labels like "Guest-2" are not names: never say \
them; just say "you" instead.
- If you didn't catch something or it's ambiguous, ask a short question.
- You can look things up on the internet, but it takes a little while, like a person \
checking their phone. When a good answer needs current or specific information you don't \
reliably know (news, weather, opening hours, prices, a specific place, business, product or \
person, sports results, recent events), start your reply with [LOOKUP: short search query] \
and then say ONE short natural sentence that you'll check, for example "Good question, let \
me check." or "Bra fråga, jag kollar!". Don't guess the answer; it will come later. Never \
look up things you know well (general knowledge, math, definitions, opinions, chit-chat). \
For the search query: the transcript comes from speech recognition and often misspells \
names, so write the name you think they meant (a company, product or place that fits). Write \
the query in the language that finds the best results: English for international things \
like stocks, tech or world news, Swedish for Swedish places, shops and events.

You will get the conversation transcript. Reply as Kiki to the last line, with only the \
words Kiki says."""

CHIME_SUFFIX = """

(Use the language they are speaking. Never use LOOKUP here. Nobody asked you. Answer exactly PASS if: someone already answered, the question is about \
their own lives (appointments, plans, belongings, people they know), or you aren't sure of \
the fact. Otherwise, if a well-known fact clearly settles what they're stuck on, say it in \
one short, friendly sentence.)"""

# proactivity level -> (min P(chime), cooldown seconds between unprompted remarks)
PROACTIVITY = {0: (2.0, 0.0), 1: (0.7, 120.0), 2: (0.5, 45.0)}

ENGAGED_LINES = 3  # Kiki must have spoken within this many lines for an unnamed reply
UNNAMED_REPLY_P = 0.85
QUIET_S = 180.0


@dataclass
class Line:
    speaker: str
    text: str
    t: float  # wall clock (time.time())


@dataclass
class Decision:
    action: str  # reply | chime | silent | quiet
    reason: str
    target: str | None = None
    verdict: Verdict | None = None
    named: bool = False

    @property
    def respond(self) -> bool:
        return self.action in ("reply", "chime")


@dataclass
class Brain:
    llm: LLM
    gate: Gate | None
    engagement_window_s: float = 30.0
    history_turns: int = 40
    proactivity: int = 1
    history: list[Line] = field(default_factory=list)
    quiet_until: float = 0.0
    last_chime_t: float = 0.0
    last_kiki_t: float = 0.0
    last_target: str | None = None  # who Kiki last answered
    looking_up: list[str] = field(default_factory=list)  # queries in progress (for the prompt)
    _maybe_named: bool = False  # set per decision: near-miss of the name at line start

    # ---- transcript ----
    def hear(self, speaker: str, text: str) -> None:
        self.history.append(Line(speaker, text, time.time()))

    def said(self, text: str, chime: bool = False, target: str | None = None) -> None:
        now = time.time()
        self.history.append(Line("Kiki", text, now))
        self.last_kiki_t = now
        self.last_target = target
        if chime:
            self.last_chime_t = now

    def transcript(self) -> str:
        recent = self.history[-self.history_turns :]
        return "\n".join(f"[{l.speaker}]: {l.text}" for l in recent)

    def messages(self, chime: bool = False) -> list[dict]:
        # Append-only on purpose: each new line only extends the prompt, so
        # llama.cpp's KV cache (kept warm by prewarm) covers everything before it.
        # The language hint goes last: small models otherwise follow the
        # conversation's majority language instead of the person they answer.
        heard = [l.text for l in self.history if l.speaker != "Kiki"]
        hint = "\n(Svara på svenska.)" if heard and detect(heard[-1]) == "sv" else "\n(Reply in English.)"
        state = f"\n(Today is {time.strftime('%A %-d %B %Y')}."
        if self.looking_up:
            state += " You're still looking up: " + "; ".join(self.looking_up) + \
                ". If asked about that, say you're still checking."
        content = self.transcript() + (CHIME_SUFFIX if chime else "") + state + ")" + hint
        return [{"role": "system", "content": PERSONA}, {"role": "user", "content": content}]

    # ---- floor policy ----
    @property
    def quiet(self) -> bool:
        return time.time() < self.quiet_until

    def engaged(self) -> bool:
        if time.time() - self.last_kiki_t > self.engagement_window_s:
            return False
        return any(l.speaker == "Kiki" for l in self.history[-(ENGAGED_LINES + 1) : -1])

    def likely_reply(self, text: str) -> bool:
        """Cheap pre-check used to start the responder speculatively."""
        return maybe_named(text) or (self.engaged() and not self.quiet)

    async def decide(self, speaker: str, text: str) -> Decision:
        self._maybe_named = maybe_named(text)
        named = NAME.search(text)
        if self.gate is None:
            return Decision("reply" if named else "silent", "no gate: name rule", speaker, named=named)
        try:
            # No quiet-mode hint for the gate: it made the small model answer "quiet"
            # to everything. The policy below enforces quiet mode itself.
            v = await self.gate.judge(self.transcript())
        except Exception as e:
            print(f"[brain] gate failed ({e!r}); falling back to the name rule", flush=True)
            return Decision("reply" if named else "silent", "gate failed: name rule", speaker, named=named)
        return self.apply(v, speaker, named)

    def apply(self, v: Verdict, speaker: str, named: bool) -> Decision:
        """Pure policy: gate verdict + verifiable context -> decision."""
        min_p, cooldown = PROACTIVITY.get(self.proactivity, PROACTIVITY[1])
        now = time.time()

        if v.action == "quiet":
            if not (named or self.engaged()):
                return Decision("silent", "'quiet' but not said to Kiki", speaker, v, named)
            self.quiet_until = now + QUIET_S
            return Decision("quiet", "asked to be quiet", speaker, v, named)

        if v.action == "reply":
            if named:
                self.quiet_until = 0.0  # being addressed by name ends quiet mode
                return Decision("reply", "addressed by name", speaker, v, named)
            if getattr(self, "_maybe_named", False) and v.p("reply") >= UNNAMED_REPLY_P:
                self.quiet_until = 0.0
                return Decision("reply", f"name probably mis-heard (p={v.p('reply'):.2f})", speaker, v, True)
            if self.quiet:
                return Decision("silent", "quiet mode", speaker, v, named)
            if self.last_target is not None and speaker != self.last_target:
                return Decision("silent", f"unnamed, and {speaker} isn't who Kiki answered", speaker, v, named)
            if self.engaged() and v.p("reply") >= UNNAMED_REPLY_P:
                return Decision("reply", f"follow-up (p={v.p('reply'):.2f})", speaker, v, named)
            return Decision("silent", f"unnamed reply not trusted (p={v.p('reply'):.2f})", speaker, v, named)

        if v.action == "chime":
            if self.quiet:
                return Decision("silent", "quiet mode", speaker, v, named)
            if v.p("chime") < min_p:
                return Decision("silent", f"chime too unsure (p={v.p('chime'):.2f})", speaker, v, named)
            if now - self.last_chime_t < cooldown:
                return Decision("silent", "chime cooldown", speaker, v, named)
            return Decision("chime", f"chime (p={v.p('chime'):.2f})", None, v, named)

        return Decision("silent", "gate: silent", speaker, v, named)


def is_pass(text: str) -> bool:
    """The responder declined a chime-in."""
    return re.fullmatch(r"\W*pass\W*", text, re.IGNORECASE) is not None
