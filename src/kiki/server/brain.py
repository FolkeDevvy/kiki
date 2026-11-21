"""Deciding when Kiki talks, and what it says.

Milestone 2: Kiki replies when its name is said, or when the person it just
talked to follows up within the engagement window. The LLM gate for proactive
remarks comes in milestone 4 and plugs into `Brain.decide`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from kiki.server.llm import LLM

# STT spellings of "Kiki" seen in practice (Parakeet mostly gets it right)
NAME = re.compile(r"\b(kiki|kikki|kiky|keeki|kikí|kicky|kiki's)\b", re.IGNORECASE)

PERSONA = """You are Kiki, a small voice assistant sitting on a table among a group of people. \
You hear the conversation through a microphone; the transcript below shows who said what as \
[Name]: text. Names like Guest-1 are people whose names you don't know yet. Your own past \
lines are marked [Kiki].

How you talk:
- You are spoken aloud by a text-to-speech voice. Write plain spoken English: no markdown, \
no lists, no emoji, no stage directions, never prefix your reply with your name.
- Be brief: usually one short sentence, at most two. People are mid-conversation; don't lecture.
- Be warm, a little playful, and direct. Answer the question first.
- Address people by name when you know it; never call anyone "Guest-1".
- If you didn't catch something or it's ambiguous, ask a short question.
- You have no internet, so you can't look up specifics like a particular restaurant's \
policy or today's opening hours; say that briefly if it matters, but still share what you \
generally know.

You will get the conversation transcript. Reply as Kiki to the last line, with only the \
words Kiki says."""


@dataclass
class Line:
    speaker: str
    text: str
    t: float  # wall clock (time.time())


@dataclass
class Decision:
    respond: bool
    reason: str
    target: str | None = None


class Brain:
    def __init__(self, llm: LLM, engagement_window_s: float, history_turns: int) -> None:
        self.llm = llm
        self.window = engagement_window_s
        self.history_turns = history_turns
        self.history: list[Line] = []
        self._last_kiki_t = 0.0
        self._last_kiki_target: str | None = None

    def hear(self, speaker: str, text: str) -> None:
        self.history.append(Line(speaker, text, time.time()))

    def said(self, text: str, target: str | None) -> None:
        self.history.append(Line("Kiki", text, time.time()))
        self._last_kiki_t = time.time()
        self._last_kiki_target = target

    def decide(self, speaker: str, text: str) -> Decision:
        if NAME.search(text):
            return Decision(True, "name", speaker)
        engaged = time.time() - self._last_kiki_t < self.window
        if engaged and speaker == self._last_kiki_target:
            return Decision(True, "engaged follow-up", speaker)
        return Decision(False, "not addressed")

    def messages(self) -> list[dict]:
        recent = self.history[-self.history_turns :]
        # Append-only on purpose: each new line only extends the prompt, so
        # llama.cpp's KV cache (kept warm by prewarm) covers everything before it.
        transcript = "\n".join(f"[{l.speaker}]: {l.text}" for l in recent)
        return [
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": transcript},
        ]
