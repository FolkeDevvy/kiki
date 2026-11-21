"""_speak end to end with fakes: audio goes out AND the reply is recorded."""

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from kiki.server.app import Session
from kiki.server.brain import Brain, Decision
from kiki.server.diagnostics import Diagnostics


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, m):
        self.sent.append(m)


class FakeClaude:
    def __init__(self, reply):
        self.reply = reply
        self.last_usage = {}

    async def stream(self, message):
        for w in self.reply.split(" "):
            yield w + " "


def make(reply):
    s = Session.__new__(Session)
    s.m = SimpleNamespace(tts=_Voice(), tts_pool=ThreadPoolExecutor(1), research=None)
    s.ws = FakeWS()
    s.brain = Brain(llm=None, gate=None)
    s.brain.hear("Anna", "Kiki, what's the capital of Denmark?")
    s.claude = FakeClaude(reply)
    s.diag = Diagnostics()
    s.edge = "test"
    s._log = io.StringIO()
    s._last_kiki_end = 0.0
    s._bg = set()
    s._warm = None
    return s


class _Voice:
    sample_rate = 24000

    def __call__(self, text, default="en"):
        return np.zeros(2400, np.float32)


async def approved(action="reply"):
    return Decision(action, "test", "Anna")


async def test_reply_is_spoken_and_logged():
    s = make("It's Copenhagen, Anna!")
    dec = asyncio.create_task(approved())
    await s._speak(dec, 0.0, chime=False, question="Kiki, what's the capital of Denmark?")
    assert any(isinstance(m, bytes) for m in s.ws.sent), "no audio sent"
    assert s.brain.history[-1].speaker == "Kiki" and "Copenhagen" in s.brain.history[-1].text
    assert s.diag.replies and "Copenhagen" in s.diag.replies[-1]["text"], "reply not in diagnostics"


async def test_marker_only_reply_starts_lookup_without_crashing():
    s = make("[LOOKUP: Nvidia stock price]")
    started = []
    s._start_lookup = started.append
    dec = asyncio.create_task(approved())
    await s._speak(dec, 0.0, chime=False, question="Kiki, what's Nvidia at?")
    assert started and started[0].query == "Nvidia stock price" and started[0].asker == "Anna"


async def test_unapproved_speculative_reply_sends_nothing():
    s = make("It's Copenhagen!")
    dec = asyncio.create_task(approved("silent"))
    await s._speak(dec, 0.0, chime=False)
    assert s.ws.sent == [] and not s.diag.replies


async def test_unapproved_marker_only_reply_starts_no_lookup():
    s = make("[LOOKUP: Nvidia stock price]")
    started = []
    s._start_lookup = started.append
    await s._speak(asyncio.create_task(approved("silent")), 0.0, chime=False, question="x")
    assert started == []
