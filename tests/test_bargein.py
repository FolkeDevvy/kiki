import asyncio
import time

import numpy as np

from kiki.config import FRAME_SAMPLES
from kiki.edge.bargein import BargeIn
from kiki.edge.player import Player


class FakeVad:
    """Speech-like when the frame is loud enough to be a voice at all."""

    def __call__(self, frame):
        return 0.95 if np.abs(frame).mean() > 50 else 0.05


def tone(db):
    amp = 32768 * 10 ** (db / 20) * np.sqrt(2)
    return (np.sin(np.arange(FRAME_SAMPLES) / 3) * amp).astype(np.int16)


def test_sustained_loud_speech_interrupts():
    b = BargeIn(FakeVad(), min_dbfs=-38, min_speech_ms=250)
    hits = [b.update(tone(-25)) for _ in range(12)]
    assert not hits[0] and any(hits), "should trigger after ~250 ms of loud speech"


def test_echo_residue_and_blips_do_not_interrupt():
    b = BargeIn(FakeVad(), min_dbfs=-38, min_speech_ms=250)
    assert not any(b.update(tone(-46)) for _ in range(40))  # quiet echo, even if VAD says speech
    b.reset()
    blip = [b.update(tone(-20)) for _ in range(3)] + [b.update(tone(-60)) for _ in range(10)]
    assert not any(blip)  # a cough / clink: too short


def test_player_stop_drops_the_rest_of_the_reply():
    p = Player(enabled=False)
    p.start("a", 24000)
    p.enabled = True  # count playback time without a real device
    p.chunk("a", np.zeros(24000, np.int16).tobytes(), 24000)
    assert p.playing
    p.stop()
    assert not p.playing
    p.chunk("a", np.zeros(24000, np.int16).tobytes(), 24000)  # late chunk of the same reply
    assert not p.playing
    p.start("b", 24000)
    p.chunk("b", np.zeros(2400, np.int16).tobytes(), 24000)  # the next reply plays normally
    assert p.playing
    p.close()


async def test_server_cancels_reply_and_tells_claude():
    from kiki.server.app import Session
    from kiki.server.brain import Brain
    from kiki.server.diagnostics import Diagnostics

    s = Session.__new__(Session)
    s.brain = Brain(llm=None, gate=None)
    s.diag = Diagnostics()
    s.edge = "t"
    s._say_task = None
    s._reply = asyncio.create_task(asyncio.sleep(10))
    await asyncio.sleep(0)
    s._interrupted()
    await asyncio.sleep(0)
    assert s._reply.cancelled()
    s.brain.hear("Anna", "wait wait, actually")
    msg = s.brain.claude_message(unsure=True)
    assert "cut off" in msg
    assert "cut off" not in s.brain.claude_message()  # only once
