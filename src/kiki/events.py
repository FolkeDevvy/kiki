"""Typed events and a tiny in-process pub/sub bus.

Times (`t`) are seconds on the audio clock: samples consumed / sample rate. This
keeps file replays and live audio identical, and lets STT word timestamps be
mapped onto the same clock.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np


@dataclass(frozen=True)
class SpeechStarted:
    t: float


@dataclass(frozen=True)
class SpeechPaused:
    t: float


@dataclass(frozen=True)
class Segment:
    """One contiguous run of speech (ends at a pause). A turn has 1+ segments."""

    t0: float
    t1: float
    audio: np.ndarray = field(repr=False)  # int16


@dataclass(frozen=True)
class TurnEnded:
    """Someone finished a turn: the floor is open."""

    t: float
    start: float
    probability: float  # Smart Turn's P(complete) at the decision point
    forced: bool  # True if ended by max_pause timeout, not by Smart Turn
    audio: np.ndarray = field(repr=False)  # int16 audio of the whole turn


Event = SpeechStarted | SpeechPaused | Segment | TurnEnded


class Bus:
    def __init__(self) -> None:
        self._subs: list[asyncio.Queue[Event | None]] = []

    def publish(self, event: Event) -> None:
        for q in self._subs:
            q.put_nowait(event)

    def close(self) -> None:
        for q in self._subs:
            q.put_nowait(None)

    def subscribe(self) -> AsyncIterator[Event]:
        q: asyncio.Queue[Event | None] = asyncio.Queue()
        self._subs.append(q)

        async def gen() -> AsyncIterator[Event]:
            while (ev := await q.get()) is not None:
                yield ev

        return gen()
