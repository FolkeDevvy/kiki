"""Turns a stream of frames into speech / pause / turn-end events.

States:  IDLE --speech--> SPEECH --short silence--> PAUSED --Smart Turn says done--> IDLE
                                   ^                   |
                                   +----speech again---+   (or max_pause -> forced end)
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Callable

import numpy as np

from kiki.config import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, Settings
from kiki.events import Event, Segment, SpeechPaused, SpeechStarted, TurnEnded

MAX_TURN_SECONDS = 30  # keep at most this much audio per turn (for speaker ID later)


class State(Enum):
    IDLE = "idle"
    SPEECH = "speech"
    PAUSED = "paused"


class Segmenter:
    def __init__(
        self,
        settings: Settings,
        vad: Callable[[np.ndarray], float],
        turn: Callable[[np.ndarray], float],
    ) -> None:
        self.s = settings
        self.vad = vad
        self.turn = turn
        self.state = State.IDLE
        self.samples = 0  # audio clock
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, settings.preroll_ms // FRAME_MS))
        self._turn: list[np.ndarray] = []
        self._turn_start = 0.0
        self._seg_start = 0  # index into self._turn where the current segment starts
        self._seg_t0 = 0.0
        self._speech_run = 0  # consecutive speech frames
        self._silence_run = 0  # consecutive non-speech frames
        self.last_probability: float | None = None
        self.last_vad = 0.0
        self.turn_ms: list[float] = []  # Smart Turn call durations (drained by edge stats)

    @property
    def t(self) -> float:
        return self.samples / SAMPLE_RATE

    @property
    def active(self) -> bool:
        return self.state is not State.IDLE

    def process(self, frame: np.ndarray) -> list[Event]:
        """Feed one 512-sample int16 frame; returns events in order."""
        assert len(frame) == FRAME_SAMPLES
        p = self.vad(frame)
        self.last_vad = p
        self.samples += FRAME_SAMPLES
        events: list[Event] = []
        is_speech = p >= self.s.vad_threshold
        is_silence = p < self.s.vad_threshold - 0.15  # hysteresis
        self._speech_run = self._speech_run + 1 if is_speech else 0
        self._silence_run = self._silence_run + 1 if is_silence else 0
        min_speech = max(1, self.s.min_speech_ms // FRAME_MS)

        if self.state is State.IDLE:
            self._preroll.append(frame)
            if self._speech_run >= min_speech:
                self._turn = list(self._preroll)
                self._preroll.clear()
                self._turn_start = self.t - len(self._turn) * FRAME_SAMPLES / SAMPLE_RATE
                self._seg_start, self._seg_t0 = 0, self._turn_start
                self.state = State.SPEECH
                events.append(SpeechStarted(t=self.t - min_speech * FRAME_MS / 1000))
            return events

        self._append(frame)

        if self.state is State.SPEECH:
            if self._silence_run * FRAME_MS >= self.s.pause_ms:
                self.state = State.PAUSED
                events.append(SpeechPaused(t=self.t))
                events.append(self._segment())
                t = time.perf_counter()
                prob = self.turn(self.turn_audio())
                self.turn_ms.append((time.perf_counter() - t) * 1000)
                self.last_probability = prob
                if prob >= self.s.turn_threshold:
                    events.append(self._end(prob, forced=False))
            return events

        # PAUSED
        if self._speech_run >= min_speech:
            self.state = State.SPEECH  # same turn continues, new segment
            back = min(len(self._turn), min_speech + max(1, self.s.preroll_ms // FRAME_MS))
            self._seg_start = len(self._turn) - back
            self._seg_t0 = self.t - back * FRAME_SAMPLES / SAMPLE_RATE
        elif self._silence_run * FRAME_MS >= self.s.max_pause_ms:
            events.append(self._end(self.last_probability or 0.0, forced=True))
        return events

    def turn_audio(self) -> np.ndarray:
        return np.concatenate(self._turn) if self._turn else np.zeros(0, np.int16)

    def _segment(self) -> Segment:
        audio = np.concatenate(self._turn[self._seg_start :])
        return Segment(t0=self._seg_t0, t1=self.t, audio=audio)

    def _append(self, frame: np.ndarray) -> None:
        self._turn.append(frame)
        max_frames = MAX_TURN_SECONDS * SAMPLE_RATE // FRAME_SAMPLES
        if len(self._turn) > max_frames:
            drop = len(self._turn) - max_frames
            del self._turn[:drop]
            self._seg_start = max(0, self._seg_start - drop)

    def _end(self, prob: float, forced: bool) -> TurnEnded:
        ev = TurnEnded(t=self.t, start=self._turn_start, probability=prob, forced=forced, audio=self.turn_audio())
        self.state = State.IDLE
        self._turn = []
        self._speech_run = self._silence_run = 0
        self.last_probability = None
        return ev
