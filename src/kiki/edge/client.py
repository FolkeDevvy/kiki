"""Edge (Pi) side: local VAD + turn detection, ships speech segments to the server."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from pathlib import Path
from typing import AsyncIterator

import numpy as np
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from kiki import models
from kiki.audio.segmenter import Segmenter
from kiki.audio.turn import SmartTurn
from kiki.audio.vad import SileroVad
from kiki.config import SAMPLE_RATE, Settings
from kiki.edge.player import Player
from kiki.events import Segment, SpeechPaused, SpeechStarted, TurnEnded
from kiki.protocol import dumps

ECHO_TAIL_S = 0.4  # keep ignoring the mic briefly after playback (room reverb)


def build_segmenter(s: Settings) -> Segmenter:
    vad = SileroVad(models.path(s.models_dir, "silero_vad.onnx"))
    turn = SmartTurn(models.path(s.models_dir, models.smart_turn_file(s.smart_turn_model)))
    return Segmenter(s, vad, turn)


class Link:
    """Websocket to the server that reconnects in the background; sends are
    dropped (with a note) while it's down, since stale audio isn't useful."""

    def __init__(self, url: str, player: Player) -> None:
        self.url = url
        self.player = player
        self.ws: ClientConnection | None = None
        self.saying = False  # between the server's "say" and "say_end"
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                async with connect(self.url, max_size=8 * 1024 * 1024, open_timeout=5) as ws:
                    await ws.send(dumps("hello", device=socket.gethostname(), sample_rate=SAMPLE_RATE))
                    self.ws = ws
                    print(f"[edge] connected to {self.url}", flush=True)
                    await self._receive(ws)
            except (OSError, ConnectionClosed, TimeoutError) as e:
                if self.ws is not None:
                    print(f"[edge] lost server: {e!r}", flush=True)
                self.ws = None
            await asyncio.sleep(2)

    async def _receive(self, ws: ClientConnection) -> None:
        say_id, rate = None, 24000
        async for msg in ws:
            if isinstance(msg, bytes):
                if say_id is not None:
                    self.player.chunk(say_id, msg, rate)
                continue
            ev = json.loads(msg)
            kind = ev.get("type")
            if kind == "say":
                say_id, rate = ev["id"], ev["sample_rate"]
                self.saying = True
                self.player.start(say_id, rate)
            elif kind == "say_end":
                saved = self.player.end(ev["id"], rate)
                print(f"   kiki  » {ev.get('text', '')}" + (f"  ({saved})" if saved else ""), flush=True)
                say_id = None
                self.saying = False
            elif kind == "stop":
                self.player.stop()

    async def send(self, *msgs: str | bytes) -> None:
        if self.ws is None:
            return
        try:
            for m in msgs:
                await self.ws.send(m)
        except ConnectionClosed:
            self.ws = None

    async def wait_connected(self, timeout: float) -> bool:
        for _ in range(int(timeout / 0.1)):
            if self.ws is not None:
                return True
            await asyncio.sleep(0.1)
        return False

    async def close(self) -> None:
        self._task.cancel()
        if self.ws is not None:
            await self.ws.close()


async def run(
    s: Settings,
    frames: AsyncIterator[np.ndarray],
    server_url: str | None,
    playback: bool = True,
    save_replies: Path | None = None,
    half_duplex: bool = True,
) -> None:
    seg = build_segmenter(s)
    player = Player(enabled=playback, save_dir=save_replies)
    link = Link(server_url, player) if server_url else None
    if link is not None and not await link.wait_connected(10):
        print(f"[edge] server {server_url} not reachable yet; will keep retrying", flush=True)
    loop = asyncio.get_running_loop()
    try:
        async for frame in frames:
            if half_duplex and time.monotonic() < player.busy_until + ECHO_TAIL_S:
                # Kiki is talking and we have no echo cancellation yet: don't
                # listen to ourselves. Feed silence so VAD/turn state settles.
                frame = np.zeros_like(frame)
            # Smart Turn takes ~100 ms on a pause; keep it off the event loop
            events = await loop.run_in_executor(None, seg.process, frame) if seg.active else seg.process(frame)
            for ev in events:
                if isinstance(ev, SpeechStarted):
                    print(f"{ev.t:7.2f}  ● speech", flush=True)
                    if link:
                        await link.send(dumps("speech_started", t=ev.t))
                elif isinstance(ev, SpeechPaused):
                    pass
                elif isinstance(ev, Segment):
                    if link:
                        await link.send(dumps("segment", t0=ev.t0, t1=ev.t1), ev.audio.astype(np.int16).tobytes())
                elif isinstance(ev, TurnEnded):
                    how = "timeout" if ev.forced else "smart-turn"
                    print(f"{ev.t:7.2f}  ■ turn end ({how}, p={ev.probability:.2f}, {ev.t - ev.start:.1f}s)", flush=True)
                    if link:
                        await link.send(dumps("turn_end", t=ev.t, start=ev.start, p=ev.probability, forced=ev.forced))
    finally:
        if link is not None:
            # replays end abruptly; give the server time to answer the last turn
            start = time.monotonic()
            while time.monotonic() - start < 30 and (
                time.monotonic() - start < 4 or link.saying or time.monotonic() < player.busy_until + 0.5
            ):
                await asyncio.sleep(0.2)
            await link.close()
        player.close()
