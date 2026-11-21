"""Kiki server: receives speech segments from an edge device, transcribes them,
figures out who spoke, and keeps the conversation log.

Runs on the laptop for now, later on the GPU box. One edge connection = one
conversation session.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from websockets.asyncio.server import ServerConnection, serve

from kiki.config import SAMPLE_RATE, Settings
from kiki.server.speakers import Embedder, Registry
from kiki.server.stt import Parakeet


@dataclass
class Utterance:
    speaker: str
    text: str
    t0: float
    t1: float
    similarity: float


@dataclass
class Turn:
    start: float
    end: float
    p_complete: float
    forced: bool
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def speaker(self) -> str:
        # whoever spoke longest in this turn
        dur: dict[str, float] = {}
        for u in self.utterances:
            dur[u.speaker] = dur.get(u.speaker, 0.0) + (u.t1 - u.t0)
        return max(dur, key=dur.get) if dur else "?"

    @property
    def text(self) -> str:
        return " ".join(u.text for u in self.utterances if u.text)


class Models:
    def __init__(self, s: Settings) -> None:
        t = time.perf_counter()
        self.stt = Parakeet(s.models_dir)
        self.embed = Embedder(s.models_dir)
        print(f"[server] models loaded in {time.perf_counter() - t:.1f}s", flush=True)


class Session:
    def __init__(self, models: Models, registry: Registry, pool: ThreadPoolExecutor, log_dir: Path) -> None:
        self.m = models
        self.registry = registry
        self.pool = pool
        self.turns: list[Turn] = []
        self._current: list[Utterance] = []
        self._jobs: asyncio.Queue[tuple[float, float, np.ndarray]] = asyncio.Queue()
        self._worker = asyncio.create_task(self._work())
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = (log_dir / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl").open("a")

    async def handle(self, ws: ServerConnection) -> None:
        pending: dict | None = None
        async for msg in ws:
            if isinstance(msg, bytes):
                if pending is None:
                    print("[server] unexpected binary frame; ignoring")
                    continue
                audio = np.frombuffer(msg, dtype=np.int16).copy()
                self._jobs.put_nowait((pending["t0"], pending["t1"], audio))
                pending = None
                continue
            ev = json.loads(msg)
            kind = ev.get("type")
            if kind == "segment":
                pending = ev
            elif kind == "turn_end":
                await self._jobs.join()  # all segments of this turn are transcribed
                await self.on_turn_end(ev)
            elif kind == "hello":
                print(f"[server] edge connected: {ev.get('device')}", flush=True)

    async def _work(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            t0, t1, audio = await self._jobs.get()
            try:
                t = time.perf_counter()
                text, emb = await asyncio.gather(
                    loop.run_in_executor(self.pool, self.m.stt, audio),
                    loop.run_in_executor(self.pool, self.m.embed, audio),
                )
                fallback = self._current[-1].speaker if self._current else None
                who, sim = self.registry.assign(emb, len(audio) / SAMPLE_RATE, fallback=fallback)
                ms = (time.perf_counter() - t) * 1000
                if text:
                    u = Utterance(who, text, t0, t1, sim)
                    self._current.append(u)
                    print(f"{t0:7.2f}  [{who} {sim:+.2f}] {text}   ({ms:.0f} ms)", flush=True)
            except Exception as e:
                print(f"[server] segment failed: {e!r}", flush=True)
            finally:
                self._jobs.task_done()

    async def on_turn_end(self, ev: dict) -> None:
        turn = Turn(ev["start"], ev["t"], ev["p"], ev["forced"], self._current)
        self._current = []
        if not turn.utterances:
            return
        self.turns.append(turn)
        how = "timeout" if turn.forced else "smart-turn"
        print(f"{turn.end:7.2f}  ■ turn by {turn.speaker} ({how}, p={turn.p_complete:.2f})", flush=True)
        self._log.write(json.dumps({"kind": "turn", **asdict(turn)}) + "\n")
        self._log.flush()
        # milestone 4 hooks in here: floor policy -> respond / stay silent

    async def close(self) -> None:
        self._worker.cancel()
        self._log.close()


async def serve_forever(s: Settings, host: str, port: int, data_dir: Path) -> None:
    models = Models(s)
    registry = Registry.load(data_dir / "speakers.json", threshold=s.speaker_threshold)
    pool = ThreadPoolExecutor(max_workers=2)

    async def handler(ws: ServerConnection) -> None:
        session = Session(models, registry, pool, data_dir / "logs")
        try:
            await session.handle(ws)
        finally:
            await session.close()
            print("[server] edge disconnected", flush=True)

    async with serve(handler, host, port, max_size=8 * 1024 * 1024) as server:
        print(f"[server] listening on ws://{host}:{port}", flush=True)
        await server.serve_forever()
