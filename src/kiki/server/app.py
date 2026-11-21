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

import uuid

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from kiki.config import SAMPLE_RATE, Settings
from kiki.protocol import dumps
from kiki.server.brain import Brain, Decision
from kiki.server.llm import LLM
from kiki.server.speakers import Embedder, Registry
from kiki.server.speech import sentences
from kiki.server.stt import Parakeet
from kiki.server.tts import Kokoro


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


RESPONDER_SLOT = 0  # llama.cpp slot reserved for replies (slot 1: the gate, later)


class Models:
    def __init__(self, s: Settings) -> None:
        t = time.perf_counter()
        self.stt = Parakeet(s.models_dir)
        self.embed = Embedder(s.models_dir)
        self.tts = Kokoro(s.models_dir, voice=s.kiki_voice)
        self.llm = LLM(s.llm_url)
        # TTS gets its own thread so replies never queue behind transcription
        self.tts_pool = ThreadPoolExecutor(max_workers=1)
        # first inference allocates buffers and is ~2x slower; pay that now
        self.tts("Hello there, warming up.")
        self.stt(np.zeros(SAMPLE_RATE, np.int16))
        self.embed(np.zeros(SAMPLE_RATE, np.int16))
        print(f"[server] models loaded and warmed up in {time.perf_counter() - t:.1f}s", flush=True)


class Session:
    def __init__(self, s: Settings, models: Models, registry: Registry, pool: ThreadPoolExecutor, log_dir: Path) -> None:
        self.m = models
        self.registry = registry
        self.pool = pool
        self.brain = Brain(models.llm, s.engagement_window_s, s.history_turns)
        self.ws: ServerConnection | None = None
        self.turns: list[Turn] = []
        self._current: list[Utterance] = []
        # segments and turn ends share one queue so they're handled in order
        self._jobs: asyncio.Queue[tuple] = asyncio.Queue()
        self._worker = asyncio.create_task(self._work())
        self._reply: asyncio.Task | None = None
        self._warm: asyncio.Task | None = None
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = (log_dir / f"session-{time.strftime('%Y%m%d-%H%M%S')}.jsonl").open("a")

    async def handle(self, ws: ServerConnection) -> None:
        self.ws = ws
        pending: dict | None = None
        async for msg in ws:
            if isinstance(msg, bytes):
                if pending is None:
                    print("[server] unexpected binary frame; ignoring")
                    continue
                audio = np.frombuffer(msg, dtype=np.int16).copy()
                self._jobs.put_nowait(("segment", pending["t0"], pending["t1"], audio))
                pending = None
                continue
            ev = json.loads(msg)
            kind = ev.get("type")
            if kind == "segment":
                pending = ev
            elif kind == "turn_end":
                self._jobs.put_nowait(("turn_end", ev, time.perf_counter()))
            elif kind == "hello":
                print(f"[server] edge connected: {ev.get('device')}", flush=True)

    async def _work(self) -> None:
        while True:
            job = await self._jobs.get()
            try:
                if job[0] == "segment":
                    await self._segment(*job[1:])
                else:
                    await self.on_turn_end(*job[1:])
            except Exception as e:
                print(f"[server] {job[0]} failed: {e!r}", flush=True)
            finally:
                self._jobs.task_done()

    async def _segment(self, t0: float, t1: float, audio: np.ndarray) -> None:
        loop = asyncio.get_running_loop()
        t = time.perf_counter()
        text, emb = await asyncio.gather(
            loop.run_in_executor(self.pool, self.m.stt, audio),
            loop.run_in_executor(self.pool, self.m.embed, audio),
        )
        fallback = self._current[-1].speaker if self._current else None
        who, sim = self.registry.assign(emb, len(audio) / SAMPLE_RATE, fallback=fallback)
        ms = (time.perf_counter() - t) * 1000
        if text:
            self._current.append(Utterance(who, text, t0, t1, sim))
            print(f"{t0:7.2f}  [{who} {sim:+.2f}] {text}   ({ms:.0f} ms)", flush=True)

    async def on_turn_end(self, ev: dict, received: float) -> None:
        turn = Turn(ev["start"], ev["t"], ev["p"], ev["forced"], self._current)
        self._current = []
        if not turn.utterances:
            return
        self.turns.append(turn)
        for u in turn.utterances:
            self.brain.hear(self.registry.merged.get(u.speaker, u.speaker), u.text)
        speaker = self.registry.merged.get(turn.speaker, turn.speaker)
        d = self.brain.decide(speaker, turn.text)
        how = "timeout" if turn.forced else "smart-turn"
        verdict = f"→ reply ({d.reason})" if d.respond else f"· silent ({d.reason})"
        print(f"{turn.end:7.2f}  ■ turn by {speaker} ({how}, p={turn.p_complete:.2f}) {verdict}", flush=True)
        self._write({"kind": "turn", **asdict(turn), "decision": asdict(d)})
        if d.respond:
            if self._reply is not None and not self._reply.done():
                print("[server] already replying; skipping", flush=True)
            else:
                self._reply = asyncio.create_task(self._speak(d, received))
        else:
            self._prewarm()

    def _prewarm(self) -> None:
        """Keep the responder slot's KV cache current with the transcript, so a
        reply only has to process the newest line (~2 s saved on the laptop)."""
        if self._warm is not None and not self._warm.done():
            return  # one in flight is enough; the next turn will catch up
        self._warm = asyncio.create_task(self.m.llm.prewarm(self.brain.messages(), slot=RESPONDER_SLOT))

    async def _speak(self, d: Decision, received: float) -> None:
        """Stream LLM -> sentences -> TTS -> edge. Logs latency to first audio."""
        assert self.ws is not None
        loop = asyncio.get_running_loop()
        say_id = uuid.uuid4().hex[:8]
        said: list[str] = []
        first_audio: float | None = None
        marks: dict[str, float] = {"start": time.perf_counter() - received}

        async def deltas():
            async for d in self.brain.llm.stream(self.brain.messages(), slot=RESPONDER_SLOT):
                marks.setdefault("token", time.perf_counter() - received)
                yield d

        try:
            await self.ws.send(dumps("say", id=say_id, sample_rate=self.m.tts.sample_rate))
            async for sentence in sentences(deltas()):
                marks.setdefault("chunk", time.perf_counter() - received)
                audio = await loop.run_in_executor(self.m.tts_pool, self.m.tts, sentence)
                pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
                await self.ws.send(pcm)
                if first_audio is None:
                    first_audio = time.perf_counter() - received
                said.append(sentence)
            text = " ".join(said)
            await self.ws.send(dumps("say_end", id=say_id, text=text))
        except ConnectionClosed:
            return
        except Exception as e:
            print(f"[server] reply failed: {e!r}", flush=True)
            return
        if text:
            self.brain.said(text, d.target)
            self._prewarm()
            breakdown = " ".join(f"{k}={v * 1000:.0f}" for k, v in marks.items())
            fa = f"{first_audio * 1000:.0f} ms" if first_audio is not None else "n/a"
            print(f"          [Kiki] {text}   (first audio {fa}; {breakdown} ms)", flush=True)
            self._write({"kind": "kiki", "text": text, "target": d.target, "first_audio_s": first_audio, "marks": marks})

    def _write(self, rec: dict) -> None:
        self._log.write(json.dumps(rec) + "\n")
        self._log.flush()

    async def close(self) -> None:
        self._worker.cancel()
        if self._reply is not None:
            self._reply.cancel()
        self._log.close()


async def serve_forever(s: Settings, host: str, port: int, data_dir: Path) -> None:
    models = Models(s)
    registry = Registry.load(data_dir / "speakers.json", threshold=s.speaker_threshold)
    pool = ThreadPoolExecutor(max_workers=2)

    async def handler(ws: ServerConnection) -> None:
        session = Session(s, models, registry, pool, data_dir / "logs")
        try:
            await session.handle(ws)
        finally:
            await session.close()
            print("[server] edge disconnected", flush=True)

    async with serve(handler, host, port, max_size=8 * 1024 * 1024) as server:
        print(f"[server] listening on ws://{host}:{port}", flush=True)
        for _ in range(60):
            if await models.llm.healthy():
                print(f"[server] LLM ready at {s.llm_url}", flush=True)
                break
            await asyncio.sleep(2)
        else:
            print(f"[server] LLM at {s.llm_url} not ready; replies will fail until it is", flush=True)
        await server.serve_forever()
