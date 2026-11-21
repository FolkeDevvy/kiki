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
from kiki.server.brain import Brain, Decision, is_pass
from kiki.server.gate import Gate
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


RESPONDER_SLOT = 0  # llama.cpp slots: each keeps its own warm KV cache
GATE_SLOT = 1


class Models:
    def __init__(self, s: Settings) -> None:
        t = time.perf_counter()
        self.stt = Parakeet(s.models_dir)
        self.embed = Embedder(s.models_dir)
        self.tts = Kokoro(s.models_dir, voice=s.kiki_voice)
        self.llm = LLM(s.llm_url)
        self.gate = Gate(s.llm_url, slot=GATE_SLOT)
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
        self.brain = Brain(models.llm, models.gate, s.engagement_window_s, s.history_turns, s.proactivity)

        self.ws: ServerConnection | None = None
        self.turns: list[Turn] = []
        self._current: list[Utterance] = []
        # segments and turn ends share one queue so they're handled in order
        self._jobs: asyncio.Queue[tuple] = asyncio.Queue()
        self._worker = asyncio.create_task(self._work())
        self._reply: asyncio.Task | None = None
        self._warm: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
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
        # decide in the background so transcription of the next turn isn't held up
        task = asyncio.create_task(self._decide_and_act(turn, speaker, received))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def _replying(self) -> bool:
        return self._reply is not None and not self._reply.done()

    async def _decide_and_act(self, turn: Turn, speaker: str, received: float) -> None:
        decision = asyncio.create_task(self.brain.decide(speaker, turn.text))
        spec: asyncio.Task | None = None
        if self.brain.likely_reply(turn.text) and not self._replying():
            # speculative: generate while the gate thinks; audio waits for its verdict
            spec = self._reply = asyncio.create_task(self._speak(decision, received, chime=False))
        d = await decision
        how = "timeout" if turn.forced else "smart-turn"
        gate_ms = f", gate {d.verdict.ms:.0f} ms" if d.verdict else ""
        arrow = {"reply": "→ reply", "chime": "→ chime", "quiet": "· quiet"}.get(d.action, "· silent")
        print(f"{turn.end:7.2f}  ■ turn by {speaker} ({how}, p={turn.p_complete:.2f}) {arrow} "
              f"({d.reason}{gate_ms})", flush=True)
        self._write({"kind": "turn", **asdict(turn), "decision": {
            "action": d.action, "reason": d.reason, "named": d.named,
            "gate": asdict(d.verdict) if d.verdict else None}})
        if d.action == "chime":
            if spec is not None:
                spec.cancel()  # a chime-in uses its own (PASS-able) prompt
            if not self._replying() or self._reply is spec:
                self._reply = asyncio.create_task(self._speak(decision, received, chime=True))
        elif d.action == "reply" and spec is None:
            if self._replying():
                print("[server] already replying; skipping", flush=True)
            else:
                self._reply = asyncio.create_task(self._speak(decision, received, chime=False))
        elif not d.respond:
            self._prewarm()

    def _prewarm(self) -> None:
        """Keep the responder slot's KV cache current with the transcript, so a
        reply only has to process the newest line (~2 s saved on the laptop)."""
        if self._warm is not None and not self._warm.done():
            return  # one in flight is enough; the next turn will catch up
        self._warm = asyncio.create_task(self.m.llm.prewarm(self.brain.messages(), slot=RESPONDER_SLOT))

    async def _speak(self, decision: asyncio.Task, received: float, chime: bool) -> None:
        """Stream LLM -> sentences -> TTS -> edge. Nothing is sent until the gate
        has approved; a chime-in may still be declined by the responder (PASS)."""
        assert self.ws is not None
        loop = asyncio.get_running_loop()
        say_id = uuid.uuid4().hex[:8]
        said: list[str] = []
        first_audio: float | None = None
        marks: dict[str, float] = {"start": time.perf_counter() - received}

        async def deltas():
            async for d in self.brain.llm.stream(self.brain.messages(chime=chime), slot=RESPONDER_SLOT):
                marks.setdefault("token", time.perf_counter() - received)
                yield d

        text = ""
        try:
            async for sentence in sentences(deltas()):
                marks.setdefault("chunk", time.perf_counter() - received)
                if not said:
                    d = await decision
                    marks.setdefault("gate", time.perf_counter() - received)
                    wanted = "chime" if chime else "reply"
                    if d.action != wanted:
                        return  # speculative reply not approved
                    if chime and is_pass(sentence):
                        print("          [Kiki] (passed on chiming in)", flush=True)
                        self._prewarm()
                        return
                audio = await loop.run_in_executor(self.m.tts_pool, self.m.tts, sentence)
                pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
                if first_audio is None:
                    await self.ws.send(dumps("say", id=say_id, sample_rate=self.m.tts.sample_rate))
                await self.ws.send(pcm)
                if first_audio is None:
                    first_audio = time.perf_counter() - received
                said.append(sentence)
            text = " ".join(said)
            if said:
                await self.ws.send(dumps("say_end", id=say_id, text=text))
        except asyncio.CancelledError:
            if said:  # cut off mid-reply: close the say so the edge isn't left hanging
                await self.ws.send(dumps("say_end", id=say_id, text=" ".join(said)))
            raise
        except ConnectionClosed:
            return
        except Exception as e:
            print(f"[server] reply failed: {e!r}", flush=True)
            return
        if text:
            self.brain.said(text, chime=chime, target=None if chime else d.target)
            self._prewarm()
            breakdown = " ".join(f"{k}={v * 1000:.0f}" for k, v in marks.items())
            fa = f"{first_audio * 1000:.0f} ms" if first_audio is not None else "n/a"
            print(f"          [Kiki] {text}   (first audio {fa}; {breakdown} ms)", flush=True)
            self._write({"kind": "kiki", "text": text, "chime": chime, "first_audio_s": first_audio, "marks": marks})

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
    pool = ThreadPoolExecutor(max_workers=2)

    async def handler(ws: ServerConnection) -> None:
        # one registry per connection: enrolled voices from disk, guests local to it
        registry = Registry.load(data_dir / "speakers.json", threshold=s.speaker_threshold)
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
                print(f"[server] LLM ready at {s.llm_url}; warming prompt caches...", flush=True)
                t = time.perf_counter()
                try:
                    warm_gate = Gate(s.llm_url, slot=GATE_SLOT, timeout=180)
                    await asyncio.gather(
                        warm_gate.judge("[Anna]: hello"),
                        models.llm.prewarm(Brain(models.llm, None).messages(), slot=RESPONDER_SLOT),
                    )
                    await warm_gate.aclose()
                    print(f"[server] prompt caches warm ({time.perf_counter() - t:.0f} s)", flush=True)
                except Exception as e:
                    print(f"[server] cache warmup failed: {e!r}", flush=True)
                break
            await asyncio.sleep(2)
        else:
            print(f"[server] LLM at {s.llm_url} not ready; replies will fail until it is", flush=True)
        await server.serve_forever()
