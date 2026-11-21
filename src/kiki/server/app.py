"""Kiki server: receives speech segments from an edge device, transcribes them,
figures out who spoke, and keeps the conversation log.

Runs on the laptop for now, later on the GPU box. One edge connection = one
conversation session.
"""

from __future__ import annotations

import asyncio
import json
import re
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
from kiki.server import claude_brain
from kiki.server.brain import NAME, PERSONA_CLAUDE, Brain, Decision, is_pass, maybe_named
from kiki.server.diagnostics import Diagnostics
from kiki.server.diagnostics import serve as serve_diagnostics
from kiki.server.gate import Gate
from kiki.server.lang import detect
from kiki.server.research import Lookup, Researcher, promised_lookup, query_from_question, split_lookup
from kiki.server.llm import LLM
from kiki.server.speakers import Embedder, Registry
from kiki.server.speech import sentences
from kiki.server.stt import Parakeet
from kiki.server.tts import Voice


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
        if s.stt_model == "bilingual":
            from kiki.server.stt import Bilingual

            self.stt = Bilingual(s.models_dir, en_min_conf=s.stt_en_min_conf, kb_size=s.kb_whisper_size)
        else:
            self.stt = Parakeet(s.models_dir, version=s.stt_model)
        self.embed = Embedder(s.models_dir)
        self.tts = Voice(s.models_dir, en_voice=s.kiki_voice, sv_voice=s.kiki_voice_sv)
        self.llm = LLM(s.llm_url)
        self.gate = Gate(s.llm_url, slot=GATE_SLOT)
        self.research = Researcher(s.search_url, self.llm) if s.search_url else None
        self.use_claude = s.brain == "claude" and claude_brain.available()
        if s.brain == "claude" and not self.use_claude:
            print("[server] KIKI_BRAIN=claude but no CLAUDE_CODE_OAUTH_TOKEN; using the local brain", flush=True)
        if self.use_claude:
            self.research = claude_brain.ClaudeResearcher(s.claude_model, s.claude_research_effort)
            print(f"[server] brain: Claude ({s.claude_model}, effort {s.claude_effort} for replies, "
                  f"{s.claude_research_effort} for research) via Claude Code", flush=True)
        self.claude_model, self.claude_effort = s.claude_model, s.claude_effort
        # TTS gets its own thread so replies never queue behind transcription
        self.tts_pool = ThreadPoolExecutor(max_workers=1)
        # first inference allocates buffers and is ~2x slower; pay that now
        self.tts("Hello there, warming up.")
        self.tts("Hej, jag värmer upp.")
        self.stt(np.zeros(SAMPLE_RATE, np.int16))
        self.embed(np.zeros(SAMPLE_RATE, np.int16))
        print(f"[server] models loaded and warmed up in {time.perf_counter() - t:.1f}s", flush=True)


class Session:
    def __init__(self, s: Settings, models: Models, registry: Registry, pool: ThreadPoolExecutor, log_dir: Path,
                 diag: Diagnostics | None = None) -> None:
        self.m = models
        self.diag = diag or Diagnostics()
        self.edge = "?"
        self.registry = registry
        self.pool = pool
        self.brain = Brain(models.llm, models.gate, s.engagement_window_s, s.history_turns, s.proactivity)
        self.brain.allow_unsure = models.use_claude
        self.claude = (claude_brain.ClaudeResponder(PERSONA_CLAUDE, models.claude_model, models.claude_effort)
                       if models.use_claude else None)
        if self.claude is not None:
            self._claude_warm = asyncio.create_task(self._warm_claude())

        self.ws: ServerConnection | None = None
        self.turns: list[Turn] = []
        self._current: list[Utterance] = []
        # segments and turn ends share one queue so they're handled in order
        self._jobs: asyncio.Queue[tuple] = asyncio.Queue()
        self._worker = asyncio.create_task(self._work())
        self._reply: asyncio.Task | None = None
        self._warm: asyncio.Task | None = None
        self._bg: set[asyncio.Task] = set()
        # floor tracking, so background answers wait for a pause
        self._speech_active = False
        self._last_turn_end = 0.0
        self._last_kiki_end = 0.0
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
            elif kind == "stats":
                ev.pop("type", None)
                self.diag.edge_stats(self.edge, ev)
            elif kind == "speech_started":
                self._speech_active = True
            elif kind == "turn_end":
                self._speech_active = False
                self._last_turn_end = time.monotonic()
                self._jobs.put_nowait(("turn_end", ev, time.perf_counter()))
            elif kind == "hello":
                self.edge = str(ev.get("device", "?"))
                self.diag.edge_hello(self.edge, {k: v for k, v in ev.items() if k != "type"})
                print(f"[server] edge connected: {self.edge}", flush=True)

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
        if text and self._is_own_echo(text):
            self.diag.segment(self.edge, t0, audio, "Kiki-echo", sim, text, ms)
            print(f"{t0:7.2f}  [echo of Kiki, ignored] {text}", flush=True)
            return
        rec_id = self.diag.segment(self.edge, t0, audio, who, sim, text, ms)
        if getattr(self.m.stt, "last", None):
            self.diag.segments[-1].stt = dict(self.m.stt.last)
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

    async def _warm_claude(self) -> None:
        t = time.perf_counter()
        try:
            await self.claude.warm()
            print(f"[server] Claude session ready ({time.perf_counter() - t:.1f} s)", flush=True)
        except Exception as e:
            print(f"[server] Claude session failed to start: {e!r}", flush=True)

    def _replying(self) -> bool:
        return self._reply is not None and not self._reply.done()

    async def _decide_and_act(self, turn: Turn, speaker: str, received: float) -> None:
        decision = asyncio.create_task(self.brain.decide(speaker, turn.text))
        spec: asyncio.Task | None = None
        likely = maybe_named(turn.text) if self.claude is not None else self.brain.likely_reply(turn.text)
        if likely and not self._replying():
            # speculative: generate while the gate thinks; audio waits for its verdict
            spec = self._reply = asyncio.create_task(self._speak(decision, received, chime=False, question=turn.text))
        d = await decision
        how = "timeout" if turn.forced else "smart-turn"
        gate_ms = f", gate {d.verdict.ms:.0f} ms" if d.verdict else ""
        arrow = {"reply": "→ reply", "chime": "→ chime", "quiet": "· quiet"}.get(d.action, "· silent")
        print(f"{turn.end:7.2f}  ■ turn by {speaker} ({how}, p={turn.p_complete:.2f}) {arrow} "
              f"({d.reason}{gate_ms})", flush=True)
        self.diag.turn(self.edge, {
            "speaker": speaker, "text": turn.text, "action": d.action, "reason": d.reason,
            "gate_probs": {k: round(v, 3) for k, v in d.verdict.probs.items()} if d.verdict else None,
            "gate_ms": round(d.verdict.ms) if d.verdict else None, "smart_turn_p": turn.p_complete,
            "forced": turn.forced})
        self._write({"kind": "turn", **asdict(turn), "decision": {
            "action": d.action, "reason": d.reason, "named": d.named,
            "gate": asdict(d.verdict) if d.verdict else None}})
        if d.action == "chime":
            if spec is not None:
                spec.cancel()  # a chime-in uses its own (PASS-able) prompt
            if not self._replying() or self._reply is spec:
                self._reply = asyncio.create_task(self._speak(decision, received, chime=True, question=turn.text))
        elif d.action == "reply" and spec is None:
            if self._replying():
                print("[server] already replying; skipping", flush=True)
            else:
                self._reply = asyncio.create_task(self._speak(decision, received, chime=False, question=turn.text))
        elif not d.respond:
            self._prewarm()

    def _prewarm(self) -> None:
        """Keep the responder slot's KV cache current with the transcript, so a
        reply only has to process the newest line (~2 s saved on the laptop)."""
        if self.claude is not None:
            return  # Claude Code caches its own session
        if self._warm is not None and not self._warm.done():
            return  # one in flight is enough; the next turn will catch up
        self._warm = asyncio.create_task(self.m.llm.prewarm(self.brain.messages(), slot=RESPONDER_SLOT))

    async def _speak(self, decision: asyncio.Task, received: float, chime: bool, question: str = "") -> None:
        """Stream LLM -> sentences -> TTS -> edge. Nothing is sent until the gate
        has approved; a chime-in may still be declined by the responder (PASS)."""
        assert self.ws is not None
        loop = asyncio.get_running_loop()
        say_id = uuid.uuid4().hex[:8]
        said: list[str] = []
        first_audio: float | None = None
        marks: dict[str, float] = {"start": time.perf_counter() - received}

        lookup_query: list[str] = []

        async def raw():
            if self.claude is not None:
                unsure = not chime  # Claude judges every reply from context, named or not
                source = self.claude.stream(self.brain.claude_message(chime=chime, unsure=unsure))
            else:
                source = self.brain.llm.stream(self.brain.messages(chime=chime), slot=RESPONDER_SLOT)
            async for d in source:
                marks.setdefault("token", time.perf_counter() - received)
                yield d

        def deltas():
            return raw() if chime else split_lookup(raw(), lookup_query.append)

        # ambiguous sentences ("1990.") are spoken in the language Kiki was addressed in
        lang_default = self.brain.reply_language()
        text = ""
        d: Decision | None = None
        try:
            async for sentence in sentences(deltas()):
                marks.setdefault("chunk", time.perf_counter() - received)
                if not said:
                    d = await decision
                    marks.setdefault("gate", time.perf_counter() - received)
                    wanted = "chime" if chime else "reply"
                    if d.action != wanted:
                        return  # speculative reply not approved
                    if (chime or self.claude is not None) and is_pass(sentence):
                        print(f"          [Kiki] (passed on {'chiming in' if chime else 'an unverified reply'})", flush=True)
                        self._prewarm()
                        return
                audio = await loop.run_in_executor(
                    self.m.tts_pool, lambda: self.m.tts(sentence, default=lang_default)
                )
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
                self._last_kiki_end = time.monotonic()
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
            self.brain.said(text, chime=chime, target=None if chime or d is None else d.target,
                            in_session=self.claude is not None)
            self._prewarm()
            breakdown = " ".join(f"{k}={v * 1000:.0f}" for k, v in marks.items())
            fa = f"{first_audio * 1000:.0f} ms" if first_audio is not None else "n/a"
            print(f"          [Kiki] {text}   (first audio {fa}; {breakdown} ms)", flush=True)
            self._write({"kind": "kiki", "text": text, "chime": chime, "first_audio_s": first_audio, "marks": marks})
            self.diag.reply(self.edge, {"text": text, "chime": chime,
                                        "claude": dict(self.claude.last_usage) if self.claude else None,
                                        "marks": {k: round(v * 1000) for k, v in marks.items()},
                                        "first_audio_ms": round(first_audio * 1000) if first_audio else None})
        if not question:
            heard = [l for l in self.brain.history if l.speaker != "Kiki"]
            question = heard[-1].text if heard else ""
        if not lookup_query and not chime and text and promised_lookup(text) and question:
            lookup_query.append(query_from_question(question))  # promised to check, forgot the marker
        if lookup_query:
            if d is None:  # marker-only reply: nothing was spoken, so check the gate now
                d = await decision
                if d.action != "reply":
                    return  # speculative reply that wasn't approved: no lookup either
            self._start_lookup(Lookup(lookup_query[0], question or lookup_query[0], lang_default, d.target))

    def _is_own_echo(self, text: str, window_s: float = 8.0) -> bool:
        """Did the mic just pick up Kiki's own voice? (the edge mutes while playing,
        but device buffers can outlast that)"""
        if time.monotonic() - self._last_kiki_end > window_s:
            return False
        kiki = [l.text for l in self.brain.history[-3:] if l.speaker == "Kiki"]
        if not kiki:
            return False
        norm = lambda x: re.sub(r"[^\w ]", "", x.lower()).split()  # noqa: E731
        heard = norm(text)
        said = norm(kiki[-1])
        if not heard:
            return False
        # fraction of heard words that appear in what Kiki just said (echoes are often partial)
        overlap = sum(w in said for w in heard) / len(heard)
        return overlap >= 0.6 and len(heard) >= 2

    # ---- background lookups -------------------------------------------------------
    def _start_lookup(self, lk: Lookup) -> None:
        print(f"          [Kiki] (looking up: {lk.query!r})", flush=True)
        self.brain.looking_up.append(lk.query)
        task = asyncio.create_task(self._lookup(lk))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _lookup(self, lk: Lookup) -> None:
        try:
            if self.m.research is None:
                lk.answer = ("Jag kan inte söka på nätet just nu." if lk.language == "sv"
                             else "I can't search the web right now.")
            else:
                await self.m.research.run(lk, self.brain.transcript())
            src = ", ".join(s.site for s in lk.sources[:3])
            print(f"          [Kiki] (lookup done in {time.time() - lk.started:.0f}s via {src or '-'}"
                  f"{'; error ' + lk.error if lk.error else ''})", flush=True)
            await self._wait_for_floor()
            await self._say_text(lk.answer, lk.language, target=lk.asker)
            self.diag.reply(self.edge, {"text": lk.answer, "lookup": lk.query,
                                        "sources": [s.url for s in lk.sources[:3]],
                                        "seconds": round(time.time() - lk.started, 1), "error": lk.error})
            self._write({"kind": "lookup", "query": lk.query, "answer": lk.answer, "error": lk.error,
                         "sources": [s.url for s in lk.sources[:5]], "timings": lk.timings})
        finally:
            if lk.query in self.brain.looking_up:
                self.brain.looking_up.remove(lk.query)

    async def _wait_for_floor(self, max_wait_s: float = 120.0) -> None:
        """Wait for a natural gap: nobody talking, Kiki not talking, a beat after both."""
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            now = time.monotonic()
            if (not self._speech_active and not self._replying()
                    and now - self._last_turn_end > 1.0 and now - self._last_kiki_end > 0.8):
                return
            await asyncio.sleep(0.2)

    async def _say_text(self, text: str, lang: str, target: str | None) -> None:
        """Speak a ready-made text (no LLM), sentence by sentence."""
        assert self.ws is not None
        loop = asyncio.get_running_loop()
        say_id = uuid.uuid4().hex[:8]

        async def once():
            yield text

        said: list[str] = []
        try:
            async for sentence in sentences(once()):
                audio = await loop.run_in_executor(self.m.tts_pool, lambda: self.m.tts(sentence, default=lang))
                if not said:
                    await self.ws.send(dumps("say", id=say_id, sample_rate=self.m.tts.sample_rate))
                await self.ws.send((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
                said.append(sentence)
            if said:
                await self.ws.send(dumps("say_end", id=say_id, text=" ".join(said)))
                self._last_kiki_end = time.monotonic()
        except ConnectionClosed:
            return
        if said:
            self.brain.said(" ".join(said), target=target)
            self._prewarm()
            print(f"          [Kiki] {' '.join(said)}   (lookup answer)", flush=True)

    def _write(self, rec: dict) -> None:
        self._log.write(json.dumps(rec) + "\n")
        self._log.flush()

    async def close(self) -> None:
        if self.claude is not None:
            await self.claude.reset()
        self._worker.cancel()
        if self._reply is not None:
            self._reply.cancel()
        self._log.close()


async def serve_forever(s: Settings, host: str, port: int, data_dir: Path) -> None:
    models = Models(s)
    pool = ThreadPoolExecutor(max_workers=2)
    diag = Diagnostics()
    await serve_diagnostics(diag, "0.0.0.0", s.debug_port)

    async def handler(ws: ServerConnection) -> None:
        # one registry per connection: enrolled voices from disk, guests local to it
        registry = Registry.load(data_dir / "speakers.json", threshold=s.speaker_threshold)
        session = Session(s, models, registry, pool, data_dir / "logs", diag)
        try:
            await session.handle(ws)
        except ConnectionClosed as e:
            print(f"[server] edge {session.edge} dropped: {e!r}", flush=True)
        finally:
            diag.edge_gone(session.edge)
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
