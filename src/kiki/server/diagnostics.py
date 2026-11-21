"""Remote diagnostics: what the server heard, decided and said, plus edge health.

A tiny stdlib HTTP server (no extra dependencies) on KIKI_DEBUG_PORT (8766):

  /                    auto-refreshing HTML overview (open in a browser)
  /api/status          edges, latest edge stats, counters (JSON)
  /api/segments        recent speech segments: transcript, speaker, level (JSON)
  /api/turns           recent turn decisions incl. gate probabilities (JSON)
  /api/replies         Kiki's replies with latency breakdown (JSON)
  /api/stats           edge stats history (JSON)
  /seg/<id>.wav        the audio of a segment, exactly as the server received it

It serves recordings of the room, so compose binds it to 127.0.0.1 by default.
"""

from __future__ import annotations

import asyncio
import html
import io
import json
import math
import time
import wave
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from kiki.config import SAMPLE_RATE


@dataclass
class SegmentRec:
    id: int
    wall: float
    edge: str
    t0: float
    dur_s: float
    level_dbfs: float
    peak_dbfs: float
    speaker: str
    similarity: float
    text: str
    stt_ms: float
    audio: np.ndarray = field(repr=False)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("audio")
        d["wav"] = f"/seg/{self.id}.wav"
        return d


class Diagnostics:
    def __init__(self, keep: int = 200) -> None:
        self.started = time.time()
        self.segments: deque[SegmentRec] = deque(maxlen=keep)
        self.turns: deque[dict] = deque(maxlen=keep)
        self.replies: deque[dict] = deque(maxlen=keep)
        self.stats: deque[dict] = deque(maxlen=720)  # ~1 h at one per 5 s
        self.edges: dict[str, dict] = {}
        self._next_id = 1

    # ---- feeding ----
    def edge_hello(self, edge: str, info: dict) -> None:
        self.edges[edge] = {"connected": True, "since": time.time(), "hello": info, "last_stats": None}

    def edge_gone(self, edge: str) -> None:
        if edge in self.edges:
            self.edges[edge]["connected"] = False
            self.edges[edge]["gone_at"] = time.time()

    def edge_stats(self, edge: str, stats: dict) -> None:
        rec = {"wall": time.time(), "edge": edge, **stats}
        self.stats.append(rec)
        if edge in self.edges:
            self.edges[edge]["last_stats"] = rec

    def segment(self, edge: str, t0: float, audio: np.ndarray, speaker: str, sim: float, text: str, stt_ms: float) -> int:
        x = audio.astype(np.float32) / 32768.0
        level = 10 * math.log10(max(float(np.mean(x * x)), 1e-9))
        peak = 20 * math.log10(max(float(np.max(np.abs(x))) if len(x) else 0.0, 1e-9))
        rec = SegmentRec(self._next_id, time.time(), edge, t0, round(len(audio) / SAMPLE_RATE, 2),
                         round(level, 1), round(peak, 1), speaker, round(sim, 3), text, round(stt_ms), audio)
        self._next_id += 1
        self.segments.append(rec)
        return rec.id

    def turn(self, edge: str, rec: dict) -> None:
        self.turns.append({"wall": time.time(), "edge": edge, **rec})

    def reply(self, edge: str, rec: dict) -> None:
        self.replies.append({"wall": time.time(), "edge": edge, **rec})

    # ---- views ----
    def status(self) -> dict:
        return {
            "uptime_s": round(time.time() - self.started),
            "edges": self.edges,
            "counts": {"segments": len(self.segments), "turns": len(self.turns), "replies": len(self.replies)},
        }

    def wav(self, seg_id: int) -> bytes | None:
        rec = next((r for r in self.segments if r.id == seg_id), None)
        if rec is None:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(rec.audio.astype(np.int16).tobytes())
        return buf.getvalue()

    def page(self) -> str:
        e = html.escape

        def ago(t: float) -> str:
            s = int(time.time() - t)
            return f"{s}s" if s < 120 else f"{s // 60}m"

        edges = []
        for name, info in self.edges.items():
            st = info.get("last_stats") or {}
            state = "🟢 connected" if info["connected"] else f"🔴 gone {ago(info.get('gone_at', 0))} ago"
            rows = "".join(f"<tr><td>{e(k)}</td><td>{e(str(v))}</td></tr>" for k, v in st.items() if k not in ("wall", "edge"))
            hello = e(json.dumps(info.get("hello", {})))
            edges.append(f"<h3>{e(name)} — {state}</h3><p class=m>{hello}</p>"
                         f"<table>{rows or '<tr><td>no stats yet</td></tr>'}</table>")
        segs = "".join(
            f"<tr><td>{ago(r.wall)}</td><td>{e(r.speaker)} <span class=m>{r.similarity:+.2f}</span></td>"
            f"<td>{r.dur_s}s</td><td>{r.level_dbfs} / {r.peak_dbfs}</td><td>{e(r.text)}</td>"
            f"<td><audio controls preload=none src='/seg/{r.id}.wav'></audio></td></tr>"
            for r in reversed(self.segments)
        )
        turns = "".join(
            f"<tr><td>{ago(t['wall'])}</td><td>{e(str(t.get('speaker')))}</td><td><b>{e(str(t.get('action')))}</b></td>"
            f"<td>{e(str(t.get('reason')))}</td><td class=m>{e(json.dumps(t.get('gate_probs')))}</td>"
            f"<td>{e(str(t.get('text', ''))[:120])}</td></tr>"
            for t in reversed(self.turns)
        )
        replies = "".join(
            f"<tr><td>{ago(r['wall'])}</td><td>{e(r.get('text', ''))}</td>"
            f"<td class=m>{e(json.dumps(r.get('marks')))}</td></tr>"
            for r in reversed(self.replies)
        )
        return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content=5><title>Kiki diagnostics</title>
<style>body{{font:14px system-ui;margin:16px;background:#fff;color:#111}}
@media (prefers-color-scheme:dark){{body{{background:#111;color:#eee}}}}
table{{border-collapse:collapse;margin:8px 0 24px}}td{{border-bottom:1px solid #8884;padding:3px 8px;vertical-align:top}}
.m{{opacity:.6;font-size:12px}}audio{{height:28px}}</style></head><body>
<h1>Kiki diagnostics</h1><p class=m>levels in dBFS (speech ≈ −35…−15, room noise ≈ −60…−45, clipping ≈ 0). Auto-refreshes every 5 s.</p>
<h2>Edges</h2>{''.join(edges) or '<p>no edge has connected yet</p>'}
<h2>Kiki said</h2><table>{replies}</table>
<h2>Turns</h2><table>{turns}</table>
<h2>Segments (newest first)</h2><table><tr><td></td><td>speaker</td><td>len</td><td>level/peak</td><td>transcript</td><td></td></tr>{segs}</table>
</body></html>"""


async def serve(diag: Diagnostics, host: str, port: int) -> asyncio.base_events.Server:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = (await asyncio.wait_for(reader.readline(), 5)).decode("latin-1")
            while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                pass  # skip headers
            path = line.split(" ")[1] if line.count(" ") >= 2 else "/"
            path = path.split("?")[0]
            status, ctype, body = "200 OK", "application/json", b""
            if path == "/":
                ctype, body = "text/html; charset=utf-8", diag.page().encode()
            elif path == "/api/status":
                body = json.dumps(diag.status(), default=str).encode()
            elif path == "/api/segments":
                body = json.dumps([r.public() for r in diag.segments]).encode()
            elif path == "/api/turns":
                body = json.dumps(list(diag.turns), default=str).encode()
            elif path == "/api/replies":
                body = json.dumps(list(diag.replies), default=str).encode()
            elif path == "/api/stats":
                body = json.dumps(list(diag.stats)).encode()
            elif path.startswith("/seg/") and path.endswith(".wav"):
                data = diag.wav(int(path[5:-4])) if path[5:-4].isdigit() else None
                if data is None:
                    status, body = "404 Not Found", b'{"error": "no such segment"}'
                else:
                    ctype, body = "audio/wav", data
            else:
                status, body = "404 Not Found", b'{"error": "not found"}'
            writer.write(
                f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
                f"Cache-Control: no-store\r\nConnection: close\r\n\r\n".encode() + body
            )
            await writer.drain()
        except Exception as e:  # never let a bad request hurt the server
            print(f"[diag] request failed: {e!r}", flush=True)
        finally:
            writer.close()

    server = await asyncio.start_server(handle, host, port)
    print(f"[diag] diagnostics on http://{host}:{port}/", flush=True)
    return server
