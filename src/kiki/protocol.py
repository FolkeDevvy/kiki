"""Edge <-> server websocket protocol.

Edge -> server (JSON text frames, audio as the binary frame right after its header):
  {"type": "hello", "device": str, "sample_rate": 16000}
  {"type": "speech_started", "t": float}
  {"type": "segment", "t0": float, "t1": float}   + binary int16 PCM frame
  {"type": "turn_end", "t": float, "start": float, "p": float, "forced": bool}

Server -> edge:
  {"type": "say", "id": str, "text": str, "sample_rate": int}  + binary int16 PCM frames
  {"type": "say_end", "id": str}
  {"type": "stop"}     # stop playback now (barge-in / cancelled)

Times are seconds on the edge's audio clock.
"""

from __future__ import annotations

import json

PORT = 8765


def dumps(kind: str, **fields) -> str:
    return json.dumps({"type": kind, **fields})
