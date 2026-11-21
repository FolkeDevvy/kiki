import asyncio
import urllib.request

import numpy as np

from kiki.edge.stats import EdgeStats
from kiki.server.diagnostics import Diagnostics, serve


def test_edge_stats_levels():
    st = EdgeStats(vad_threshold=0.5)
    loud = (np.sin(np.arange(512) / 5) * 8000).astype(np.int16)
    for _ in range(10):
        st.frame(loud, 0.9, muted=False)
    for _ in range(10):
        st.frame(np.zeros(512, np.int16), 0.0, muted=False)
    st.frame(loud, None, muted=True)
    snap = st.snapshot()
    assert snap["frames"] == 20 and snap["muted_frames"] == 1
    assert snap["speech_frac"] == 0.5
    assert -22 < snap["speech_dbfs"] < -10
    assert snap["clipped_frames"] == 0


async def test_http_endpoints():
    d = Diagnostics()
    d.edge_hello("kiki", {"arch": "aarch64"})
    d.edge_stats("kiki", {"level_dbfs": -40.0})
    sid = d.segment("kiki", 1.0, (np.ones(16000) * 1000).astype(np.int16), "Guest-1", 0.8, "hej", 120)
    d.turn("kiki", {"speaker": "Guest-1", "action": "silent", "reason": "x", "text": "hej"})
    server = await serve(d, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    def get(path):
        return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}").read()

    loop = asyncio.get_running_loop()
    page = await loop.run_in_executor(None, get, "/")
    wav = await loop.run_in_executor(None, get, f"/seg/{sid}.wav")
    status = await loop.run_in_executor(None, get, "/api/status")
    assert b"Kiki diagnostics" in page and b"hej" in page
    assert wav[:4] == b"RIFF" and len(wav) > 32000
    assert b"aarch64" in status
    server.close()
