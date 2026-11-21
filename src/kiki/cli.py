"""`kiki` command line."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from kiki import config, models
from kiki.protocol import PORT


async def _edge(args: argparse.Namespace) -> None:
    from kiki.audio.sources import file_frames, mic_frames
    from kiki.edge import client

    s = config.load()
    frames = file_frames(Path(args.wav), realtime=not args.fast) if args.wav else mic_frames(s.input_device or None)
    await client.run(
        s, frames, None if args.local else args.server,
        playback=not args.no_playback,
        save_replies=Path(args.save_replies) if args.save_replies else None,
        half_duplex=not args.full_duplex,
        simulate_playback=args.simulate_playback,
    )


def _serve(args: argparse.Namespace) -> None:
    from kiki.server import app

    asyncio.run(app.serve_forever(config.load(), args.host, args.port, Path(args.data)))


def _enroll(args: argparse.Namespace) -> None:
    from kiki.audio.sources import read_wav
    from kiki.server.speakers import Embedder, Registry

    s = config.load()
    reg = Registry.load(Path(args.data) / "speakers.json")
    emb = Embedder(s.models_dir)
    for w in args.wav:
        v = reg.enroll(args.name, emb(read_wav(Path(w))))
    print(f"enrolled {v.name} ({v.n} samples) -> {reg.path}")


def _bench(args: argparse.Namespace) -> None:
    """Time the local models; run this on the Pi to check headroom."""
    import numpy as np

    from kiki.audio.turn import SmartTurn
    from kiki.audio.vad import SileroVad

    s = config.load()
    vad = SileroVad(models.path(s.models_dir, "silero_vad.onnx"))
    turn = SmartTurn(models.path(s.models_dir, models.smart_turn_file(s.smart_turn_model)))
    rng = np.random.default_rng(0)
    frame = (rng.standard_normal(512) * 1000).astype(np.int16)
    audio = (rng.standard_normal(16000 * 8) * 1000).astype(np.int16)

    def timeit(fn, n):
        fn()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n * 1000

    print(f"smart-turn model: {models.smart_turn_file(s.smart_turn_model)}", flush=True)
    print(f"silero vad / 32 ms frame : {timeit(lambda: vad(frame), 300):6.2f} ms", flush=True)
    print(f"smart-turn features (8 s): {timeit(lambda: turn.features(audio), 10):6.2f} ms", flush=True)
    print(f"smart-turn total     (8 s): {timeit(lambda: turn(audio), 5):6.2f} ms", flush=True)


def _devices(_: argparse.Namespace) -> None:
    import sounddevice as sd

    print(sd.query_devices())


def main() -> None:
    p = argparse.ArgumentParser(prog="kiki")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch-models", help="download local models")
    f.add_argument("--edge", action="store_true", help="only the small Pi models")
    f.add_argument("--server", action="store_true", help="only the server models")
    f.add_argument("--force", action="store_true")

    sub.add_parser("devices", help="list audio devices")
    sub.add_parser("bench", help="time the edge models")

    e = sub.add_parser("edge", help="run the device side: mic -> VAD/turns -> server")
    e.add_argument("--server", default=f"ws://localhost:{PORT}")
    e.add_argument("--local", action="store_true", help="no server; just print VAD/turn events")
    e.add_argument("--wav", help="replay a 16-bit WAV instead of the mic")
    e.add_argument("--fast", action="store_true", help="replay faster than real time")
    e.add_argument("--no-playback", action="store_true", help="don't play Kiki's voice")
    e.add_argument("--save-replies", help="also save Kiki's replies as WAVs in this dir")
    e.add_argument("--full-duplex", action="store_true", help="keep listening while Kiki talks (needs AEC)")
    e.add_argument("--simulate-playback", action="store_true",
                   help="pretend to play Kiki's replies (real timing, no sound): tests barge-in on replays")

    sv = sub.add_parser("serve", help="run the server: STT, speaker ID, brain")
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=PORT)
    sv.add_argument("--data", default="data")

    en = sub.add_parser("enroll", help="enroll a voice from WAV files")
    en.add_argument("name")
    en.add_argument("wav", nargs="+")
    en.add_argument("--data", default="data")

    args = p.parse_args()
    if args.cmd == "fetch-models":
        groups = {"edge"} if args.edge else {"server"} if args.server else {"edge", "server"}
        models.fetch(config.load().models_dir, groups, force=args.force)
    elif args.cmd == "devices":
        _devices(args)
    elif args.cmd == "bench":
        _bench(args)
    elif args.cmd == "edge":
        try:
            asyncio.run(_edge(args))
        except KeyboardInterrupt:
            pass
    elif args.cmd == "serve":
        try:
            _serve(args)
        except KeyboardInterrupt:
            pass
    elif args.cmd == "enroll":
        _enroll(args)


if __name__ == "__main__":
    main()
