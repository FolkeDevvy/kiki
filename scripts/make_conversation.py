"""Synthesize a multi-speaker test conversation (runs in the dev image).

    python scripts/make_conversation.py recordings/conv3.wav            # Kokoro voices
    python scripts/make_conversation.py recordings/conv3e.wav --espeak  # robotic, fast

Writes a 16 kHz WAV plus a .txt with the ground truth (speaker, start, text).
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from kiki.audio.sources import read_wav, write_wav

KOKORO = {"anna": "af_heart", "bob": "am_michael", "carl": "bm_george"}
ESPEAK = {"anna": ["-v", "en-us+f3", "-s", "150", "-p", "60"],
          "bob": ["-v", "en-us+m3", "-s", "140", "-p", "35"],
          "carl": ["-v", "en-gb+m7", "-s", "160", "-p", "45"]}

SCRIPTS = {}
SCRIPTS["dinner"] = [
    ("anna", "So, has anyone decided where we're going for dinner on Friday?", 1.2),
    ("bob", "I was thinking about that new Thai place downtown.", 1.0),
    ("carl", "Oh, I've heard good things about it. Do they take reservations?", 1.4),
    ("anna", "I'm not sure. Kiki, do you know if Thai restaurants usually take reservations?", 2.5),
    ("bob", "Anyway, I can call them tomorrow morning and check.", 1.0),
    ("carl", "Great, thanks Bob.", 2.0),
]
SCRIPTS["manners"] = [
    ("anna", "Kiki, don't respond to the next thing I say.", 2.5),
    ("anna", "There's a lot of apples in the garden this year.", 3.0),
    ("bob", "What's the tallest mountain in Africa again?", 1.2),
    ("anna", "No idea. Something with a K?", 5.0),
    ("carl", "Kiki, what's twelve times fifteen?", 5.0),
    ("carl", "And what about thirteen times fifteen?", 5.0),
    ("anna", "Anyway Bob, did you feed the cat this morning?", 1.0),
    ("bob", "Yes, I did.", 2.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="recordings/conv3.wav")
    ap.add_argument("--espeak", action="store_true")
    ap.add_argument("--script", default="dinner", choices=sorted(SCRIPTS))
    args = ap.parse_args()
    out = Path(args.out)

    if not args.espeak:
        from kiki import config
        from kiki.server.tts import Kokoro, resample

        tts = Kokoro(config.load().models_dir)

    parts, truth, t = [np.zeros(8000, np.int16)], [], 0.5
    with tempfile.TemporaryDirectory() as d:
        for i, (who, text, gap) in enumerate(SCRIPTS[args.script]):
            if args.espeak:
                wav = Path(d) / f"{i}.wav"
                subprocess.run(["espeak-ng", *ESPEAK[who], "-w", str(wav), text], check=True)
                audio = read_wav(wav)
            else:
                t0 = time.perf_counter()
                x = tts(text, voice=KOKORO[who])
                print(f"  tts {time.perf_counter() - t0:.2f}s for {len(x) / tts.sample_rate:.2f}s audio")
                audio = (np.clip(resample(x, tts.sample_rate, 16000), -1, 1) * 32767).astype(np.int16)
            truth.append(f"{who}\t{t:.2f}\t{text}")
            parts += [audio, np.zeros(int(gap * 16000), np.int16)]
            t += len(audio) / 16000 + gap
            print(f"{who:5s} {text}")
    write_wav(out, np.concatenate(parts))
    out.with_suffix(".txt").write_text("\n".join(truth) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
