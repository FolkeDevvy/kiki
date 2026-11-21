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

# All test speakers are female, and none shares Kiki's own voices (af_heart, alma).
KOKORO = {"anna": "af_bella", "bea": "bf_emma", "cleo": "af_sarah"}
ESPEAK = {"anna": ["-v", "en-us+f3", "-s", "150", "-p", "60"],
          "bea": ["-v", "en-gb+f2", "-s", "140", "-p", "50"],
          "cleo": ["-v", "en-us+f4", "-s", "160", "-p", "45"]}

SCRIPTS = {}
SCRIPTS["dinner"] = [
    ("anna", "So, has anyone decided where we're going for dinner on Friday?", 1.2),
    ("bea", "I was thinking about that new Thai place downtown.", 1.0),
    ("cleo", "Oh, I've heard good things about it. Do they take reservations?", 1.4),
    ("anna", "I'm not sure. Kiki, do you know if Thai restaurants usually take reservations?", 2.5),
    ("bea", "Anyway, I can call them tomorrow morning and check.", 1.0),
    ("cleo", "Great, thanks Bea.", 2.0),
]
SCRIPTS["followup"] = [  # follow-ups that aren't questions should still get replies
    ("anna", "Kiki, what's the capital of Norway?", 5.0),
    ("anna", "Oh nice, I've never been there.", 5.0),
    ("anna", "My sister actually lives in Bergen though.", 5.0),
    ("anna", "Anyway Bea, are you coming to dinner tomorrow?", 1.2),
    ("bea", "Yes, I'll be there at seven.", 4.0),
]
SCRIPTS["bargein"] = [  # Bea talks over a long answer
    ("anna", "Kiki, tell us a long story about a dragon who loves baking.", 6.5),
    ("bea", "Okay okay, stop, that's enough. Kiki, what's two plus two?", 8.0),
]
SCRIPTS["lookup"] = [
    ("anna", "Kiki, what's the weather going to be like in Stockholm tomorrow?", 3.0),
    ("bea", "Anna, do you want to go to the cinema tonight?", 1.0),
    ("anna", "Sure, which movie?", 1.5),
    ("bea", "Kiki, what's the capital of Norway?", 5.0),
    ("bea", "I don't know which movie yet, we'll see.", 40.0),
]
SCRIPTS["manners"] = [
    ("anna", "Kiki, don't respond to the next thing I say.", 2.5),
    ("anna", "There's a lot of apples in the garden this year.", 3.0),
    ("bea", "What's the tallest mountain in Africa again?", 1.2),
    ("anna", "No idea. Something with a K?", 5.0),
    ("cleo", "Kiki, what's twelve times fifteen?", 5.0),
    ("cleo", "And what about thirteen times fifteen?", 5.0),
    ("anna", "Anyway Bea, did you feed the cat this morning?", 1.0),
    ("bea", "Yes, I did.", 2.0),
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
                x, sr = tts(text, voice=KOKORO[who]), tts.sample_rate
                audio = (np.clip(resample(x, sr, 16000), -1, 1) * 32767).astype(np.int16)
            truth.append(f"{who}\t{t:.2f}\t{text}")
            parts += [audio, np.zeros(int(gap * 16000), np.int16)]
            t += len(audio) / 16000 + gap
            print(f"{who:5s} {text}")
    write_wav(out, np.concatenate(parts))
    out.with_suffix(".txt").write_text("\n".join(truth) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
