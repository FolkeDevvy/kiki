import numpy as np

from kiki.audio.segmenter import Segmenter
from kiki.audio.turn import SmartTurn
from kiki.audio.vad import SileroVad
from kiki.config import FRAME_SAMPLES, Settings
from kiki.events import SpeechStarted, TurnEnded


def frames(audio):
    for i in range(len(audio) // FRAME_SAMPLES):
        yield audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]


def run(seg, audio):
    out = []
    for f in frames(audio):
        out += seg.process(f)
    return out


def make(models_dir):
    s = Settings(models_dir=models_dir)
    return Segmenter(
        s,
        SileroVad(models_dir / "silero_vad.onnx"),
        SmartTurn(models_dir / "smart-turn-v3.2-cpu.onnx"),
    )


def silence(sec):
    return np.zeros(int(16000 * sec), np.int16)


def test_silence_produces_nothing(models_dir):
    assert run(make(models_dir), silence(3)) == []


def test_one_utterance_is_one_turn(models_dir, say):
    speech = say("Hello everyone, how was your day today?", "q")
    events = run(make(models_dir), np.concatenate([silence(0.5), speech, silence(3)]))
    starts = [e for e in events if isinstance(e, SpeechStarted)]
    ends = [e for e in events if isinstance(e, TurnEnded)]
    assert len(starts) == 1
    assert len(ends) == 1
    end = ends[0]
    assert 0.3 < starts[0].t < 0.8
    assert 0 <= end.probability <= 1
    # turn audio covers the utterance
    assert len(end.audio) >= len(speech) * 0.8


def test_two_utterances_with_long_gap_are_two_turns(models_dir, say):
    a = say("Did you see the game last night?", "a")
    b = say("Yes, it was amazing.", "b")
    events = run(make(models_dir), np.concatenate([silence(0.3), a, silence(3), b, silence(3)]))
    assert len([e for e in events if isinstance(e, TurnEnded)]) == 2
