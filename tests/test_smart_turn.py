from kiki.audio.turn import SmartTurn


def test_probability_range(models_dir, say):
    st = SmartTurn(models_dir / "smart-turn-v3.2-cpu.onnx")
    p = st(say("What time is the meeting tomorrow?", "complete"))
    assert 0.0 <= p <= 1.0
