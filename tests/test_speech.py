from kiki.server.brain import NAME
from kiki.server.speech import clean, sentences


async def agen(parts):
    for p in parts:
        yield p


async def collect(parts):
    return [s async for s in sentences(agen(parts))]


async def test_sentences_stream_out_as_they_complete():
    out = await collect(["Hi Anna! Most Thai pla", "ces do take reservations. ", "Call ahead", " on Fridays."])
    assert out == ["Hi Anna!", "Most Thai places do take reservations.", "Call ahead on Fridays."]


async def test_short_fragments_merge_after_first():
    out = await collect(["Sure. ", "Ok. ", "Yes. It is a long sentence here."])
    assert out[0] == "Sure."
    assert out[1] == "Ok. Yes."


def test_clean_strips_markdown_emoji_and_name_tag():
    assert clean("[**Kiki**]: Oh, *hi* Anna! 😊🌮") == "Oh, hi Anna!"
    assert clean("Kiki: sure thing") == "sure thing"


def test_name_matches_common_spellings():
    for t in ["Kiki, what time is it?", "hey kiki", "Kicky do you know", "is that kiki's?"]:
        assert NAME.search(t), t
    assert not NAME.search("I like kicking the ball")


async def test_first_chunk_can_end_at_a_comma_when_long_enough():
    out = await collect(["I don't actually know what Thai places here do, ", "but most of them take reservations."])
    assert out == ["I don't actually know what Thai places here do,", "but most of them take reservations."]
