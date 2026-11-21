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
    # Swedish mis-hearings at the start of a line are only "maybe named"
    from kiki.server.brain import maybe_named

    for t in ["Kiva är huvudstaden i Norge?", "Kikes vara inte på nästa sak", "Kik i hur länge ska man koka ett ägg",
              "Klick i vilket år föll Berlinmuren"]:
        assert maybe_named(t) and not NAME.search(t), t
    for t in ["Kom hit nu", "Kaffe, någon?", "Can you pass the milk"]:
        assert not maybe_named(t), t
