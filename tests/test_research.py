from kiki.server.research import LOOKUP_MARK, split_lookup


async def agen(parts):
    for p in parts:
        yield p


async def run(parts):
    found = []
    out = "".join([d async for d in split_lookup(agen(parts), found.append)])
    return out, found


async def test_marker_is_stripped_and_reported():
    out, found = await run(["[LOOK", "UP: väder Stockholm imorgon] Bra fråga, ", "jag kollar!"])
    assert found == ["väder Stockholm imorgon"]
    assert out.strip() == "Bra fråga, jag kollar!"


async def test_normal_reply_passes_through_untouched():
    out, found = await run(["Oslo är ", "huvudstaden i Norge."])
    assert found == [] and out == "Oslo är huvudstaden i Norge."


async def test_other_brackets_are_not_lookups():
    out, found = await run(["[laughs] ", "that's funny"])
    assert found == [] and "that's funny" in out


def test_marker_variants():
    assert LOOKUP_MARK.match("[lookup: opening hours ICA Maxi]").group(1) == "opening hours ICA Maxi"
    assert LOOKUP_MARK.match("  [ LOOKUP : x y ] rest").group(1).strip() == "x y"


def test_promise_without_marker_is_caught():
    from kiki.server.research import promised_lookup, query_from_question

    assert promised_lookup("Jag kollar! Bra fråga, jag kollar!")
    assert promised_lookup("Good question, let me check.")
    assert not promised_lookup("Oslo är huvudstaden i Norge.")
    assert query_from_question("Kiki, vem vann senaste Melodifestivalen?") == "vem vann senaste Melodifestivalen"
    assert query_from_question("Hej Kiki! What's the weather tomorrow?") == "What's the weather tomorrow"
