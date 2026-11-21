from kiki.server.lang import detect


def test_detects_swedish_and_english():
    assert detect("Hej! Huvudstaden i Sverige är Stockholm.") == "sv"
    assert detect("Det finns väldigt många äpplen i trädgården i år") == "sv"
    assert detect("Kiki, vad är klockan?") == "sv"
    assert detect("Kiki, what time is it?") == "en"
    assert detect("Most Thai restaurants do take reservations.") == "en"
    assert detect("Stockholm.") == "en"  # no evidence -> default
    assert detect("Kiki, vem vann senaste Melodifestivalen?") == "sv"
    assert detect("Kiki, vad kostar en biljett till Liseberg i år?") == "sv"
    assert detect("Kiki, who won the last Eurovision?") == "en"
