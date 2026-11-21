"""Floor policy on top of fixed gate verdicts (no LLM needed)."""

import time

from kiki.server.brain import Brain, is_pass
from kiki.server.gate import Verdict


def brain(*lines):
    b = Brain(llm=None, gate=None)
    for line in lines:
        who, text = line[1:].split("]: ", 1)
        (b.said(text) if who == "Kiki" else b.hear(who, text))
    return b


def v(action, p=1.0):
    return Verdict(action, {action: p})


def test_unnamed_reply_needs_engagement_and_confidence():
    b = brain("[Anna]: Who called you earlier?", "[Bob]: Hmm, I can't remember.")
    assert b.apply(v("reply", 0.95), "Bob", named=False).action == "silent"  # Kiki never spoke
    b = brain("[Folke]: Kiki, capital of Sweden?", "[Kiki]: Stockholm.", "[Folke]: And Finland?")
    assert b.apply(v("reply", 0.95), "Folke", named=False).action == "reply"
    assert b.apply(v("reply", 0.75), "Folke", named=False).action == "silent"


def test_quiet_mode_blocks_chimes_and_follow_ups_but_not_the_name():
    b = brain("[Bob]: Kiki, be quiet.")
    assert b.apply(v("quiet"), "Bob", named=True).action == "quiet"
    assert b.quiet
    b.hear("Anna", "Who wrote Hamlet?")
    assert b.apply(v("chime"), "Anna", named=False).action == "silent"
    b.hear("Anna", "Kiki, who wrote Hamlet?")
    assert b.apply(v("reply"), "Anna", named=True).action == "reply"
    assert not b.quiet


def test_chime_threshold_cooldown_and_proactivity_off():
    b = brain("[Anna]: Tallest mountain in Africa?", "[Bob]: No idea.")
    assert b.apply(v("chime", 0.5), "Bob", named=False).action == "silent"
    assert b.apply(v("chime", 0.9), "Bob", named=False).action == "chime"
    b.said("Kilimanjaro.", chime=True)
    assert b.apply(v("chime", 0.9), "Bob", named=False).action == "silent"  # cooldown
    b.last_chime_t = time.time() - 1000
    b.proactivity = 0
    assert b.apply(v("chime", 1.0), "Bob", named=False).action == "silent"


def test_is_pass():
    assert is_pass("PASS") and is_pass(" pass.") and is_pass("(PASS)")
    assert not is_pass("Passau is in Germany.")


def test_unnamed_follow_up_must_come_from_who_kiki_answered():
    b = brain("[Carl]: Kiki, twelve times fifteen?")
    b.said("180.", target="Carl")
    b.hear("Anna", "Did you feed the cat?")
    b.hear("Bob", "Yes, I did.")
    assert b.apply(v("reply", 0.99), "Bob", named=False).action == "silent"
    b.hear("Carl", "And thirteen times fifteen?")
    assert b.apply(v("reply", 0.99), "Carl", named=False).action == "reply"
