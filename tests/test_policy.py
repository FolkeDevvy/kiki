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


def test_misheard_name_needs_a_confident_gate():
    b = brain("[Sara]: Kik i hur länge ska man koka ett ägg?")
    b._maybe_named = True
    assert b.apply(v("reply", 0.95), "Sara", named=False).action == "reply"
    assert b.apply(v("reply", 0.6), "Sara", named=False).action == "silent"
    b._maybe_named = True  # "Kids are asleep": the gate says silent, so silent
    assert b.apply(v("silent", 1.0), "Sara", named=False).action == "silent"


def test_contextual_mode_sends_plausible_lines_to_the_responder():
    b = brain("[Folke]: Frågan när brukar första snön komma?")
    assert b.apply(v("reply", 1.0), "Folke", named=False).action == "silent"  # local brain: no
    b.allow_unsure = True
    d = b.apply(v("reply", 1.0), "Folke", named=False)
    assert d.action == "reply" and d.unsure
    assert b.apply(v("reply", 0.4), "Folke", named=False).unsure  # plausible enough
    b.hear("Folke", "Jag har köpt nya skor.")
    assert b.apply(v("silent", 1.0), "Folke", named=False).action == "silent"  # clear table talk
    b.hear("Folke", "Har någon sett min mobil?")
    assert b.apply(v("silent", 1.0), "Folke", named=False).unsure  # a question: Claude decides
    assert "PASS" in b.claude_message(unsure=True)


def test_contextual_mode_follow_up_without_name_and_quiet_mode():
    b = brain("[Anna]: Kiki, capital of Sweden?")
    b.allow_unsure = True
    b.said("Stockholm.", target="Anna")
    b.hear("Anna", "And Norway?")
    from kiki.server.gate import Verdict
    d = b.apply(Verdict("silent", {"silent": 0.8, "reply": 0.2}), "Anna", named=False)
    assert d.action == "reply" and d.unsure  # recent exchange: let the responder judge
    b.quiet_until = 10**12
    assert b.apply(v("reply", 1.0), "Anna", named=False).action == "silent"
    assert b.apply(v("reply", 1.0), "Anna", named=True).action == "reply"


def test_contextual_mode_names_and_stuck_moments_go_to_claude():
    b = brain("[Anna]: Kiki är faktiskt ganska smart.")
    b.allow_unsure = True
    d = b.apply(v("silent", 1.0), "Anna", named=True)
    assert d.action == "reply" and d.unsure  # a mention: Claude decides, usually PASS
    b = brain("[Anna]: How far is it to Gothenburg?", "[Bob]: I don't know, far.")
    b.allow_unsure = True
    assert b.apply(v("silent", 1.0), "Bob", named=False).action == "chime"
    b = brain("[Anna]: Kiki, dinner idea?")
    b.allow_unsure = True
    b.said("Pasta? Do you have tomatoes?", target="Anna")
    b.hear("Anna", "Yeah, we have some.")
    assert b.apply(v("silent", 1.0), "Anna", named=False).unsure  # answering Kiki's question


def test_gate_window_moves_in_steps_so_the_cache_survives():
    b = Brain(llm=None, gate=None)
    starts = []
    for i in range(60):
        b.hear("Anna", f"line {i}")
        starts.append(b.gate_transcript().split("\n")[0])
    changes = sum(1 for a, c in zip(starts, starts[1:]) if a != c)
    assert changes <= 60 // 8 + 1  # the prefix changes once per 8 lines, not every line
    assert len(b.gate_transcript().split("\n")) <= 20


async def test_gate_failure_in_claude_mode_uses_context():
    class Broken:
        async def judge(self, *a, **k):
            raise TimeoutError

    b = Brain(llm=None, gate=Broken())
    b.allow_unsure = True
    b.hear("Folke", "Do you think I should buy a million dollars for a million dollars?")
    d = await b.decide("Folke", "Do you think I should buy a million dollars for a million dollars?")
    assert d.action == "reply" and d.unsure  # a question: Claude still gets to decide
