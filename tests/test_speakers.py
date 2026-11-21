import numpy as np

from kiki.server.speakers import Registry


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_new_voices_become_guests_and_are_recognised_again():
    r = Registry(threshold=0.7)
    a, b = unit([1, 0, 0]), unit([0, 1, 0])
    assert r.assign(a, 2.0)[0] == "Guest-1"
    assert r.assign(b, 2.0)[0] == "Guest-2"
    assert r.assign(unit([0.95, 0.1, 0]), 2.0)[0] == "Guest-1"


def test_short_segments_never_create_voices():
    r = Registry(threshold=0.7)
    assert r.assign(unit([1, 0, 0]), 0.4)[0] == "?"
    assert r.voices == []
    assert r.assign(unit([1, 0, 0]), 0.4, fallback="Anna")[0] == "Anna"


def test_enroll_persists_and_rename(tmp_path):
    p = tmp_path / "speakers.json"
    r = Registry.load(p)
    r.enroll("Folke", unit([1, 0, 0]))
    g = r.assign(unit([0, 0, 1]), 2.0)[0]
    assert r.rename(g, "Anna")
    r2 = Registry.load(p)
    assert sorted(v.name for v in r2.voices) == ["Anna", "Folke"]


def test_long_ambiguous_segment_starts_new_guest_and_drifted_guests_merge():
    r = Registry(threshold=0.7, new_voice_margin=0.12)
    a = unit([1, 0, 0])
    assert r.assign(a, 3.0)[0] == "Guest-1"
    amb = unit([0.65, np.sqrt(1 - 0.65**2), 0])
    assert r.assign(amb, 1.5)[0] == "Guest-1"  # short + ambiguous: no new voice
    assert r.assign(amb, 3.0)[0] == "Guest-2"  # long + ambiguous: new guest
    # Guest-2 keeps hearing voices that pull it onto Guest-1 -> merged
    for _ in range(10):
        name, _ = r.assign(unit([0.97, 0.24, 0]), 3.0)
    assert [v.name for v in r.voices] == ["Guest-1"]
    assert r.merged == {"Guest-2": "Guest-1"}
