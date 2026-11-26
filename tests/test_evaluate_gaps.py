"""Exp 31 live check (sim/evaluate_gaps.py): waits, zero-wait gaps, filler lines and guesses from synthetic rows."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from sim import evaluate_gaps as eg  # noqa: E402


def rows(waits, bridges=0, guesses=()):
    """A show whose bit i (from 1) waits `waits[i-1]` s after the previous bit ends; bit 0 waits 5 s."""
    out = [{"event": "show_start", "at": 0.0, "room": "sim", "generator_id": "g"}]
    end = 0.0
    for bit, wait in enumerate([5.0, *waits]):
        began = end
        out.append({"event": "joke_played", "bit": bit, "audio_start_at": began + wait})
        end = began + wait + 10.0
        out.append({"event": "bit_end", "bit": bit, "at": end})
    out += [{"event": "bridge", "bit": 1} for _ in range(bridges)]
    out += [{"event": "prediction", "bit": b, "hit": h, "learned": True, "adopted": h, "spec_prefetched": True}
            for b, h in guesses]
    out += [{"event": "show_end", "at": end, "status": "finished"}, {"event": "rating", "value": 3}]
    return out


def test_waits_zero_gaps_and_filler_are_counted_for_bits_one_and_up():
    s = eg.show_report("x", rows([0.0, 12.0, 0.05], bridges=4))
    assert s["gaps"] == [0.0, 12.0, 0.05]  # bit 0's 5 s is left out
    assert s["zero_waits"] == 2 and s["bridge_lines"] == 4 and s["rating"] == 3 and s["status"] == "finished"


def test_guess_counts_and_wasted_safety_requests():
    g = eg.show_report("x", rows([0.0, 0.0, 9.0], guesses=[(1, True), (2, True), (3, False)]))["guesses"]
    assert (g["n"], g["hits"], g["adopted"], g["wasted_prefetch"]) == (3, 2, 2, 1)


def test_pool_adds_shows_and_takes_the_median_over_all_gaps():
    a, b = eg.show_report("a", rows([1.0, 2.0])), eg.show_report("b", rows([3.0, 4.0], bridges=2))
    p = eg.pool([a, b])
    assert p["gaps"] == 4 and p["wait_s"]["p50"] == 2.5 and p["bridge_lines"] == 2


def test_a_wait_overlapping_a_pause_or_a_failed_first_render_is_dropped():
    base = rows([12.0, 12.0, 12.0])
    base += [{"event": "pause", "at": 20.0, "paused_at": 20.0}, {"event": "resume", "at": 22.0, "resumed_at": 22.0}]  # inside bit 1's wait (15 to 27 s)
    base.append({"event": "prep", "bit": 3, "first_render_failed": True, "generate_s": 1, "safety_s": 1, "render_wait_s": 1})
    s = eg.show_report("x", base)
    assert len(s["gaps"]) == 1 and s["total_wait_s"] == 12.0  # bits 1 (paused) and 3 (failed) are gone


def test_a_show_that_did_not_finish_is_listed_and_counted_nowhere(tmp_path):
    import json
    good, bad = rows([1.0, 2.0]), rows([3.0])
    good[0].update(schema=1, session="a"); bad[0].update(schema=1, session="b")
    bad[-2]["status"] = "aborted"
    res = {s: eg.show_report(s, r) for s, r in (("a", good), ("b", bad))}
    assert res["b"]["status"] == "aborted"
    shown = eg.load  # the filter itself lives in run(); check it through a stub of load
    eg.load = lambda d, ids: ([("a", good), ("b", bad)], [])
    try:
        out = eg.run(["a"], ["b"], tmp_path)
    finally:
        eg.load = shown
    assert out["before"] and not out["after"] and out["excluded"] == [{"session": "b", "status": "aborted"}]
    assert out["after_pooled"]["hit_ci95"] is None and out["after_pooled"]["shows"] == 0


def test_wilson_interval_matches_the_known_value_and_total_wait_is_per_show():
    lo, hi = eg.wilson(17, 21)
    assert (round(lo, 2), round(hi, 2)) == (0.60, 0.92) and eg.wilson(0, 0) is None
    p = eg.pool([eg.show_report("a", rows([1.0, 2.0])), eg.show_report("b", rows([3.0, 4.0]))])
    assert p["total_wait_per_show_s"] == 5.0
