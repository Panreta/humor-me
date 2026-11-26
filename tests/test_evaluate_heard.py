"""Exp 28 (D-112): the heard-bridges counts on synthetic logs. Counts only, never text."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import cache as live_cache  # noqa: E402
from app.generate import BRIDGES  # noqa: E402
from app.log import SessionLog  # noqa: E402
from sim import evaluate_heard as eh  # noqa: E402
from tests.test_cached_bits import CTX, START, frozen_time, lines, live_show  # noqa: E402,F401


def replay_show(directory, now, at, bridges, room="sim"):
    """A show that played only bridge rows: (text, kind) pairs."""
    now[0] = at
    log = SessionLog(directory)
    log.write("show_start", room=room, **{**START, "context": CTX})
    for text, kind in bridges:
        extra = {"kind": "cached", "source": {"session": "x", "bit": 0, "joke": 0, "mode": "neurotic"}} if kind == "cached" else {"kind": "filler"}
        log.write("bridge", bit=1, text=text, tts_seconds=0.1, audio_start_at=1.0, audio_end_at=2.0, **extra)
    log.write("show_end", status="finished")


def test_counts_are_per_show_against_earlier_shows_only_and_hold_no_text(tmp_path, frozen_time):
    live_show(tmp_path, [("neurotic", [3] * 5, lines("top"))], at=1000.0, now=frozen_time)
    replay_show(tmp_path, frozen_time, 2000.0, [("TOP line 0", "cached"), ("top  line 1", "cached"), (BRIDGES[0], "filler"),
                                                (BRIDGES[1], "filler"), (BRIDGES[1], "filler")])
    replay_show(tmp_path, frozen_time, 3000.0, [("a fresh line", "cached"), (BRIDGES[0], "filler")])
    res = eh.run(tmp_path)
    live, second, third = res["shows"]
    assert (live["bridges"], live["jokes"], live["jokes_heard_before"]) == (0, 5, 0)
    assert (second["cached"], second["cached_heard_before"]) == (2, 2)  # re-rendered spacing and case still count
    assert (second["filler"], second["filler_heard_before"], second["filler_repeated_in_show"]) == (3, 0, 1)
    assert (third["cached"], third["cached_heard_before"], third["filler_heard_before"]) == (1, 0, 1)  # BRIDGES[0] was in show 2
    assert res["filler_lines_heard"] == 2 and res["n_filler_lines"] == len(BRIDGES)
    assert res["worst_cached_heard_share"] == 1.0 and res["rule_needed"]
    text = json.dumps(res)
    assert not any(t in text for t in (*lines("top"), "a fresh line", BRIDGES[0], BRIDGES[1]))  # numbers and ids only


def test_the_rule_never_leaves_a_heard_line_in_a_pool_and_the_unruled_pool_does(tmp_path, frozen_time):
    live_show(tmp_path, [("neurotic", [3] * 5, lines("top"))], at=1000.0, now=frozen_time)
    replay_show(tmp_path, frozen_time, 2000.0, [])
    replay_show(tmp_path, frozen_time, 3000.0, [])
    res = eh.run(tmp_path)
    assert res["rule_pool_heard_lines"] == 0
    assert [s["pool_lines_without_rule"] for s in res["shows"]] == [0, 5, 5]  # the source show's lines replay without it
    assert [s["pool_lines_with_rule"] for s in res["shows"]] == [0, 0, 0]


def test_no_cached_bridge_means_no_verdict(tmp_path, frozen_time):
    live_show(tmp_path, [("neurotic", [3] * 5, lines("top"))], at=1000.0, now=frozen_time)
    res = eh.run(tmp_path)
    assert res["worst_cached_heard_share"] is None and not res["rule_needed"]
    assert "NOT DECIDED" in eh.verdict(res)
