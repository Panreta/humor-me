"""Exp 33's harness (sim/tonight.py): fetch with fakes, the filter bar, the pairs, the blind review and the verdict.
Nothing here calls a model or the network."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import generate, tonight as app_tonight  # noqa: E402
from sim import thread_coherence as tc  # noqa: E402
from sim import tonight as ex  # noqa: E402
from tests.test_generate import parallel_client, thread_reply  # noqa: E402
from tests.test_tonight import FakeClient  # noqa: E402

REPORT = {"fetched": 8, "dropped_by_category": {"death": 1, "disaster": 1, "crime": 1, "war": 1}, "dropped_no_gist": 0,
          "rejected_by_name_check": 1, "rejected_by_safety": 0, "kept": 3}
BUILT = {"date": "2026-11-26", "fetched_at": "t", "sources": {"news": "n", "market": "m"}, "errors": [],
         "market": {"line": "The S&P 500 closed up 0.4% at its last close.", "pct": 0.4},
         "gists": ["a chip appears", "a vote happens", "a parade moves"], "report": REPORT}


def scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def saved(tmp_path, answers=("y", "n", "y", "n", "y", "n"), fact=""):
    return ex.fetch(read=scripted(*answers, fact), out=lambda *_: None, directory=tmp_path, build_fn=lambda: BUILT,
                    safety_client=FakeClient())


# --- fetch ---

def test_fetch_asks_about_each_gist_then_for_at_most_one_line_and_saves(tmp_path):
    rec = saved(tmp_path, ("y", "n", "n", "n", "y", "y"), fact="  it's snowing ")
    assert [(a["gist"], a["approved"], a["name"]) for a in rec["answers"]] == [
        ("a chip appears", True, False), ("a vote happens", False, False), ("a parade moves", True, True)]
    assert rec["approved"] == ["a chip appears", "a parade moves"] and rec["facts"] == ["it's snowing"]
    assert ex.read_fetch(tmp_path) == rec


def test_the_typed_line_is_optional_and_safety_checked(tmp_path):
    assert saved(tmp_path)["facts"] == []
    assert saved(tmp_path, fact="the host has a fat head")["facts"] == []


def test_an_unanswerable_reply_is_asked_again():
    asked = []

    def read(prompt):
        asked.append(prompt)
        return ["maybe", "y", "x", "n", ""][len(asked) - 1]
    rec = ex.approve({**BUILT, "gists": ["a chip appears"]}, read, lambda *_: None)
    assert rec["answers"] == [{"gist": "a chip appears", "approved": True, "name": False}] and len(asked) == 5


def test_a_day_with_no_surviving_gist_stops_the_fetch(tmp_path):
    with pytest.raises(SystemExit, match="nothing to approve"):
        ex.fetch(read=scripted(), out=lambda *_: None, directory=tmp_path, build_fn=lambda: {**BUILT, "gists": []})


# --- bar 1 ---

def rec_of(approved_flags, names=None, market=True):
    names = names or [False] * len(approved_flags)
    answers = [{"gist": f"g{i}", "approved": a, "name": n} for i, (a, n) in enumerate(zip(approved_flags, names))]
    return {**BUILT, "answers": answers, "approved": [a["gist"] for a in answers if a["approved"]], "facts": [],
            "market": BUILT["market"] if market else None}


def test_bar1_passes_with_at_most_20_percent_rejected_and_no_name():
    b = ex.bar1(rec_of([True] * 4 + [False]))
    assert b["reject_share"] == 0.2 and b["passes"] and b["kept"] == 5 and b["approved"] == 4
    assert b["reject_wilson95"][0] < 0.2 < b["reject_wilson95"][1]


@pytest.mark.parametrize("flags,names", [([True] * 3 + [False] * 2, None), ([True] * 5, [False, False, True, False, False]), ([], None)])
def test_bar1_fails_on_too_many_rejections_any_name_or_nothing_kept(flags, names):
    assert ex.bar1(rec_of(flags, names))["passes"] is False


def test_wilson_matches_a_known_value():
    lo, hi = ex.wilson(1, 5)
    assert (round(lo, 3), round(hi, 3)) == (0.036, 0.624) and ex.wilson(0, 0) is None


# --- the pairs ---

def test_pairs_cycle_the_approved_gists_with_a_seeded_mode_and_room():
    rec = rec_of([True, True, True])
    s = ex.slots(rec)
    assert len(s) == 10 and [p["gist"] for p in s[:4]] == ["g0", "g1", "g2", "g0"]
    assert s == ex.slots(rec) and all(p["mode"] in generate.MODES and set(p["context"]) == {"venue", "energy", "age"} for p in s)


def test_no_approved_gist_means_no_run():
    with pytest.raises(SystemExit, match="approved no gists"):
        ex.slots(rec_of([False, False]))


def test_the_digest_follows_what_the_run_is_made_of():
    rec = rec_of([True, True])
    d = ex.digest(rec)
    assert d == ex.digest(rec) and d != ex.digest({**rec, "facts": ["it's snowing"]}) and d != ex.digest(rec_of([True, True], market=False))
    assert d != ex.digest(rec_of([True, True, True]))


def test_arm_b_gets_the_headline_the_fact_and_the_market_and_arm_a_none_of_them():
    rec = {**rec_of([True]), "facts": ["it's snowing"]}
    slot = ex.slots(rec)[0]
    for arm in ex.ARMS:
        c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
        out = ex.write_bit(c, rec, slot, arm)
        system, user = (m["content"] for m in seen[0]["messages"])
        user = json.loads(user)
        assert out["premise"] == "p" and len(out["lines"]) == 5
        if arm == "A":
            assert "topic" not in user and "tonight" not in user and app_tonight.PARAGRAPH not in system and out["gist"] is None
        else:
            assert user["topic"] == "g0" and user["tonight"] == {"facts": ["it's snowing"], "market": BUILT["market"]["line"]}
            assert system.endswith(app_tonight.PARAGRAPH) and out["gist"] == "g0"


def test_bits_are_written_once_and_a_resumed_run_reads_them_back(tmp_path):
    rec = rec_of([True, True])
    pairs_ = ex.slots(rec, n=2)
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    first = ex.generate_bits(c, rec, pairs_, tmp_path, http=lambda url: {})
    calls = c.attempts
    assert calls == 4 * 3 and len(first) == 4 and ex.gather(tmp_path, pairs_) == first
    assert ex.generate_bits(c, rec, pairs_, tmp_path, http=lambda url: {}) == first and c.attempts == calls


# --- review and the verdict ---

def bit(pair, arm, premise="p"):
    return {"pair": pair, "arm": arm, "mode": "cynic", "gist": None, "premise": premise, "lines": [f"{arm} {pair} {i}" for i in range(5)],
            "lost": [], "failure": None, "wall_s": 1.0, "bad_replies": 0}


def records(n=10, missing=()):
    return [bit(f"p{i:02d}", a, None if (f"p{i:02d}", a) in missing else "p") for i in range(n) for a in ex.ARMS]


def test_a_pair_where_an_arm_delivered_nothing_is_not_reviewed():
    drawn = ex.pairs_to_review(records(missing={("p03", "B")}))
    assert len(drawn) == 9 and "p03" not in {d["pair"] for d in drawn}
    assert drawn == ex.pairs_to_review(records(missing={("p03", "B")})) and {d["left"] for d in drawn} == {"A", "B"}


def test_review_hides_the_arm_saves_each_answer_and_resumes(tmp_path):
    recs = records(3)
    shown_text = []
    answers = iter(["l", "r", "r", "t", "q"])
    done = ex.review(tmp_path, recs, read=lambda p: next(answers), out=shown_text.append)
    assert done == 2 and len(ex.read_answers(tmp_path)) == 2
    assert all("gist" not in t and "Arm" not in t for t in shown_text)
    rest = iter(["l", "l"])
    assert ex.review(tmp_path, recs, read=lambda p: next(rest), out=lambda *_: None) == 3
    assert ex.review(tmp_path, recs, read=lambda p: pytest.fail("asked again"), out=lambda *_: None) == 3


def answers_for(recs, funnier_b_wins, tonight_b_wins=0, ties=0):
    """Answers that give B exactly the asked wins on (a)."""
    drawn = ex.pairs_to_review(recs)
    out = {}
    for k, p in enumerate(drawn):
        b_side = "l" if p["left"] == "B" else "r"
        a_side = "r" if b_side == "l" else "l"
        f = "t" if k >= len(drawn) - ties else (b_side if k < funnier_b_wins else a_side)
        out[p["pair"]] = {"funnier": f, "tonight": b_side if k < tonight_b_wins else a_side}
    return out


def test_bar2_is_five_wins_of_ten_and_a_tie_is_not_a_win():
    recs = records()
    m = ex.metric2(recs, answers_for(recs, 5))
    assert m["funnier"]["B_wins"] == 5 and m["funnier"]["passes"] and m["complete"]
    m = ex.metric2(recs, answers_for(recs, 4))
    assert m["funnier"]["passes"] is False
    m = ex.metric2(recs, answers_for(recs, 5, ties=1))
    assert m["funnier"]["B_wins"] == 5 and m["funnier"]["ties"] == 1
    m = ex.metric2(recs, answers_for(recs, 4, ties=1))
    assert m["funnier"]["B_wins"] == 4 and m["funnier"]["passes"] is False
    assert ex.metric2(recs, answers_for(recs, 10, tonight_b_wins=7))["tonight"]["B_share"] == 0.7


def test_the_verdict_follows_both_bars():
    recs, good, bad = records(), rec_of([True] * 5), rec_of([True] * 2 + [False] * 3)
    assert ex.summarize(good, recs, answers_for(recs, 6))["verdict"] == "adopt"
    assert ex.summarize(good, recs, answers_for(recs, 3))["verdict"] == "keep_off"
    assert ex.summarize(good, recs, {})["verdict"] == "pending_review"
    assert ex.summarize(bad, recs, answers_for(recs, 10))["verdict"] == "keep_off"  # the pairs cannot rescue the filter
    four_of_five = dict(list(answers_for(recs, 4).items())[:5])
    assert ex.summarize(good, recs, four_of_five)["verdict"] == "pending_review"
    five_of_five = dict(list(answers_for(recs, 10).items())[:5])
    assert ex.summarize(good, recs, five_of_five)["verdict"] == "adopt"  # 5 wins cannot be undone by the pairs left


def test_score_reads_the_run_folder_and_the_table_prints_every_row(tmp_path):
    rec = {**rec_of([True] * 5), "facts": []}
    tc._write(ex.fetch_path(tmp_path), rec)
    with pytest.raises(SystemExit, match="not all bits"):
        ex.score(tmp_path)
    directory = ex.run_dir(rec, root=tmp_path)
    for s in ex.slots(rec):
        for a in ex.ARMS:
            tc._write(ex.bit_path(directory, s["id"], a), bit(s["id"], a))
    recs = ex.gather(directory, ex.slots(rec))
    tc._write(ex.review_path(directory), {"answers": answers_for(recs, 6, tonight_b_wins=4)})
    res = ex.score(tmp_path)
    assert res["verdict"] == "adopt" and res["bar2_pairs"]["funnier"]["B_wins"] == 6
    text = ex.table(res)
    assert len(text.splitlines()) == 10 and "Wilson" in text and "adopt" in text
    json.dumps(res)  # a record is plain JSON
