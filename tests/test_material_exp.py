"""Exp 32 (D-116): the bookkeeping of the material experiment, with a fake Ollama and a fake judge, no model."""
import json
import pathlib
import sys
import threading

import httpx
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import generate, llm  # noqa: E402
from sim import material as m32  # noqa: E402
from sim import thread_coherence as tc  # noqa: E402


class FakeOllama:
    """Arm A (no topic in the request) writes about the battery; arm B writes about the topic it was given."""

    def __init__(self):
        self.seen, self.lock, self.n = [], threading.Lock(), 0

    def __call__(self, request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][1]["content"])
        with self.lock:
            self.seen.append((body, user))
            self.n += 1
            n = self.n
        about = user.get("topic", "my battery")
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"premise": f"premise {n} on {about}", "jokes": [f"{about} line {j} number {n}." for j in range(5)]})},
            "finish_reason": "stop"}]})

    def client(self):
        return llm.Client("http://local/v1", "", "bonsai-2:4b", 10.0, retries=2, deadline_s=12.0, local=True,
                          extra_body=llm.OLLAMA_NO_THINK, json_schema=True, transport=httpx.MockTransport(self))


class FakeJudge:
    """hardware iff the shown text says battery."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def chat(self, system, user, **kw):
        self.calls.append((system, user, kw))
        if self.fail:
            raise llm.LLMError("http 503", 503)
        return json.dumps({"hardware": "battery" in user})


def ps(share=1.0, context=8192):
    return lambda url: {"models": [{"name": "bonsai-2:4b", "model": "bonsai-2:4b", "size": 100,
                                    "size_vram": int(100 * share), "context_length": context}]}


def small(tmp_path, shows=2, bits=3):
    fake = FakeOllama()
    s = m32.slots(shows, bits)
    recs = m32.generate_bits(fake.client(), s, tmp_path, http=ps())
    return fake, s, recs


def rec(slot, arm, hw, topic=None, kind=None, premise=None):
    premise = premise or f"p {slot} {arm}"
    text = "battery" if hw else "garden"
    return {"slot": slot, "arm": arm, "mode": "cynic", "premise": premise, "lines": [f"{text} {i}" for i in range(5)],
            "topic": topic, "from": kind, "facts": [], "recycled": False if topic else None, "lost": [], "failure": None,
            "wall_s": 1.0, "bad_replies": 0}


# --- the slots and digests ---

def test_the_slots_are_seeded_the_same_for_both_arms_and_a_show_has_one_room():
    s = m32.slots()
    assert s == m32.slots() and len(s) == 40 and s != m32.slots(seed="other")
    assert [x["bit"] for x in s[:8]] == list(range(8)) and len({json.dumps(x["context"]) for x in s[:8]}) == 1
    assert {x["mode"] for x in s} <= set(generate.MODES) and len({x["mode"] for x in s}) > 2


def test_the_digest_follows_the_prompt_the_material_files_and_the_paragraph_and_a_stale_call_is_refused(monkeypatch):
    before, jbefore = m32.digest(), m32.judge_digest()
    with monkeypatch.context() as m:
        m.setattr(generate, "MATERIAL_PARAGRAPH", generate.MATERIAL_PARAGRAPH + " x")
        assert m32.digest() != before
    with monkeypatch.context() as m:
        m.setattr(generate, "PROMPT_ID", "gen-v5")
        assert m32.digest() != before
    with monkeypatch.context() as m:
        m.setattr(m32, "file_hash", lambda name: "changed")
        assert m32.digest() != before
    monkeypatch.setattr(m32, "HARDWARE_JUDGE", m32.HARDWARE_JUDGE + " x")
    assert m32.judge_digest() != jbefore and m32.digest() == before
    with pytest.raises(AssertionError):
        m32.run(digest_=before, judge_digest_=jbefore)


# --- writing ---

def test_arm_a_has_no_material_arm_b_gets_a_topic_that_no_earlier_show_of_the_run_used(tmp_path):
    fake, s, recs = small(tmp_path, shows=3, bits=4)
    a = [r for r in recs if r["arm"] == "A"]
    b = [r for r in recs if r["arm"] == "B"]
    assert all(r["topic"] is None for r in a) and all(r["topic"] and len(r["facts"]) == 2 for r in b)
    for show in range(3):
        mine = {r["topic"] for r, x in zip(b, s) if x["show"] == show}
        earlier = {r["topic"] for r, x in zip(b, s) if x["show"] < show}
        assert len(mine) == 4 and not mine & earlier
    assert [r["from"] for r in b[:4]] == ["everyday", "character", "everyday", "everyday"]
    # the request carries the live prompt with the paragraph only for B
    asks = [(body["messages"][0]["content"], user) for body, user in fake.seen[1:]]
    assert sum(generate.MATERIAL_PARAGRAPH in sys for sys, _ in asks) == 3 * 4 * 3
    assert sum("topic" in user for _, user in asks) == 3 * 4 * 3


def test_a_finished_bit_is_read_back_and_a_resumed_run_draws_what_an_uninterrupted_one_would(tmp_path, monkeypatch):
    fake, s, first = small(tmp_path)
    n = len(fake.seen)
    monkeypatch.setattr(generate, "warm_up", lambda *a, **k: pytest.fail("warm-up with nothing to write"))
    assert m32.generate_bits(fake.client(), s, tmp_path) == first and len(fake.seen) == n
    # drop the last bit of arm B and rewrite it: the same topic comes back
    gone = tmp_path / "bits" / f"{s[-1]['id']}-B.json"
    topic = json.loads(gone.read_text())["topic"]
    gone.unlink()
    monkeypatch.undo()
    again = m32.generate_bits(FakeOllama().client(), s, tmp_path, http=ps())
    assert again[-1]["topic"] == topic and again[:-1] == first[:-1]
    assert m32.gather(tmp_path, s) == again and m32.gather(tmp_path, m32.slots(3, 3)) is None


def test_a_bit_that_fails_is_a_result_with_reasons_and_no_text():
    c = llm.Client("http://local/v1", "", "m", 5.0, retries=2, deadline_s=5.0, local=True, json_schema=True,
                   transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {
                       "content": "not json"}, "finish_reason": "stop"}]})))
    slot = m32.slots(1, 1)[0]
    r = m32.write_bit(c, m32.new_generator(c, "B", 0, {}, "x"), slot, "B")
    assert r["premise"] is None and r["failure"] == "GenerationError" and len(r["lost"]) == 3 and r["topic"] is None


# --- labeling ---

def test_the_judge_is_blind_plain_and_cached_by_text_and_an_outage_is_not_cached(tmp_path):
    r = rec("s0b0", "A", True, premise="It’s a “plan”")
    j = FakeJudge()
    assert m32.label(j, r, tmp_path) is True and m32.label(j, r, tmp_path) is True and len(j.calls) == 1
    system, user, kw = j.calls[0]
    assert user.startswith("Premise: It's a \"plan\"") and kw["json_reply"] == m32.JUDGE_SCHEMA
    for word in ("arm a", "arm b", "topic list", "character sheet", "gen-v", "material on"):
        assert word not in (system + user).lower()
    assert m32.label(FakeJudge(fail=True), rec("s0b1", "A", True), tmp_path) is None
    assert not (tmp_path / "judge_cache" / m32.judge_digest() / f"{m32.bit_hash(rec('s0b1', 'A', True))}.json").exists()
    for bad in ("nope", '{"hardware": "yes"}', "{}"):
        with pytest.raises(Exception):
            m32.parse_verdict(bad)


def test_the_word_list_cross_check_finds_the_hardware_words():
    assert m32.regex_hardware(rec("x", "A", True)) and not m32.regex_hardware(rec("x", "A", False))
    assert m32.regex_hardware({**rec("x", "A", False), "premise": "my processor top"})


# --- metric 1 ---

def records(a_hw, b_hw, kinds=None):
    out = []
    for i, (a, b) in enumerate(zip(a_hw, b_hw)):
        kind = (kinds or {}).get(i, "everyday")
        out += [rec(f"s0b{i}", "A", a), rec(f"s0b{i}", "B", b, topic=f"t{i}", kind=kind)]
    return out


def verdicts_of(recs):
    return {m32.bit_hash(r): "battery" in r["lines"][0] for r in recs if r["premise"] is not None}


def test_the_repetition_bar_needs_b_at_most_15_percent_and_the_ci_entirely_below_zero():
    n = 40
    good = records([True] * 12 + [False] * 28, [True] * 3 + [False] * 37)
    r = m32.metric1(good, verdicts_of(good))
    assert r["share"]["A"]["share"] == 0.3 and r["share"]["B"]["share"] == 3 / 40 and r["bar"]["passes"]
    assert r["bar"]["ci95"][1] < 0
    high = records([True] * 30 + [False] * 10, [True] * 7 + [False] * 33)  # B 17.5%: the CI is below 0, the share is not
    r = m32.metric1(high, verdicts_of(high))
    assert r["bar"]["ci_below_zero"] and not r["bar"]["share_ok"] and not r["bar"]["passes"]
    same = records([True] * 4 + [False] * 36, [True] * 4 + [False] * 36)  # B under 15% but no better than A
    r = m32.metric1(same, verdicts_of(same))
    assert r["bar"]["share_ok"] and not r["bar"]["ci_below_zero"] and not r["bar"]["passes"]
    assert n == len(good) // 2


def test_a_bit_that_failed_or_went_unjudged_drops_out_of_the_pairing_not_the_other_arms_share():
    recs = records([True, True, False, False], [False, False, False, False])
    recs[1] = {**recs[1], "premise": None, "lines": []}  # B's bit 0 never delivered
    v = verdicts_of(recs)
    v[m32.bit_hash(recs[2])] = None  # the judge could not answer A's bit 1
    r = m32.metric1(recs, v)
    assert r["share"]["B"]["n"] == 3 and r["share"]["A"]["n"] == 3 and r["paired_slots"] == ["s0b2", "s0b3"]


def test_the_hardware_share_is_split_by_character_and_everyday_topic_bits_with_no_bar():
    recs = records([True, True, False, False], [True, False, False, False], kinds={1: "character", 3: "character"})
    r = m32.metric1(recs, verdicts_of(recs))["by_topic_kind"]
    assert r["B_character"] == {"n": 2, "hardware": 0, "share": 0.0} and r["B_everyday"]["hardware"] == 1
    assert r["A_character"]["hardware"] == 1 and r["A_everyday"]["hardware"] == 1  # the same slots in arm A


# --- metric 2 ---

def test_the_pairs_are_a_fixed_seeded_draw_with_randomized_sides_and_only_slots_both_arms_delivered():
    recs = records([False] * 30, [False] * 30)
    recs[1] = {**recs[1], "premise": None}
    p = m32.pairs(recs)
    assert p == m32.pairs(recs) and len(p) == 20 and len({x["slot"] for x in p}) == 20 and "s0b0" not in {x["slot"] for x in p}
    assert {x["left"] for x in p} == {"A", "B"}


def test_the_review_hides_the_arm_saves_each_answer_and_resumes(tmp_path):
    recs = records([False] * 30, [False] * 30)
    shown, replies = [], iter(["l", "r", "t", "r", "q"])
    done = m32.review(tmp_path, recs, read=lambda prompt: next(replies), out=shown.append)
    assert done == 2  # pair 1 (a, b) answered; pair 2 stopped at (b)... a was l, b was r; then t, r; then q at pair 3
    text = "\n".join(shown)
    assert "LEFT" in text and "RIGHT" in text and "arm" not in text.lower() and "topic" not in text.lower()
    saved = m32.read_answers(tmp_path)
    assert len(saved) == 2 and all(set(v) == {"funnier", "character"} for v in saved.values())
    more = m32.review(tmp_path, recs, read=lambda prompt: "l", out=lambda *_: None)
    assert more == 20 and len(m32.read_answers(tmp_path)) == 20


def answers_for(recs, funny, char):
    """funny/char: for pair i, 'B' (B wins), 'A', or 't'; translated to the side it was shown on."""
    out = {}
    for i, p in enumerate(m32.pairs(recs)):
        side = lambda who: "t" if who == "t" else ("l" if p["left"] == who else "r")  # noqa: E731
        out[p["slot"]] = {"funnier": side(funny[i]), "character": side(char[i])}
    return out


def test_b_needs_10_of_20_funnier_and_a_tie_is_not_a_win_the_character_answer_gets_a_wilson_interval():
    recs = records([False] * 30, [False] * 30)
    nine = m32.metric2(recs, answers_for(recs, ["B"] * 9 + ["t"] * 2 + ["A"] * 9, ["B"] * 15 + ["A"] * 5))
    ten = m32.metric2(recs, answers_for(recs, ["B"] * 10 + ["A"] * 10, ["B"] * 15 + ["A"] * 5))
    assert not nine["funnier"]["passes"] and nine["funnier"]["ties"] == 2 and ten["funnier"]["passes"] and ten["complete"]
    lo, hi = ten["character"]["wilson95"]
    assert ten["character"]["B_wins"] == 15 and lo < 0.75 < hi and 0.5 < lo and hi < 0.95
    assert m32.wilson(0, 0) is None and m32.wilson(0, 20)[0] == 0.0
    assert m32.metric2(recs, {})["answered"] == 0 and not m32.metric2(recs, {})["complete"]


def test_adopt_only_when_both_bars_pass_and_the_pairs_cannot_rescue_a_failed_repetition_bar():
    good = records([True] * 12 + [False] * 18, [False] * 30)
    bad = records([True] * 12 + [False] * 18, [True] * 12 + [False] * 18)
    win = answers_for(good, ["B"] * 12 + ["A"] * 8, ["B"] * 20)
    lose = answers_for(good, ["B"] * 8 + ["A"] * 12, ["B"] * 20)
    m1g, m1b = m32.metric1(good, verdicts_of(good)), m32.metric1(bad, verdicts_of(bad))
    assert m32.decide(m1g, m32.metric2(good, win)) == "adopt"
    assert m32.decide(m1g, m32.metric2(good, lose)) == "keep_off"
    assert m32.decide(m1g, m32.metric2(good, {})) == "pending_review"
    assert m32.decide(m1b, m32.metric2(bad, answers_for(bad, ["B"] * 20, ["B"] * 20))) == "keep_off"


def test_summarize_and_table_report_both_arms_and_score_reads_the_run_folder(tmp_path, monkeypatch):
    fake, s, recs = small(tmp_path, shows=2, bits=3)
    verdicts = m32.label_all(FakeJudge(), recs, tmp_path)
    res = m32.summarize(recs, verdicts, {}, n_boot=500)
    assert res["arms"]["B"]["distinct_topics"] == 6 and res["arms"]["B"]["recycled_topics"] == 0
    assert res["metric1"]["share"]["A"]["share"] == 1.0 and res["metric1"]["share"]["B"]["share"] == 0.0
    assert res["verdict"] == "pending_review" and "pending_review" in m32.table(res) and "Wilson" in m32.table(res)
    monkeypatch.setattr(m32, "run_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(m32, "slots", lambda *a, **k: s)
    scored = m32.score()
    assert scored["metric1"]["share"] == res["metric1"]["share"]
    np.testing.assert_array_equal(m32.paired(recs, verdicts)[1], -np.ones(6))
