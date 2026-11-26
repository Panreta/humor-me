"""Exp 27 (D-111): the bookkeeping of the connected-threads experiment, with a fake Ollama and a fake judge, no model."""
import json
import pathlib
import subprocess
import sys
import threading

import httpx
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import generate, llm  # noqa: E402
from sim import cache  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim import thread_coherence as tc  # noqa: E402
from sim import thread_timing as tt  # noqa: E402


# --- fakes ---------------------------------------------------------------------------------------------------------

class FakeOllama:
    """Answers a writer request with a thread whose lines say which arm's prompt it got (zzA for gen-v4, zzB for
    gen-v5) and carry a unique number, so every thread has its own hash. Records every request body."""

    def __init__(self):
        self.seen, self.lock, self.n = [], threading.Lock(), 0

    def __call__(self, request):
        body = json.loads(request.content)
        with self.lock:
            self.seen.append(body)
            self.n += 1
            n = self.n
        arm = "zzB" if "one bit of stand-up" in body["messages"][0]["content"] else "zzA"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(
            {"premise": f"{arm} premise {n}", "jokes": [f"{arm} {n} line {j}." for j in range(5)]})}, "finish_reason": "stop"}]})

    def client(self):
        return llm.Client("http://local/v1", "", "bonsai-2:4b", 10.0, retries=2, deadline_s=12.0, local=True,
                          extra_body=llm.OLLAMA_NO_THINK, json_schema=True, transport=httpx.MockTransport(self))


class FakeJudge:
    """connected iff the shown lines carry `marker`; keeps what it was shown."""

    def __init__(self, marker="zzB", fail=False):
        self.marker, self.fail, self.calls = marker, fail, []

    def chat(self, system, user, **kw):
        self.calls.append((system, user, kw))
        if self.fail:
            raise llm.LLMError("http 503", 503)
        return json.dumps({"connected": self.marker in user})


def ps(share=1.0, context=8192):
    return lambda url: {"models": [{"name": "bonsai-2:4b", "model": "bonsai-2:4b", "size": 100,
                                    "size_vram": int(100 * share), "context_length": context}]}


def rec(prompt, arm, threads, lost=(), wall=1.0, bad=0):
    return {"prompt": prompt, "arm": arm, "wall_s": wall, "threads": threads, "lost": list(lost), "failure": None,
            "bad_replies": bad}


def lines(tag):
    return [f"{tag} line {j}." for j in range(5)]


def user_of(body):
    return json.loads(body["messages"][1]["content"])


# --- the prompts ---------------------------------------------------------------------------------------------------

def test_the_prompts_are_seeded_disjoint_from_exp_24_and_26_and_carry_0_to_4_told_premises():
    mine = tc.prompts()
    assert tc.prompts() == mine and len(mine) == 20 and tc.prompts(5, "other") != mine[:5]
    ids = {p["id"] for p in mine}
    assert ids == {f"c{i:02d}" for i in range(20)}
    assert not ids & {p["id"] for p in cg.prompts(65)} and not ids & {p["id"] for p in tt.prompts()}
    assert tc.SEED not in (cg.SEED, tt.SEED) and tc.SEED != str(cg.SEED)
    assert {len(p["told"]) for p in mine} == {0, 1, 2, 3, 4}
    for p in mine:
        assert set(p["told"]) <= set(tc.TOPICS) and len(set(p["told"])) == len(p["told"])
    assert {p["mode"] for p in mine} == set(generate.MODES)


def test_the_digest_follows_both_prompts_the_model_and_the_judge_and_a_stale_call_is_refused(monkeypatch):
    before, jbefore = tc.digest(), tc.judge_digest()
    with monkeypatch.context() as m:
        m.setitem(generate.PROMPTS, "gen-v5", generate.SYSTEM_CONNECTED + " x")
        assert tc.digest() != before
    with monkeypatch.context() as m:
        m.setenv("JOKE_WRITER_MODEL", "other:4b")
        assert tc.digest() != before
    monkeypatch.setattr(tc, "JUDGE_SYSTEM", tc.JUDGE_SYSTEM + " x")
    assert tc.judge_digest() != jbefore and tc.digest() == before
    with pytest.raises(AssertionError):
        tc.run(digest_=before, judge_digest_=jbefore)
    assert tc.params()["n_prompts"] == 20


# --- writing the bits ----------------------------------------------------------------------------------------------

def test_both_arms_write_through_the_real_thread_generator_with_their_own_prompt_and_the_told_premises(tmp_path):
    fake = FakeOllama()
    bits = tc.prompts(4)
    recs = tc.generate_bits(fake.client(), bits, tmp_path, http=ps())
    assert [(r["prompt"], r["arm"]) for r in recs] == [(p["id"], a) for p in bits for a in "AB"]
    assert all(len(r["threads"]) == 3 and r["lost"] == [] and r["failure"] is None for r in recs)
    calls = fake.seen[1:]  # the first request is the warm-up probe
    assert len(fake.seen) == 1 + 4 * 2 * 3
    by_told: dict[str, list[str]] = {}
    for body in calls:
        system = body["messages"][0]["content"]
        is_b = "one bit of stand-up" in system
        assert system.startswith("You write material") and ("callback to line 1" in system) == is_b
        assert body["response_format"]["json_schema"]["schema"] == generate.COUNTED_REPLY_SCHEMA
        assert set(user_of(body)) == {"venue", "energy", "premises_already_told"}
        by_told.setdefault(json.dumps(user_of(body)["premises_already_told"]), []).append("B" if is_b else "A")
    assert sorted(len(json.loads(k)) for k in by_told) == [0, 1, 2, 3]  # the first 4 prompts carry 0..3 premises
    assert all(sorted(v) == ["A"] * 3 + ["B"] * 3 for v in by_told.values())  # both arms see the same list
    assert all(t[0].startswith("zzA") for r in recs if r["arm"] == "A" for t in r["threads"])
    assert all(t[0].startswith("zzB") for r in recs if r["arm"] == "B" for t in r["threads"])


def test_a_finished_bit_is_read_back_not_rewritten_and_warm_up_is_skipped_when_nothing_is_left(tmp_path, monkeypatch):
    fake = FakeOllama()
    first = tc.generate_bits(fake.client(), tc.prompts(3), tmp_path, http=ps())
    n = len(fake.seen)
    monkeypatch.setattr(generate, "warm_up", lambda *a, **k: pytest.fail("warm-up with nothing to write"))
    again = tc.generate_bits(fake.client(), tc.prompts(3), tmp_path)
    assert len(fake.seen) == n and again == first
    assert tc.gather(tmp_path, tc.prompts(3)) == first and tc.gather(tmp_path, tc.prompts(4)) is None


def test_the_arm_order_alternates_by_prompt(tmp_path):
    order = []

    class Spy(FakeOllama):
        def __call__(self, request):
            body = json.loads(request.content)
            if "premises_already_told" in body["messages"][1]["content"]:
                order.append("B" if "one bit of stand-up" in body["messages"][0]["content"] else "A")
            return super().__call__(request)
    tc.generate_bits(Spy().client(), tc.prompts(2), tmp_path, http=ps())
    assert [order[0], order[3]] == ["A", "B"]


def test_a_bit_that_fails_is_a_result_with_its_reasons_and_no_text():
    c = llm.Client("http://local/v1", "", "m", 5.0, retries=2, deadline_s=5.0, local=True, json_schema=True,
                   transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{"message": {
                       "content": "not json"}, "finish_reason": "stop"}]})))
    r = tc.write_bit(c, tc.prompts(1)[0], "B")
    assert r["threads"] == [] and r["failure"] == "GenerationError" and len(r["lost"]) == 3 and r["bad_replies"] >= 3
    assert all(x.startswith("GenerationError: malformed reply") for x in r["lost"])


# --- the judge -----------------------------------------------------------------------------------------------------

def test_the_judge_is_blind_plain_cached_by_text_and_one_call_per_thread(tmp_path):
    j = FakeJudge()
    curly = ["It’s “fine” — really…"] + lines("zzB")[1:]
    v1, called1 = tc.judge_thread(j, curly, tmp_path)
    v2, called2 = tc.judge_thread(j, curly, tmp_path)
    assert (v1, called1, v2, called2) == (True, True, True, False) and len(j.calls) == 1
    system, user, kw = j.calls[0]
    assert kw["json_reply"] == tc.JUDGE_SCHEMA == {"type": "object", "additionalProperties": False,
                                                  "required": ["connected"], "properties": {"connected": {"type": "boolean"}}}
    assert user.startswith("1. It's \"fine\" - really...") and user.count("\n") == 4  # plain punctuation, 5 numbered lines
    assert not set(user) & set("’“”—…")
    for word in ("gen-v4", "gen-v5", "arm", "connected prompt"):
        assert word not in system + user
    assert "ONE stand-up bit" in system and "separate jokes" in system
    other, _ = tc.judge_thread(FakeJudge(), lines("zzA"), tmp_path)
    assert other is False


def test_a_judge_outage_is_not_cached_and_does_not_decide_a_verdict(tmp_path):
    assert tc.judge_thread(FakeJudge(fail=True), lines("zzB"), tmp_path) == (None, True)
    assert not list((tmp_path / "judge_cache").rglob("*.json"))
    assert tc.judge_thread(FakeJudge(), lines("zzB"), tmp_path) == (True, True)


def test_the_judge_client_is_the_safety_filters_haiku_with_thinking_off(monkeypatch):
    monkeypatch.setenv("SAFETY_CHECK_PROVIDER", "gemini")  # a stray env must not move the judge
    c = tc.make_judge_client()
    assert (c.model, c.thinking, c.retries, c.deadline_s) == ("claude-haiku-5-5", False, generate.FILTER_RETRIES,
                                                             generate.FILTER_TIMING[1])
    assert tc.parse_verdict('{"connected": true}') is True
    for bad in ("nope", '{"connected": "yes"}', "{}"):
        with pytest.raises(Exception):
            tc.parse_verdict(bad)


# --- the bar -------------------------------------------------------------------------------------------------------

def test_the_bar_needs_a_positive_mean_and_a_ci_lower_bound_above_zero():
    good = tc.bar(np.full(20, 0.3))
    assert good["passes"] is True and good["ci95"][0] > 0 and good["mean_diff"] == pytest.approx(0.3)
    assert good["n_paired"] == 20
    noisy = np.array([1.0, -1.0] * 9 + [1.0, 0.0])  # mean +0.1 but far from settled
    b = tc.bar(noisy)
    assert b["mean_diff"] > 0 and b["ci95"][0] < 0 and b["passes"] is False
    assert tc.bar(np.zeros(20))["passes"] is False and tc.bar(np.full(20, -0.2))["passes"] is False
    assert tc.bar(np.array([0.5]))["passes"] is False and tc.bar(np.array([]))["mean_diff"] is None


def test_prompts_are_paired_by_id_whatever_the_order_and_a_prompt_missing_an_arm_drops_out():
    verdicts = {tc.thread_hash(lines(t)): v for t, v in (("a1", True), ("a2", False), ("b1", True), ("b2", True), ("b3", False))}
    recs = [rec("c00", "A", [lines("a1"), lines("a2")]), rec("c00", "B", [lines("b1"), lines("b2")]),
            rec("c01", "A", [lines("a1")]), rec("c01", "B", [lines("b3")]),
            rec("c02", "A", [lines("a1")]), rec("c02", "B", [])]  # c02: B delivered nothing
    ids, d = tc.paired_diffs(tc.per_prompt(recs, verdicts))
    assert ids == ["c00", "c01"] and list(d) == [pytest.approx(1.0 - 0.5), pytest.approx(0.0 - 1.0)]
    shuffled = [recs[i] for i in (5, 2, 0, 4, 1, 3)]
    ids2, d2 = tc.paired_diffs(tc.per_prompt(shuffled, verdicts))
    assert ids2 == ids and list(d2) == list(d)
    rates = tc.per_prompt(recs, verdicts)
    assert rates["c02"]["B"]["rate"] is None and rates["c02"]["B"]["delivered"] == 0


def test_an_unjudged_thread_is_left_out_of_the_rate_not_counted_as_not_connected():
    verdicts = {tc.thread_hash(lines("a1")): True, tc.thread_hash(lines("a2")): None}
    r = tc.per_prompt([rec("c00", "A", [lines("a1"), lines("a2")])], verdicts)["c00"]["A"]
    assert (r["delivered"], r["judged"], r["connected"], r["rate"]) == (2, 1, 1, 1.0)


# --- the spot-check ------------------------------------------------------------------------------------------------

def mixed(per_cell=4):
    """Both arms have threads the judge called connected and threads it called not, `per_cell` of each."""
    recs, verdicts = [], {}
    for p in range(2):
        for arm in "AB":
            ts = [lines(f"{arm}{p}c{i}") for i in range(per_cell)] + [lines(f"{arm}{p}n{i}") for i in range(per_cell)]
            recs.append(rec(f"c{p:02d}", arm, ts))
            verdicts.update({tc.thread_hash(t): ("c" in t[0].split()[0][2:]) for t in ts})
    return recs, verdicts


def few(connected, not_connected):
    """One bit of arm B with `connected` judged-connected and `not_connected` judged-not threads."""
    cs = [lines(f"c{i}") for i in range(connected)]
    ns = [lines(f"n{i}") for i in range(not_connected)]
    verdicts = {tc.thread_hash(t): True for t in cs} | {tc.thread_hash(t): False for t in ns}
    return [rec("c00", "B", cs + ns)], verdicts


def answers_for(recs, verdicts, disagree):
    """Human answers that differ from the judge on exactly `disagree` of the picks (the first ones), in both directions."""
    picks = tc.draw(recs, verdicts)["picks"]
    out = {}
    for i, h in enumerate(picks):
        judged = verdicts[h]
        out[h] = ("n" if judged else "y") if i < disagree else ("y" if judged else "n")
    return out


def test_the_check_asks_10_threads_half_judged_connected_half_not_spread_over_both_arms():
    recs, verdicts = mixed()
    d = tc.draw(recs, verdicts)
    assert d["asked"] == 10 and len(set(d["picks"])) == 10 and tc.draw(recs, verdicts) == d  # a fixed draw
    assert (d["connected"], d["not_connected"]) == (5, 5)
    assert d["arm_A"] + d["arm_B"] == 10 and abs(d["arm_A"] - d["arm_B"]) <= 2  # 3 and 2 within each half
    arms = {h: arm for r in recs for t in r["threads"] for h, arm in [(tc.thread_hash(t), r["arm"])]}
    for want in (True, False):
        half = [h for h in d["picks"] if verdicts[h] is want]
        assert sorted(sum(arms[h] == a for h in half) for a in "AB") == [2, 3]
    assert d["picks"] != sorted(d["picks"], key=lambda h: not verdicts[h])  # shuffled: order does not give the half away


def test_a_short_half_is_filled_from_the_other_and_the_draw_is_recorded():
    recs, verdicts = few(2, 20)
    d = tc.draw(recs, verdicts)
    assert (d["asked"], d["connected"], d["not_connected"], d["arm_B"], d["arm_A"]) == (10, 2, 8, 10, 0)
    recs, verdicts = few(20, 1)
    d = tc.draw(recs, verdicts)
    assert (d["asked"], d["connected"], d["not_connected"]) == (10, 9, 1)
    recs, verdicts = few(2, 3)
    d = tc.draw(recs, verdicts)
    assert (d["asked"], d["connected"], d["not_connected"]) == (5, 2, 3)  # all there is
    c = tc.spot_check(recs, verdicts, {})
    assert (c["asked"], c["drawn_connected"], c["drawn_not_connected"], c["short"]) == (5, 2, 3, True)
    unanswered = {h: None for h in ()}
    recs, verdicts = few(3, 3)
    verdicts[tc.thread_hash(recs[0]["threads"][0])] = None  # a thread the judge could not answer is never drawn
    assert tc.draw(recs, verdicts)["asked"] == 5 and unanswered == {}


def test_the_review_shows_lines_only_resumes_and_records_what_was_drawn(tmp_path):
    recs, verdicts = mixed()
    picks = tc.draw(recs, verdicts)["picks"]
    shown, answers = [], iter(["y", "n", "q"])
    done = tc.review(tmp_path, recs, verdicts, read=lambda prompt: next(answers), out=shown.append)
    assert done == 2
    text = "\n".join(shown)
    assert text.count("Thread ") == 3 and "Thread 1 of 10" in text
    assert all(w not in text for w in ("gen-v4", "gen-v5", "arm", "judge", "connected", "not connected"))
    saved = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert len(saved["answers"]) == 2 and set(saved["answers"]) <= set(picks)
    assert saved["drawn"] == {"asked": 10, "connected": 5, "not_connected": 5, "arm_A": 5, "arm_B": 5} or (
        saved["drawn"]["asked"] == 10 and saved["drawn"]["connected"] == 5)
    shown.clear()
    rest = iter(["n"] * 8)
    assert tc.review(tmp_path, recs, verdicts, read=lambda prompt: next(rest), out=shown.append) == 10
    text = "\n".join(shown)
    assert "Thread 1 of 10" not in text and "Thread 3 of 10" in text  # started at the first unanswered


def test_four_disagreements_of_ten_void_the_judge_and_three_do_not_in_either_direction():
    recs, verdicts = mixed()
    for n, void in ((0, False), (3, False), (4, True), (10, True)):
        c = tc.spot_check(recs, verdicts, answers_for(recs, verdicts, n))
        assert (c["disagree"], c["void"], c["complete"], c["asked"], c["answered"], c["short"]) == (n, void, True, 10, 10, False)
        assert c["disagreement"] == pytest.approx(n / 10) and c["yes"] + c["no"] == 10
    picks = tc.draw(recs, verdicts)["picks"]
    judged_yes = next(h for h in picks if verdicts[h])
    judged_no = next(h for h in picks if not verdicts[h])
    one = tc.spot_check(recs, verdicts, {judged_yes: "n"})
    two = tc.spot_check(recs, verdicts, {judged_no: "y"})
    assert one["disagree"] == two["disagree"] == 1  # the judge's false positives and false negatives both count
    agree = tc.spot_check(recs, verdicts, {judged_yes: "y", judged_no: "n"})
    assert agree["disagree"] == 0 and agree["complete"] is False
    early = tc.spot_check(recs, verdicts, {h: answers_for(recs, verdicts, 4)[h] for h in picks[:4]})
    assert early["void"] is True and early["complete"] is False  # four disagreements already void it
    assert tc.spot_check(recs, verdicts, {})["disagreement"] is None


def test_with_fewer_than_ten_threads_to_check_the_judge_can_never_be_cleared_and_three_of_four_voids_it():
    recs, verdicts = few(2, 2)  # 4 threads: the old count rule (4 no) could never fire here
    picks = tc.draw(recs, verdicts)["picks"]
    assert len(picks) == 4
    for n, void in ((0, False), (1, False), (2, True), (3, True)):  # 40% of 4 is 1.6: two disagreements void it
        c = tc.spot_check(recs, verdicts, answers_for(recs, verdicts, n))
        assert (c["void"], c["short"], c["complete"]) == (void, True, True)
        b = tc.bar(np.full(20, 0.3))  # a passing bar adopts nothing on a check this small
        assert tc.decide(b, c) == "inconclusive_judge"
    three_of_four = tc.spot_check(recs, verdicts, answers_for(recs, verdicts, 3))
    assert three_of_four["disagree"] == 3 and three_of_four["void"] is True
    assert tc.decide(tc.bar(np.full(20, 0.3)), tc.spot_check(*few(0, 0), {})) == "inconclusive_judge"  # nothing to check at all


def test_the_decision_needs_a_complete_check_that_holds_whatever_the_bar_says():
    good, bad = tc.bar(np.full(20, 0.3)), tc.bar(np.zeros(20))
    ok = {"void": False, "short": False, "complete": True}
    assert tc.decide(good, ok) == "adopt" and tc.decide(bad, ok) == "keep_current"
    for b in (good, bad):
        assert tc.decide(b, {**ok, "void": True}) == "inconclusive_judge"  # whatever the CI says
        assert tc.decide(b, {**ok, "short": True}) == "inconclusive_judge"
        assert tc.decide(b, {**ok, "complete": False}) == "pending_review"  # a failed bar needs the check too


def test_nothing_is_decided_by_the_bar_alone():
    recs = [rec(f"c{p:02d}", arm, [lines(f"{arm}{p}{i}") for i in range(3)]) for p in range(7) for arm in "AB"]
    verdicts = {tc.thread_hash(t): r["arm"] == "B" for r in recs for t in r["threads"]}  # B always connected, A never
    res = tc.summarize(recs, verdicts, {}, n_boot=300)
    assert res["bar"]["passes"] is True and res["verdict"] == "pending_review"
    assert tc.summarize(recs, verdicts, answers_for(recs, verdicts, 0), n_boot=300)["verdict"] == "adopt"
    assert tc.summarize(recs, verdicts, answers_for(recs, verdicts, 4), n_boot=300)["verdict"] == "inconclusive_judge"
    same, same_v = mixed()  # both arms alike: the bar fails, and a clean check still has to be done first
    res = tc.summarize(same, same_v, {}, n_boot=300)
    assert res["bar"]["passes"] is False and res["verdict"] == "pending_review"
    assert tc.summarize(same, same_v, answers_for(same, same_v, 0), n_boot=300)["verdict"] == "keep_current"
    assert tc.summarize(same, same_v, answers_for(same, same_v, 4), n_boot=300)["verdict"] == "inconclusive_judge"


def test_the_chance_the_check_voids_a_judge_is_the_binomial_tail():
    got = [tc.p_void(e) for e in (0.2, 0.3, 0.4, 0.5)]
    assert got == pytest.approx([0.121, 0.350, 0.618, 0.828], abs=0.001)
    assert tc.p_void(0.0) == 0.0 and tc.p_void(1.0) == 1.0 and got == sorted(got)
    rng = np.random.default_rng(1)  # and a simulation of the same rule agrees
    sim = (rng.binomial(10, 0.3, 20000) >= 4).mean()
    assert abs(sim - tc.p_void(0.3)) < 0.02


# --- the whole run, and what reaches the metrics -------------------------------------------------------------------

def test_a_whole_run_writes_numbers_and_reasons_but_no_thread_text_to_the_metrics(tmp_path, monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(tc, "DATA", tmp_path / "data")
    monkeypatch.setattr(cache, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(generate, "writer_client", fake.client)
    monkeypatch.setattr(generate, "warm_up", lambda *a, **k: 0.1)
    monkeypatch.setattr(generate, "placement", lambda *a, **k: (1.0, 8192, []))
    judge = FakeJudge()
    monkeypatch.setattr(tc, "make_judge_client", lambda: judge)
    res = cache.run_or_load(tc.NAME, tc.run, **tc.params(6))
    assert res["bar"]["passes"] is True and res["verdict"] == "pending_review" and res["n_prompts"] == 6
    assert res["writer_on_gpu_after_run"] is True and res["spot_check"]["asked"] == 10
    assert res["arms"]["A"]["threads_delivered"] == res["arms"]["B"]["threads_delivered"] == 18
    assert res["arms"]["A"]["threads_connected"] == 0 and res["arms"]["B"]["threads_connected"] == 18
    assert len(judge.calls) == 36  # one call per thread
    text = (tmp_path / "metrics" / "thread_coherence.json").read_text(encoding="utf-8")
    assert "zz" not in text and "line " not in text and "premise" not in text
    assert not any(t in text for t in tc.TOPICS)
    assert json.loads(text)["params"]["n_prompts"] == 6
    assert any("zzB" in p.read_text(encoding="utf-8") for p in (tmp_path / "data").rglob("*.json"))  # the text is in data/
    n = len(judge.calls)
    again = tc.score(6)  # no model call: the run folder and the verdict cache are enough
    assert len(judge.calls) == n and again["bar"] == res["bar"]
    assert "zz" not in json.dumps(again)


def test_a_run_refuses_a_hosted_writer_and_the_table_prints_numbers(monkeypatch, tmp_path):
    monkeypatch.setattr(tc, "DATA", tmp_path / "data")
    monkeypatch.setattr(generate, "writer_client", lambda: llm.Client("http://x", "k", "m"))
    with pytest.raises(SystemExit):
        tc.run(digest_=tc.digest(), judge_digest_=tc.judge_digest())
    recs, verdicts = mixed()
    out = tc.table(tc.summarize(recs, verdicts, {}, n_boot=200))
    assert "pending_review" in out and "zz" not in out and "line " not in out


def test_the_score_command_refuses_to_guess_when_bits_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "DATA", tmp_path / "data")
    with pytest.raises(SystemExit):
        tc.score()


def test_thread_text_is_gitignored():
    r = subprocess.run(["git", "check-ignore", "-q", "data/thread_coherence/run-x/bits/c00-A.json"], cwd=ROOT)
    assert r.returncode == 0


# --- power ---------------------------------------------------------------------------------------------------------

def test_the_power_simulation_has_few_false_alarms_and_rises_with_the_size_of_the_difference():
    null = tc.simulate_power(0.0, reps=150, n_boot=300)
    small, mid, big = (tc.simulate_power(d, reps=150, n_boot=300) for d in (0.1, 0.2, 0.4))
    assert null < 0.12 and small < mid < big and big > 0.95


def test_the_writers_gpu_placement_is_read_after_the_run_and_only_when_something_was_written(tmp_path, monkeypatch):
    fake = FakeOllama()
    tc.generate_bits(fake.client(), tc.prompts(2), tmp_path, http=ps())
    assert tc.writer_on_gpu(tmp_path) is True
    monkeypatch.setattr(generate, "warm_up", lambda *a, **k: 0.1)  # the show's own warm-up would refuse these
    tc.generate_bits(fake.client(), tc.prompts(2), tmp_path / "off", http=ps(share=0.8))
    assert tc.writer_on_gpu(tmp_path / "off") is False
    tc.generate_bits(fake.client(), tc.prompts(2), tmp_path / "ctx", http=ps(context=4096))
    assert tc.writer_on_gpu(tmp_path / "ctx") is False
    assert tc.writer_on_gpu(tmp_path / "never") is None


def test_wall_time_is_split_by_whether_a_bit_re_asked():
    recs = [rec("c00", "A", [lines("a")], wall=5.0), rec("c01", "A", [lines("b")], wall=11.0, bad=1),
            rec("c02", "A", [lines("c")], wall=6.0), rec("c00", "B", [lines("d")], wall=12.0, bad=2)]
    a = tc.arm_stats(recs, "A", {})
    assert a["bits_reasked"] == 1 and a["bit_p50_s_reasked"] == 11.0 and a["bit_p50_s_no_reask"] == 5.0
    assert a["bit_p95_s_no_reask"] == 6.0 and a["bad_replies"] == 1
    assert tc.arm_stats(recs, "B", {})["bit_p95_s_reasked"] == 12.0 and tc.arm_stats([], "A", {})["bit_p50_s_reasked"] is None
