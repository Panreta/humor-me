"""Exp 30 (D-114): judge v1 against judge v2 on a fresh sample, with a fake Ollama and fake judges, no model."""
import hashlib
import json
import pathlib
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import generate, llm  # noqa: E402
from sim import cache  # noqa: E402
from sim import judge_calibration as jc  # noqa: E402
from sim import thread_coherence as tc  # noqa: E402
from tests.test_thread_coherence import FakeOllama, ps, rec  # noqa: E402

V1_SHA = "332acb1c283b1b076d2a8fc8f3ece7c4bd91512e5be8bb27b77ff3a5abc57b91"


class ByJudge:
    """v1 calls 'connected' iff the thread is arm A's, v2 iff it is arm B's; records the system prompt of every call."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def chat(self, system, user, **kw):
        self.calls.append((system, user))
        if self.fail:
            raise llm.LLMError("http 503", 503)
        marker = "zzA" if system == tc.JUDGE_SYSTEM else "zzB"
        return json.dumps({"connected": marker in user})


def records(n_bits=10, per_bit=3, tag=""):
    """n_bits prompts, both arms, `per_bit` threads each; every thread has its own text."""
    return [rec(f"f{i:02d}", arm, [[f"zz{arm}{tag} {i} {k} line {j}." for j in range(5)] for k in range(per_bit)])
            for i in range(n_bits) for arm in "AB"]


def seed_reviewed(n=10):
    """Exp 27's review.json in the (patched) data folder, so the fresh draw has its 10 reviewed hashes."""
    tc._write(tc.review_path(tc.run_dir()), {"answers": {f"old{i:02d}": "y" for i in range(n)}})


def scenario(recs, v1_right, v2_right):
    """Verdicts and answers on the drawn picks: the user says one bit on even picks and separate jokes on odd ones;
    v1 agrees with the user on the first `v1_right` picks and v2 on the first `v2_right` (nested agreeing sets)."""
    picks = jc.draw(recs, set())["picks"]
    answers = {h: "y" if i % 2 == 0 else "n" for i, h in enumerate(picks)}
    verdicts = {v: {h: (answers[h] == "y") == (i < k) for i, h in enumerate(picks)} for v, k in (("v1", v1_right), ("v2", v2_right))}
    return verdicts, answers


# --- the judges ----------------------------------------------------------------------------------------------------

def test_judge_v1_and_exp_27_digests_are_pinned():
    assert hashlib.sha256(tc.JUDGE_SYSTEM.encode()).hexdigest() == V1_SHA
    assert tc.JUDGES["v1"] is tc.JUDGE_SYSTEM and tc.JUDGES["v2"] is tc.JUDGE_SYSTEM_V2
    assert (tc.judge_digest(), tc.judge_digest("v1"), tc.digest()) == ("00f9662b", "00f9662b", "1097f143")


def test_v2_asks_for_structure_with_the_same_reply_and_never_for_funniness():
    v2 = tc.JUDGE_SYSTEM_V2
    assert v2 != tc.JUDGE_SYSTEM and '{"connected": true}' in v2 and '{"connected": false}' in v2
    assert "callback" in v2 and "shared theme alone" in v2 and "never whether it is funny" in v2


def test_v2_verdicts_cannot_collide_with_v1s(tmp_path):
    assert tc.judge_digest("v1") != tc.judge_digest("v2")
    lines = [f"zzA line {j}." for j in range(5)]
    h = tc.thread_hash(lines)
    assert tc.verdict_path(tmp_path, h, "v1") != tc.verdict_path(tmp_path, h, "v2")
    j = ByJudge()
    assert tc.judge_thread(j, lines, tmp_path, "v1") == (True, True)
    assert tc.judge_thread(j, lines, tmp_path, "v2") == (False, True)  # v1's cached verdict was not read back as v2's
    assert tc.judge_thread(j, lines, tmp_path, "v2") == (False, False) and tc.judge_thread(j, lines, tmp_path) == (True, False)
    assert [s for s, _ in j.calls] == [tc.JUDGE_SYSTEM, tc.JUDGE_SYSTEM_V2]


def test_judge_all_takes_a_version(tmp_path):
    j = ByJudge()
    recs = records(2, 1)
    assert set(tc.judge_all(j, recs, tmp_path, "v2").values()) == {True, False}
    assert {s for s, _ in j.calls} == {tc.JUDGE_SYSTEM_V2}


# --- the fresh prompts ---------------------------------------------------------------------------------------------

def test_the_fresh_prompts_are_seeded_and_disjoint_from_exp_27s_in_ids_topics_and_seed():
    mine, theirs = jc.prompts(), tc.prompts()
    assert [p["id"] for p in mine] == [f"f{i:02d}" for i in range(10)] and jc.prompts() == mine
    assert not set(jc.TOPICS) & set(tc.TOPICS) and len(set(jc.TOPICS)) == len(jc.TOPICS)
    assert all(set(p["told"]) <= set(jc.TOPICS) for p in mine) and [len(p["told"]) for p in mine] == [i % 5 for i in range(10)]
    assert jc.SEED != tc.SEED and not {p["id"] for p in mine} & {p["id"] for p in theirs}
    assert {p["mode"] for p in mine} == set(generate.MODES)
    assert jc.digest() != tc.digest() and jc.run_dir().name == f"calib-{jc.digest()}"


# --- the draw ------------------------------------------------------------------------------------------------------

def test_the_draw_is_ten_per_arm_one_thread_per_bit_first_and_seeded():
    recs = records()
    d = jc.draw(recs, set())
    assert d["asked"] == 20 and d["arm_A"] == d["arm_B"] == 10 and not d["short"]
    assert d["max_from_one_bit"] == 1 and len(set(d["picks"])) == 20
    assert jc.draw(recs, set())["picks"] == d["picks"] and jc.draw(list(reversed(recs)), set())["picks"] == d["picks"]
    assert [d["arm_of"][h] for h in d["picks"][:10]] != ["A"] * 10  # shuffled across arms
    text = tc.threads_by_hash(recs)
    assert all(text[h][0].startswith("zz" + d["arm_of"][h]) for h in d["picks"])


def test_the_draw_takes_at_most_two_from_a_bit_and_is_short_when_it_cannot_reach_ten():
    five = jc.draw(records(5), set())  # 5 bits x 3 threads per arm: 10 only by taking 2 from each bit
    assert five["arm_A"] == five["arm_B"] == 10 and five["max_from_one_bit"] == 2
    three = jc.draw(records(3), set())  # 3 bits per arm, capped at 2 each: 6 per arm
    assert three["arm_A"] == three["arm_B"] == 6 and three["asked"] == 12 and three["short"] and three["max_from_one_bit"] == 2


def test_the_draw_skips_threads_already_reviewed_and_counts_them():
    recs = records()
    base = jc.draw(recs, set())
    gone = set(base["picks"][:4])
    d = jc.draw(recs, gone)
    assert not gone & set(d["picks"]) and d["asked"] == 20 and d["dropped_as_reviewed"] == 4
    only_one = jc.draw(records(4, 3), {tc.thread_hash(t) for r in records(4, 3) for t in r["threads"]})
    assert only_one["asked"] == 0 and only_one["short"]


def test_the_draw_does_not_look_at_a_verdict():
    import inspect
    assert list(inspect.signature(jc.draw).parameters) == ["records", "exclude"]


def test_a_thread_in_both_arms_is_drawn_once():
    same = [["same line %d." % j for j in range(5)]]
    recs = [rec("f00", "A", same), rec("f00", "B", same), rec("f01", "A", [["a1 %d." % j for j in range(5)]])]
    d = jc.draw(recs, set())
    assert d["asked"] == 2 and len(set(d["picks"])) == 2


# --- agreement, the paired difference and the rule ------------------------------------------------------------------

def test_agreement_counts_and_the_paired_difference_are_the_arithmetic_of_the_threads():
    recs = records()
    verdicts, answers = scenario(recs, v1_right=8, v2_right=14)
    res = jc.summarize(recs, verdicts, answers, set(), n_boot=500)
    assert res["judges"]["v1"]["agree"] == 8 and res["judges"]["v2"]["agree"] == 14
    assert res["judges"]["v2"]["disagreement"] == 0.3 and res["judges"]["v1"]["disagreement"] == 0.6
    assert res["paired"]["v2_only_right"] == 6 and res["paired"]["v1_only_right"] == 0
    assert res["paired"]["mean_diff"] == pytest.approx(0.3)
    lo, hi = res["paired"]["ci95"]
    assert 0 < lo <= 0.3 <= hi <= 0.6
    assert res["judges"]["v2"]["confusion"] == {"judge_yes_user_yes": 7, "judge_yes_user_no": 3,
                                                "judge_no_user_yes": 3, "judge_no_user_no": 7}
    assert res["judges"]["v2"]["agreement_ci95"][0] < 0.7 < res["judges"]["v2"]["agreement_ci95"][1]


def test_the_paired_difference_counts_each_direction_when_the_judges_are_right_on_different_threads():
    recs = records()
    picks = jc.draw(recs, set())["picks"]
    answers = {h: "y" for h in picks}
    v1 = {h: i < 10 for i, h in enumerate(picks)}  # right on picks 0-9
    v2 = {h: 5 <= i < 17 for i, h in enumerate(picks)}  # right on picks 5-16
    res = jc.summarize(recs, {"v1": v1, "v2": v2}, answers, set(), n_boot=200)
    assert res["paired"]["v2_only_right"] == 7 and res["paired"]["v1_only_right"] == 5
    assert res["paired"]["mean_diff"] == pytest.approx(0.1) and res["judges"]["v1"]["agree"] == 10 and res["judges"]["v2"]["agree"] == 12


def test_v2_wins_a_clear_gap_and_is_trusted_and_may_rescore():
    recs = records()
    verdicts, answers = scenario(recs, 8, 20)
    res = jc.summarize(recs, verdicts, answers, set(), n_boot=500)
    assert (res["kept"], res["kept_trusted"], res["rescore_allowed"], res["verdict"]) == ("v2", True, True, "keep_v2_trusted")
    assert res["paired"]["ci95"][0] > 0 and res["paired"]["mean_diff"] == pytest.approx(0.6)


def test_a_tie_keeps_v1_even_when_both_are_trusted():
    recs = records()
    verdicts, answers = scenario(recs, 15, 15)
    res = jc.summarize(recs, verdicts, answers, set(), n_boot=200)
    assert (res["kept"], res["verdict"], res["rescore_allowed"]) == ("v1", "keep_v1_trusted", False)


def test_v2_one_thread_ahead_is_kept_on_the_point_estimate_though_the_ci_includes_zero():
    recs = records()
    picks = jc.draw(recs, set())["picks"]
    answers = {h: "y" for h in picks}
    v1 = {h: i < 10 for i, h in enumerate(picks)}  # right on 10
    v2 = {h: 5 <= i < 16 for i, h in enumerate(picks)}  # right on 11, different threads
    res = jc.summarize(recs, {"v1": v1, "v2": v2}, answers, set(), n_boot=500)
    assert res["kept"] == "v2" and res["judges"]["v2"]["agree"] - res["judges"]["v1"]["agree"] == 1 and res["paired"]["ci95"][0] < 0 < res["paired"]["ci95"][1]


def test_the_trust_rule_is_under_40_percent_so_8_of_20_voids_and_7_does_not():
    recs = records()
    seven, ans = scenario(recs, 3, 13)  # v2 disagrees on 7
    eight, ans8 = scenario(recs, 3, 12)  # on 8
    r7 = jc.summarize(recs, seven, ans, set(), n_boot=100)
    r8 = jc.summarize(recs, eight, ans8, set(), n_boot=100)
    assert (r7["verdict"], r7["rescore_allowed"]) == ("keep_v2_trusted", True)
    assert (r8["verdict"], r8["rescore_allowed"]) == ("keep_v2_untrusted", False)


def test_a_kept_v1_never_re_scores_even_when_trusted():
    recs = records()
    verdicts, answers = scenario(recs, 18, 10)
    res = jc.summarize(recs, verdicts, answers, set(), n_boot=100)
    assert (res["verdict"], res["rescore_allowed"]) == ("keep_v1_trusted", False)


def test_the_rule_functions_match_the_stated_rule_exhaustively():
    # these loops are the mutation check: `>` -> `>=` in keep, or `<` -> `<=` / 0.4 -> 0.5 in trusted, fails them
    for a1 in range(21):
        for a2 in range(21):
            assert jc.keep(a1, a2) == ("v2" if a2 > a1 else "v1")
    for n in (10, 20):
        for dis in range(n + 1):
            assert jc.trusted(dis, n) == (Fraction(dis, n) < Fraction(2, 5))
    assert not jc.trusted(0, 0)
    assert jc.verdict(True, True, "v2", True) == "inconclusive_sample" and jc.verdict(False, False, None, False) == "pending"
    assert jc.verdict(False, True, "v1", False) == "keep_v1_untrusted"


def test_missing_labels_a_judge_failure_or_a_short_draw_decide_nothing():
    recs = records()
    verdicts, answers = scenario(recs, 5, 20)
    some = dict(list(answers.items())[:19])
    res = jc.summarize(recs, verdicts, some, set(), n_boot=100)
    assert res["verdict"] == "pending" and res["kept"] is None and res["rescore_allowed"] is False and res["labelled"] == 19
    verdicts["v2"][next(iter(answers))] = None
    res = jc.summarize(recs, verdicts, answers, set(), n_boot=100)
    assert res["verdict"] == "pending" and res["judge_failures"] == {"v1": 0, "v2": 1}
    small = records(3)
    v, a = scenario(small, 1, 11)
    res = jc.summarize(small, v, a, set(), n_boot=100)
    assert res["verdict"] == "inconclusive_sample" and res["rescore_allowed"] is False and res["drawn"]["short"]


def test_the_user_share_by_arm_is_judge_free_and_has_its_own_interval():
    recs = records()
    d = jc.draw(recs, set())
    answers = {h: ("y" if (i % 10) < (3 if d["arm_of"][h] == "A" else 7) else "n")
               for i, h in enumerate(sorted(d["picks"], key=lambda h: (d["arm_of"][h], h)))}
    verdicts, _ = scenario(recs, 10, 10)
    res = jc.summarize(recs, verdicts, answers, set(), n_boot=500)
    assert res["user_one_bit_by_arm"]["A"] == {"n": 10, "one_bit": 3, "share": 0.3}
    assert res["user_one_bit_by_arm"]["B"] == {"n": 10, "one_bit": 7, "share": 0.7}
    diff = res["user_one_bit_diff_B_minus_A"]
    assert diff["diff"] == pytest.approx(0.4) and diff["ci95"][0] < 0.4 < diff["ci95"][1]
    assert sum(s["n"] for s in res["judge_one_bit_by_arm"]["v1"].values()) == 20


def test_the_prompt_paired_bootstrap_is_the_difference_of_means_and_a_missing_arm_has_none():
    items = [(f"f{i:02d}", arm, 1.0 if arm == "B" else 0.0) for i in range(10) for arm in "AB"]
    assert jc.boot_by_prompt(items, 200) == [1.0, 1.0]
    assert jc.boot_by_prompt([i for i in items if i[1] == "B"]) is None and jc.boot_by_prompt(items[:2]) is None


def test_the_paired_ci_resamples_whole_bits_so_two_threads_of_a_bit_move_together():
    x = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    one_each = jc.cluster_boot_mean(x, [str(i) for i in range(8)], 2000)
    by_bit = jc.cluster_boot_mean(x, ["a", "a", "b", "b", "c", "c", "d", "d"], 2000)
    assert by_bit[1] - by_bit[0] > one_each[1] - one_each[0]  # same data, wider when the pairs are tied
    assert jc.cluster_boot_mean(x, ["a"] * 8) is None


def test_the_judge_yes_rate_kappa_balanced_accuracy_and_wilson_interval():
    user = [True] * 13 + [False] * 7
    always_yes = [True] * 20
    assert jc.kappa(always_yes, user) == 0.0 or jc.kappa(always_yes, user) is None  # no information beyond the yes-rate
    assert jc.balanced_accuracy(always_yes, user) == pytest.approx(0.5)
    assert jc.kappa(user, user) == pytest.approx(1.0) and jc.kappa([True] * 4, [True] * 4) is None
    lo, hi = jc.wilson(4, 10)
    assert (round(lo, 2), round(hi, 2)) == (0.17, 0.69) and jc.wilson(0, 0) is None
    assert jc.balanced_accuracy([True, True], [True, True]) is None  # no user "no" class


def test_the_trust_gate_chance_by_true_agreement_is_the_binomial_tail():
    got = {a: round(jc.p_trusted(a), 2) for a in (0.5, 0.6, 0.65, 0.7)}
    assert got == {0.5: 0.13, 0.6: 0.42, 0.65: 0.6, 0.7: 0.77}
    assert jc.p_trusted(1.0) == 1.0 and jc.p_trusted(0.0) == 0.0


def test_an_always_yes_judge_can_win_the_rate_but_is_not_trusted_without_being_right_on_the_users_nos():
    recs = records()
    picks = jc.draw(recs, set())["picks"]
    answers = {h: "y" if i < 13 else "n" for i, h in enumerate(picks)}
    v1 = {h: False for h in picks}  # right on the 7 nos
    v2 = {h: True for h in picks}  # right on the 13 yeses: 7 disagreements, under the 8 that void it
    res = jc.summarize(recs, {"v1": v1, "v2": v2}, answers, set(), n_boot=100)
    assert res["kept"] == "v2" and jc.trusted(res["judges"]["v2"]["disagree"], 20)
    assert res["kept_right_in_each_class"] is False and res["verdict"] == "keep_v2_untrusted" and res["rescore_allowed"] is False
    assert res["judges"]["v2"]["yes_rate"] == 1.0 and res["user_yes_rate"] == 0.65 and res["judges"]["v2"]["balanced_accuracy"] == 0.5
    assert jc.both_classes({"judge_yes_user_yes": 1, "judge_no_user_no": 1}) and not jc.both_classes({"judge_yes_user_yes": 5, "judge_no_user_no": 0})


def test_each_judges_confusion_and_false_connected_rate_are_split_by_arm_and_the_guards_use_both_diffs():
    recs = records()
    d = jc.draw(recs, set())
    picks = d["picks"]
    # the user: arm A threads mostly separate, arm B mostly one bit; the judge says one bit on everything in B and A
    answers = {h: ("y" if d["arm_of"][h] == "B" else "n") for h in picks}
    always = {h: True for h in picks}
    ok_v = {h: answers[h] == "y" for h in picks}
    res = jc.summarize(recs, {"v1": always, "v2": ok_v}, answers, set(), n_boot=300)
    j1 = res["judges"]["v1"]
    assert j1["confusion_by_arm"]["A"]["judge_yes_user_no"] == 10 and j1["false_connected_by_arm"] == {"A": 1.0, "B": None}
    assert res["judges"]["v2"]["false_connected_by_arm"] == {"A": 0.0, "B": None}
    assert res["judge_one_bit_diff_B_minus_A"] == {"v1": 0.0, "v2": 1.0}
    assert res["user_one_bit_diff_B_minus_A"]["diff"] == 1.0 and res["kept"] == "v2"
    assert res["adoption_guards"] == {"user_diff_not_negative": True, "judge_diff_within_user_ci": True}
    # the judge says B is better than the user does: outside the user's interval, the guard fails
    mixed = {h: (("y" if i % 4 else "n") if d["arm_of"][h] == "B" else ("y" if i % 4 == 0 else "n")) for i, h in enumerate(picks)}
    res2 = jc.summarize(recs, {"v1": always, "v2": {h: d["arm_of"][h] == "B" for h in picks}}, mixed, set(), n_boot=300)
    assert res2["judge_one_bit_diff_B_minus_A"]["v2"] == 1.0
    assert res2["adoption_guards"]["judge_diff_within_user_ci"] is False
    neg = {h: ("y" if d["arm_of"][h] == "A" else "n") for h in picks}
    assert jc.summarize(recs, {"v1": always, "v2": always}, neg, set(), n_boot=100)["adoption_guards"]["user_diff_not_negative"] is False


# --- review --------------------------------------------------------------------------------------------------------

def test_the_review_shows_lines_only_in_a_neutral_question_saves_each_answer_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "DATA", tmp_path / "data")
    seed_reviewed()
    recs = records()
    shown, asked = [], []
    said = iter(["x", "y", "n", "q"])

    def ask(prompt):
        asked.append(prompt)
        return next(said)
    n = jc.review(tmp_path, recs, read=ask, out=shown.append)
    assert n == 2 and len(asked) == 4  # an invalid answer is asked again
    text = "\n".join(shown)
    assert text.count("Thread ") == 3 and "line 0." in text
    assert not any(w in text + " ".join(asked) for w in ("judge", "verdict", "arm", "premise", "builds", "v1", "v2"))
    saved = json.loads(jc.labels_path(tmp_path).read_text(encoding="utf-8"))
    assert len(saved["answers"]) == 2 and "picks" not in saved["drawn"] and saved["drawn"]["asked"] == 20
    assert saved["drawn"]["reviewed_known"] == 10
    later = iter(["y"] * 40)
    shown.clear()
    assert jc.review(tmp_path, recs, read=lambda p: next(later), out=shown.append) == 20
    assert len(shown) == 18  # the 2 answered are not asked again


def test_the_review_excludes_exp_27s_ten_reviewed_threads_and_fails_loudly_when_they_are_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "DATA", tmp_path / "data")
    recs = records()
    with pytest.raises(SystemExit):  # nothing found
        jc.reviewed_before()
    with pytest.raises(SystemExit):
        jc.review(tmp_path, recs, read=lambda p: "q", out=lambda s: None)
    old = jc.draw(recs, set())["picks"][:10]
    tc._write(tc.review_path(tc.run_dir()), {"answers": {h: "y" for h in old[:9]}})
    with pytest.raises(SystemExit):  # 9 of 10
        jc.reviewed_before()
    tc._write(tc.review_path(tc.run_dir()), {"answers": {h: "y" for h in old}})
    assert jc.reviewed_before() == set(old)
    jc.review(tmp_path, recs, read=lambda p: "q", out=lambda s: None)
    assert not set(old) & set(jc.draw(recs, jc.reviewed_before())["picks"])


# --- a whole run ---------------------------------------------------------------------------------------------------

def patched(monkeypatch, tmp_path, judge):
    fake = FakeOllama()
    monkeypatch.setattr(tc, "DATA", tmp_path / "data")
    monkeypatch.setattr(cache, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(generate, "writer_client", fake.client)
    monkeypatch.setattr(generate, "warm_up", lambda *a, **k: 0.1)
    monkeypatch.setattr(generate, "placement", lambda *a, **k: (1.0, 8192, []))
    monkeypatch.setattr(tc, "make_judge_client", lambda: judge)
    seed_reviewed()
    return fake


def test_a_whole_run_judges_the_drawn_threads_with_both_judges_and_keeps_no_text_in_the_metrics(tmp_path, monkeypatch):
    judge = ByJudge()
    patched(monkeypatch, tmp_path, judge)
    res = cache.run_or_load(jc.NAME, jc.run, **jc.params())
    assert res["verdict"] == "pending" and res["drawn"]["asked"] == 20 and res["labelled"] == 0
    assert len(judge.calls) == 40 and {s for s, _ in judge.calls} == {tc.JUDGE_SYSTEM, tc.JUDGE_SYSTEM_V2}
    text = (tmp_path / "metrics" / "judge_calibration.json").read_text(encoding="utf-8")
    assert "zz" not in text and "line " not in text and "premise" not in text and not any(t in text for t in jc.TOPICS)
    assert json.loads(text)["params"]["judge_digest_v2"] == tc.judge_digest("v2")
    n = len(judge.calls)
    # the user labels everything "one bit": v1 (A is connected) and v2 (B is connected) are each right on 10 of 20
    bits = jc.prompts()
    recs = jc.gather(jc.run_dir(), bits)
    picks = jc.draw(recs, set())["picks"]
    jc.tc._write(jc.labels_path(jc.run_dir()), {"answers": {h: "y" for h in picks}})
    again = jc.score()
    assert len(judge.calls) == n  # no model call
    assert again["judges"]["v1"]["agree"] == again["judges"]["v2"]["agree"] == 10 and again["verdict"] == "keep_v1_untrusted"
    assert "zz" not in json.dumps(again) and "keep_v1_untrusted" in jc.table(again) and "zz" not in jc.table(again)


def test_a_run_refuses_a_hosted_writer_and_a_stale_call_and_score_refuses_without_bits(monkeypatch, tmp_path):
    monkeypatch.setattr(tc, "DATA", tmp_path / "data")
    monkeypatch.setattr(generate, "writer_client", lambda: llm.Client("http://x", "k", "m"))
    with pytest.raises(SystemExit):
        jc.run(**jc.params())
    with pytest.raises(AssertionError):
        jc.run(**{**jc.params(), "judge_digest_v2": "0" * 8})
    with pytest.raises(SystemExit):
        jc.score()


def test_thread_text_is_gitignored():
    r = subprocess.run(["git", "check-ignore", "-q", "data/thread_coherence/calib-x/bits/f00-A.json"], cwd=ROOT)
    assert r.returncode == 0


# --- re-scoring Exp 27 with the kept judge --------------------------------------------------------------------------

def exp27_folder(monkeypatch, tmp_path):
    fake = FakeOllama()
    monkeypatch.setattr(tc, "DATA", tmp_path / "data")
    monkeypatch.setattr(cache, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(generate, "warm_up", lambda *a, **k: 0.1)
    monkeypatch.setattr(generate, "placement", lambda *a, **k: (1.0, 8192, []))
    tc.generate_bits(fake.client(), tc.prompts(6), tc.run_dir(6), http=ps())


def allow(monkeypatch, **res):
    monkeypatch.setattr(jc, "stored", lambda: res or None)


def test_the_rescore_judges_with_v2_under_its_own_cache_keys_and_keeps_no_text(tmp_path, monkeypatch):
    exp27_folder(monkeypatch, tmp_path)
    allow(monkeypatch, kept="v2", rescore_allowed=True)
    judge = ByJudge()
    res = tc.rescore("v2", 6, tc.SEED, tc.digest(6), tc.judge_digest("v2"), client=judge)
    assert {s for s, _ in judge.calls} == {tc.JUDGE_SYSTEM_V2} and len(judge.calls) == 36
    d = tc.run_dir(6) / "judge_cache"
    assert [p.name for p in d.iterdir()] == [tc.judge_digest("v2")] and len(list((d / tc.judge_digest("v2")).glob("*.json"))) == 36
    assert res["judge"] == "v2" and res["bar"]["passes"] is True and res["arms"]["B"]["threads_connected"] == 18
    assert res["arms"]["A"]["threads_connected"] == 0 and res["judge_failures"] == 0 and "zz" not in json.dumps(res)
    n = len(judge.calls)
    tc.rescore("v2", 6, tc.SEED, tc.digest(6), tc.judge_digest("v2"), client=judge)
    assert len(judge.calls) == n  # cached
    # Exp 27's own v1 record is untouched
    assert not (tc.run_dir(6) / "judge_cache" / tc.judge_digest("v1")).exists()


@pytest.mark.parametrize("state", [None, {"kept": "v2", "rescore_allowed": False}, {"kept": "v1", "rescore_allowed": True},
                                   {"kept": "v2"}])
def test_the_rescore_refuses_unless_exp_30_kept_and_trusted_v2(tmp_path, monkeypatch, state):
    exp27_folder(monkeypatch, tmp_path)
    monkeypatch.setattr(jc, "stored", lambda: state)
    judge = ByJudge()
    with pytest.raises(SystemExit):
        tc.rescore("v2", 6, tc.SEED, tc.digest(6), tc.judge_digest("v2"), client=judge)
    assert judge.calls == []


def test_v1_is_never_re_scored(tmp_path, monkeypatch):
    exp27_folder(monkeypatch, tmp_path)
    allow(monkeypatch, kept="v1", rescore_allowed=True)
    with pytest.raises(SystemExit):
        tc.rescore("v1", 6, tc.SEED, tc.digest(6), tc.judge_digest("v1"), client=ByJudge())


def test_a_stale_rescore_is_refused(tmp_path, monkeypatch):
    allow(monkeypatch, kept="v2", rescore_allowed=True)
    with pytest.raises(AssertionError):
        tc.rescore("v2", 6, tc.SEED, "0" * 8, tc.judge_digest("v2"), client=ByJudge())


# --- noise of the rule ----------------------------------------------------------------------------------------------

def test_the_chance_of_keeping_v2_is_exact_and_a_tie_is_below_a_coin_flip():
    assert jc.p_keep_v2(0.3, 0.8, 1) == pytest.approx(0.8 * 0.7)
    assert jc.p_keep_v2(0.5, 0.5, 2) == pytest.approx(0.3125)
    assert 0.40 < jc.p_keep_v2(0.5, 0.5) < 0.47  # equal judges: v2 about 44% of the time, v1 the rest (ties included)
    assert jc.p_keep_v2(0.4, 0.8) > jc.p_keep_v2(0.5, 0.7) > jc.p_keep_v2(0.5, 0.5)


def test_the_power_simulation_rarely_flags_equal_judges_and_rises_with_the_gap():
    tie = jc.simulate_power(0.5, 0.5, reps=200, n_boot=300)
    big = jc.simulate_power(0.4, 0.8, reps=200, n_boot=300)
    assert tie["ci_lower_above_zero"] < 0.08 and big["ci_lower_above_zero"] > 0.5 and big["keep_v2"] > tie["keep_v2"]


# --- command lines --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["sim.judge_calibration", "sim.thread_coherence", "sim.thread_timing"])
def test_the_commands_take_an_env_file(module):
    r = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and "--env-file" in r.stdout


def test_the_coherence_command_offers_rescore_with_a_judge_choice():
    r = subprocess.run([sys.executable, "-m", "sim.thread_coherence", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert "rescore" in r.stdout and "--judge" in r.stdout


# --- a stored result through the gates -----------------------------------------------------------------------------

def stored_calibration(monkeypatch, tmp_path, v1_right=8, v2_right=18):
    """A real summarize result saved through cache.save and read back with cache.load (not a hand-written dict)."""
    monkeypatch.setattr(cache, "METRICS", tmp_path / "metrics")
    recs = records()
    verdicts, answers = scenario(recs, v1_right, v2_right)
    res = jc.summarize(recs, verdicts, answers, set(), n_boot=200)
    cache.save(jc.NAME, jc.params(), res)
    return res


def test_a_real_summarize_result_saved_and_loaded_allows_the_rescore_only_when_trusted(monkeypatch, tmp_path):
    res = stored_calibration(monkeypatch, tmp_path)
    assert res["rescore_allowed"] is True and jc.stored() == json.loads(json.dumps(res))
    assert tc.rescore_allowed("v2") is True and tc.rescore_allowed("v1") is False
    stored_calibration(monkeypatch, tmp_path, v1_right=15, v2_right=15)  # tie keeps v1
    assert tc.rescore_allowed("v2") is False
    stored_calibration(monkeypatch, tmp_path, v1_right=3, v2_right=12)  # v2 kept, 8 disagreements
    assert tc.rescore_allowed("v2") is False


def test_a_rescore_made_under_an_earlier_exp_30_record_is_stale_and_blocks_adoption(tmp_path, monkeypatch):
    exp27_folder(monkeypatch, tmp_path)
    stored_calibration(monkeypatch, tmp_path)
    res = tc.rescore("v2", 6, tc.SEED, tc.digest(6), tc.judge_digest("v2"), client=ByJudge())
    assert res["exp30"]["verdict"] == "keep_v2_trusted" and res["exp30"]["agree_v2"] == 18 and res["exp30"]["scored"] == 20
    cache.save(tc.RESCORE_NAME, tc.rescore_params("v2"), res)
    assert tc.rescore_status("v2")[0] == "current"
    assert tc.rescore_status("v1")[0] == "none"
    cal = jc.stored()
    assert jc.adoptable()["rescore_current"] is True and jc.adoptable()["v2_kept_and_trusted"] is True
    # the user relabels and Exp 30 comes out differently: the saved re-score no longer matches it
    stored_calibration(monkeypatch, tmp_path, v1_right=8, v2_right=19)
    assert jc.stored() != cal and tc.rescore_status("v2")[0] == "stale"
    a = jc.adoptable()
    assert a["rescore_current"] is False and a["adoptable"] is False


def test_adoption_needs_every_condition_and_one_missing_blocks_it(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "METRICS", tmp_path / "metrics")
    assert jc.adoptable()["adoptable"] is False  # nothing stored
    cal = {"verdict": "keep_v2_trusted", "kept": "v2", "rescore_allowed": True, "scored": 20,
           "judges": {"v1": {"agree": 8}, "v2": {"agree": 15}},
           "adoption_guards": {"user_diff_not_negative": True, "judge_diff_within_user_ci": True}}
    cache.save(jc.NAME, jc.params(), cal)
    prov = tc.exp30_provenance()
    cache.save(tc.RESCORE_NAME, tc.rescore_params("v2"), {"exp30": prov, "bar_passes": True})
    assert jc.adoptable()["adoptable"] is True
    for key in ("user_diff_not_negative", "judge_diff_within_user_ci"):
        cache.save(jc.NAME, jc.params(), {**cal, "adoption_guards": {**cal["adoption_guards"], key: False}})
        assert jc.adoptable()["adoptable"] is False
    cache.save(jc.NAME, jc.params(), cal)
    cache.save(tc.RESCORE_NAME, tc.rescore_params("v2"), {"exp30": prov, "bar_passes": False})
    assert jc.adoptable()["adoptable"] is False


def test_a_user_difference_of_exactly_zero_does_not_contradict_the_judge():
    recs = records()
    picks = jc.draw(recs, set())["picks"]
    same = {h: "y" if i % 2 == 0 else "n" for i, h in enumerate(picks)}
    d = jc.draw(recs, set())
    same = {h: ("y" if sum(d["arm_of"][x] == d["arm_of"][h] for x in picks[:i]) % 2 == 0 else "n") for i, h in enumerate(picks)}
    res = jc.summarize(recs, {"v1": {h: True for h in picks}, "v2": {h: True for h in picks}}, same, set(), n_boot=100)
    assert res["user_one_bit_diff_B_minus_A"]["diff"] == 0.0 and res["adoption_guards"]["user_diff_not_negative"] is True
