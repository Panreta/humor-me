"""The SFT -> DPO -> O-GRPO pipeline (docs/plans/sft-dpo-ogrpo.md): data builders, scoring and the gate.
Synthetic jokes only; no network, no model weights."""
import random

import pytest

from explorer import build_judge_pairs as bjp
from explorer import eval_judge as ej
from explorer import train_dpo


def joke_rows(n=400, seed=0):
    rng = random.Random(seed)
    return [{"id": f"j{i}", "text": " ".join(["w"] * rng.randint(5, 40)) + f" #{i}",
             "pct": rng.random(), "year": rng.choice([2016, 2017]), "flags": ["kw"] if i % 10 == 0 else []}
            for i in range(n)]


# --- step 0: judge pairs -------------------------------------------------------------------------------

def test_judge_pairs_follow_the_rule():
    rows = joke_rows()
    by_text = {r["text"]: r for r in rows}
    recs = bjp.build(rows)
    assert recs and len(recs) % 2 == 0
    assert sum(r["chosen"] == "A" for r in recs) == len(recs) // 2      # both orders: A and B balanced
    used = []
    for r in recs:
        m = ej.PAIR.match(r["prompt"])
        a, b = by_text[m.group(1)], by_text[m.group(2)]
        win, lose = (a, b) if r["chosen"] == "A" else (b, a)
        assert win["pct"] >= bjp.HIGH and lose["pct"] <= bjp.LOW         # high vs low
        assert win["year"] == lose["year"]                                 # same cell year
        assert not win["flags"] and not lose["flags"]                      # unflagged only
        assert r["rejected"] != r["chosen"]
        used += [a["id"], b["id"]]
    assert all(used.count(i) == 2 for i in set(used))                      # each joke in one pair, two orders


def test_judge_pairs_are_deterministic():
    assert bjp.build(joke_rows()) == bjp.build(joke_rows())
    assert bjp.build(joke_rows(), seed=1) != bjp.build(joke_rows())


# --- step 0: judge eval and gate ------------------------------------------------------------------------

def both_orders(n_pairs):
    rows = []
    for i in range(n_pairs):
        a, b = f"good {i}", f"bad {i}"
        rows.append({"prompt": bjp.PROMPT.format(a=a, b=b), "chosen": "A"})
        rows.append({"prompt": bjp.PROMPT.format(a=b, b=a), "chosen": "B"})
    return rows


def test_pair_key_ignores_order():
    rows = both_orders(1)
    assert ej.pair_key(rows[0]["prompt"]) == ej.pair_key(rows[1]["prompt"])
    assert "good" not in ej.pair_key(rows[0]["prompt"])                    # hashed: no joke text in results


def test_always_A_judge_scores_exactly_chance_and_never_consistent():
    rows = both_orders(50)
    s = ej.summarize(rows, [0.9] * len(rows))
    assert s["accuracy"] == 0.5 and s["consistent_share"] == 0 and s["share_answered_A"] == 1


def test_perfect_judge_scores_one():
    rows = both_orders(50)
    s = ej.summarize(rows, [0.9 if r["chosen"] == "A" else 0.1 for r in rows])
    assert s["accuracy"] == 1 and s["consistent_share"] == 1


def test_nan_scores_count_as_wrong():
    rows = both_orders(2)
    s = ej.summarize(rows, [float("nan")] * 4)
    assert s["accuracy"] == 0 and s["n_nan"] == 4


def test_first_pairs_keeps_both_orders():
    rows = both_orders(10)
    random.Random(0).shuffle(rows)
    sub = ej.first_pairs(rows, 3)
    assert len(sub) == 6 and ej.summarize(sub, [0.9] * 6)["n_pairs_both_orders"] == 3


def test_gate_needs_both_above_chance_and_above_baseline():
    rows = both_orders(200)
    good = ej.summarize(rows, [0.9 if r["chosen"] == "A" else 0.1 for r in rows])
    base = ej.summarize(rows, [0.9] * len(rows))
    good["vs_baseline"] = ej.paired_diff(good["pair_scores"], base["pair_scores"])
    assert good["vs_baseline"]["diff"] == pytest.approx(0.5) and ej.gate(good)
    base["vs_baseline"] = ej.paired_diff(base["pair_scores"], base["pair_scores"])
    assert not ej.gate(base)


# --- train_dpo: stage presets ---------------------------------------------------------------------------

def test_train_dpo_judge_defaults_match_the_first_run():
    a = train_dpo.parse_args(["--stage", "judge", "--out_dir", "x"])
    assert (a.lr, a.epochs, a.batch * a.accum, a.lora_alpha, a.max_prompt_tokens) == (2e-5, 1, 32, 32, 448)


def test_train_dpo_generator_uses_humorgen_settings_and_needs_sft():
    a = train_dpo.parse_args(["--stage", "generator", "--init_adapter", "sft", "--out_dir", "x"])
    assert (a.lr, a.scheduler, a.epochs, a.batch * a.accum, a.patience) == (5e-7, "constant_with_warmup", 5, 16, 2)
    with pytest.raises(SystemExit):
        train_dpo.parse_args(["--stage", "generator", "--out_dir", "x"])


def test_to_chat_wraps_judge_strings_and_passes_messages():
    judge = train_dpo.to_chat({"prompt": "p", "chosen": "A", "rejected": "B"})
    assert judge["chosen"] == [{"role": "assistant", "content": "A"}]
    msgs = {"prompt": [{"role": "user", "content": "p"}], "chosen": [{"role": "assistant", "content": "x"}],
            "rejected": [{"role": "assistant", "content": "y"}], "extra": 1}
    assert train_dpo.to_chat(msgs) == {k: msgs[k] for k in ("prompt", "chosen", "rejected")}


# --- step 1: prompts, premises, candidates --------------------------------------------------------------

from explorer import build_sets, gen_candidates, gen_premises, joke_prompts, rank_pool  # noqa: E402
from judge.rank import ordered_pairs, rank_from_probs  # noqa: E402


def test_modes_match_the_live_generator():
    pytest.importorskip("httpx")
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "backend"))
    from app.generate import MODES
    assert joke_prompts.MODES == MODES


def test_parse_lines_cleans_and_caps():
    reply = 'Sure! ["one joke", "one joke", 3, "  two   jokes ", "???", "' + "w " * 50 + '", "three"]'
    assert joke_prompts.parse_lines(reply) == ["one joke", "two jokes", "three"]
    assert joke_prompts.parse_lines(reply, 2) == ["one joke", "two jokes"]
    assert joke_prompts.parse_lines("no list here") == [] and joke_prompts.parse_lines("[broken") == []


def test_student_prompt_with_and_without_mode():
    with_mode = joke_prompts.student_messages("queues", "cynic")[1]["content"]
    assert "cynic" in with_mode and "queues" in with_mode
    assert joke_prompts.student_messages("queues")[1]["content"] == "Premise: queues"


def test_premises_dedupe_and_split_stably():
    replies = [("work", '["The office printer", "office printer!", "a meeting that could be an email"]'),
               ("food", '["Microwave popcorn", "x"]')]
    rows = gen_premises.collect(replies, n=10)
    assert [r["premise"] for r in rows] == ["The office printer", "Microwave popcorn",
                                            "a meeting that could be an email"]   # round-robin, dup and 1-word gone
    assert all(r["split"] == gen_premises.split_of(r["id"]) for r in rows)
    assert gen_premises.premise_id("The office printer") == gen_premises.premise_id("office printer!!")
    ids = [gen_premises.premise_id(f"premise number {i}") for i in range(2000)]
    assert 0.16 < sum(gen_premises.split_of(i) == "eval" for i in ids) / 2000 < 0.24


def test_fill_retries_short_cells_and_never_invents():
    calls = []

    def fake_generate(convs):
        calls.append(len(convs))
        return ['["a", "b"]' if len(calls) == 1 else '["c", "d", "e"]' for _ in convs]

    premises = [{"id": "p1", "premise": "queues", "split": "train"}]
    rows, short = gen_candidates.fill(premises, fake_generate, per_mode=4, retries=1)
    assert calls == [6, 6] and short == 0
    assert all(sum(r["mode"] == m for r in rows) == 4 for m in joke_prompts.MODES)
    rows, short = gen_candidates.fill(premises, lambda c: ["[]"] * len(c), per_mode=4, retries=2)
    assert rows == [] and short == 6


# --- step 2: ranking ------------------------------------------------------------------------------------

def test_rank_from_probs_recovers_the_true_order():
    strength = [0.0, 1.0, 2.0, 3.0, 4.0]
    p = {(i, j): 1 / (1 + 2.718 ** -(strength[i] - strength[j])) for i, j in ordered_pairs(5)}
    rk = rank_from_probs(5, p, "fake")
    assert list(rk.order()) == [4, 3, 2, 1, 0] and rk.n_calls == 20 and rk.stop_reason == "exhausted"


def test_always_first_judge_gives_a_flat_ranking():
    rk = rank_from_probs(4, {pair: 0.99 for pair in ordered_pairs(4)}, "biased")
    assert max(rk.theta) - min(rk.theta) < 1e-9          # build_sets drops these premises


class FakeJudge:
    name = "fake"

    def message(self, prompt, first, second):
        return f"{first}|{second}"

    def score(self, prompts):                              # longer line wins
        return [1.0 if len(a) > len(b) else 0.0 for a, b in (p.split("|") for p in prompts)]


def candidates(premise_id="p1", split="train", texts=None):
    texts = texts or [f"line {'x' * i}" for i in range(12)]
    modes = list(joke_prompts.MODES)
    return [{"premise_id": premise_id, "premise": "queues", "split": split, "mode": modes[i % 6], "k": i // 6,
             "text": t} for i, t in enumerate(texts)]


def test_rank_chunk_screens_scores_and_skips():
    unsafe = lambda t: (False, "test") if "bad" in t else (True, "ok")
    good = candidates()
    small = candidates("p2", texts=["a", "b", "bad"])
    dup = candidates("p3", texts=[f"line {'x' * i}" for i in range(12)] + ["line "])
    rows, skipped = rank_pool.rank_chunk([good, small, dup], FakeJudge(), unsafe, min_lines=8)
    assert skipped == ["p2"] and len(rows) == 2
    r = rows[0]
    assert r["n_calls"] == 12 * 11 and r["consistent_share"] == 1.0
    assert [l["text"] for l in r["lines"]] == sorted((c["text"] for c in good), key=len, reverse=True)
    assert len(rows[1]["lines"]) == 12                     # the exact duplicate was dropped


# --- step 3: sets ---------------------------------------------------------------------------------------

def ranked(premise_id, split="train", n=24, flat=False):
    modes = list(joke_prompts.MODES)
    lines = [{"text": f"{premise_id} line {i}", "mode": modes[i % 6], "k": i // 6,
              "theta": 0.0 if flat else float(n - i), "se": 0.1, "borda": 0.0} for i in range(n)]
    return {"premise_id": premise_id, "premise": f"premise {premise_id}", "split": split, "lines": lines,
            "unsafe": [], "consistent_share": 0.9, "first_shown_win_share": 0.5}


def pids(n, start=1):
    """Hex ids that are train premises (not eval: id % 5 != 0)."""
    out, i = [], start
    while len(out) < n:
        if i % 5:
            out.append(f"{i:010x}")
        i += 1
    return out


def test_sets_follow_humorgen_rules():
    rows = [ranked(p) for p in pids(60)] + [ranked("00000000a0", split="eval"), ranked(pids(1, 900)[0], flat=True)]
    sets, stats = build_sets.build(rows)
    assert stats["dropped_flat_premises"] == 1 and stats["eval_premises"] == 1
    assert sets["sft_dev"] and sets["sft_train"]                                       # some premises are dev
    train_ids = {r["premise_id"] for r in sets["sft_train"]}
    assert not train_ids & {r["premise_id"] for r in sets["sft_dev"]}                  # dev disjoint
    assert "00000000a0" not in train_ids | {r["premise_id"] for k in sets for r in sets[k] if k != "eval_prompts"}
    per = {}
    for r in sets["sft_train"]:
        per.setdefault(r["premise_id"], []).append(r["theta"])
    assert all(sorted(v, reverse=True) == [24.0 - i for i in range(10)] for v in per.values())   # top 10
    for r in sets["dpo_train"]:
        assert r["theta_gap"] > 0 and r["mode"] in r["prompt"][1]["content"]           # within one mode
    assert len(sets["dpo_train"]) == 6 * len(per)
    sums = {}
    for r in sets["ogrpo_train"]:
        sums[r["premise_id"]] = sums.get(r["premise_id"], 0) + r["weight"]
    assert all(abs(s - 24) < 1e-9 for s in sums.values())                              # w * G sums to G


def test_eval_prompts_come_from_the_premise_file_too():
    premises = [{"id": "00000000a0", "premise": "eval one", "split": "eval"},
                {"id": "00000000a1", "premise": "train one", "split": "train"}]
    sets, stats = build_sets.build([ranked(p) for p in pids(3)], premises=premises)
    assert [e["premise"] for e in sets["eval_prompts"]] == ["eval one"] and stats["eval_premises"] == 1
    assert sets["eval_prompts"][0]["mode"] in joke_prompts.MODES


def test_dev_and_eval_ids_never_overlap():
    ids = [f"{i:010x}" for i in range(5000)]
    assert not any(build_sets.is_dev(i) and gen_premises.split_of(i) == "eval" for i in ids)


def test_sets_without_mode_use_global_pairs():
    sets, stats = build_sets.build([ranked(p) for p in pids(5)], use_mode=False)
    assert not stats["mode_in_prompt"]
    for r in sets["dpo_train"] + sets["dpo_dev"]:
        assert r["prompt"][1]["content"].startswith("Premise:")
        assert r["theta_gap"] >= 24 - 5 - 5 + 1                                       # top 5 vs bottom 5


def test_advantages_are_z_scores_with_softmax_weights():
    a, w = build_sets.advantages([3.0, 1.0, 1.0, 1.0])
    assert abs(sum(a)) < 1e-9 and a[0] > 0 > a[1]
    assert abs(sum(w) - 1) < 1e-12 and w[0] > w[1] == w[2] == w[3]


# --- step 4: the weighted loss --------------------------------------------------------------------------

def test_weighted_loss_is_weighted_mean_of_per_line_losses():
    torch = pytest.importorskip("torch")
    from explorer.train_sft import weighted_loss
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 5, -100, -100]])
    one = weighted_loss(logits, labels, torch.tensor([1.0, 1.0]))
    l0 = weighted_loss(logits[:1], labels[:1], torch.tensor([1.0]))
    l1 = weighted_loss(logits[1:], labels[1:], torch.tensor([1.0]))
    assert torch.isclose(one, (l0 + l1) / 2)
    assert torch.isclose(weighted_loss(logits, labels, torch.tensor([2.0, 0.0])), l0)


def test_weighted_trainer_lets_the_trainer_scale_for_accumulation():
    pytest.importorskip("transformers")
    from explorer.train_sft import make_trainer_class
    src = make_trainer_class().__init__.__code__.co_names
    assert "model_accepts_loss_kwargs" in src


def test_train_sft_stage_presets():
    from explorer import train_sft
    s = train_sft.parse_args(["--stage", "sft", "--out_dir", "x"])
    assert (s.lr, s.epochs, s.batch * s.accum, s.weighted) == (2e-4, 3, 16, False)
    o = train_sft.parse_args(["--stage", "ogrpo", "--init_adapter", "sft", "--out_dir", "x"])
    assert (o.lr, o.epochs, o.weighted, o.warmup_steps) == (1e-6, 5, True, 10)
    with pytest.raises(SystemExit):
        train_sft.parse_args(["--stage", "ogrpo", "--out_dir", "x"])


# --- step 5: evaluation ---------------------------------------------------------------------------------

from explorer import eval_generators as eg  # noqa: E402


def eval_lines(n_premises=40, models=("base", "sft", "dpo")):
    return [{"model": m, "premise_id": f"e{i}", "premise": "p", "mode": "cynic", "text": f"{m} {i}"}
            for i in range(n_premises) for m in models]


def fake_results(lines, strength):
    """A consistent judge: the stronger model wins in both orders."""
    return {(pid, a, b): 1.0 if strength[a] > strength[b] else 0.0 for pid, a, b in eg.jobs(lines)}


def test_eval_jobs_cover_every_ordered_model_pair():
    lines = eval_lines(3)
    assert len(eg.jobs(lines)) == 3 * 3 * 2


def test_eval_win_rates_and_verdicts():
    lines = eval_lines()
    pts = eg.pair_points(fake_results(lines, {"base": 0, "sft": 1, "dpo": 2}))
    assert eg.win_rate(pts, "sft", "base")["win_rate"] == 1 and eg.win_rate(pts, "sft", "base")["verdict"] == "better"
    assert eg.win_rate(pts, "base", "dpo")["verdict"] == "worse"
    biased = {k: 1.0 for k in eg.jobs(lines)}                         # always the first shown
    assert eg.win_rate(eg.pair_points(biased), "dpo", "sft") == {"n": 40, "win_rate": 0.5, "ci95": [0.5, 0.5],
                                                                 "verdict": "inconclusive"}


def test_eval_leaderboard_orders_models():
    lines = eval_lines(20)
    pts = eg.pair_points(fake_results(lines, {"base": 0, "sft": 1, "dpo": 2}))
    board = eg.leaderboard(pts, ["base", "dpo", "sft"], boot=50)
    assert list(board) == ["dpo", "sft", "base"]
    assert all(v["ci95"][0] <= v["theta"] <= v["ci95"][1] for v in board.values())


def test_eval_parses_model_specs_and_cleans_replies():
    assert eg.parse_models(["base", "sft=/runs/sft/final"]) == {"base": None, "sft": "/runs/sft/final"}
    assert eg.clean('\n  "My joke."  \nexplanation') == "My joke."
