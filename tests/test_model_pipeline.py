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
