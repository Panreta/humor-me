"""Exp 17 record (sim/judge_gate.py) on synthetic eval files: numbers recomputed from per-pair scores, the gate rule."""
import json

from explorer import eval_judge
from sim import judge_gate as jg


def _write(tmp_path, base_scores, adapter_scores):
    def result(scores, adapter):
        acc, lo, hi = eval_judge.mean_ci(list(scores.values()))
        return {"base": "m", "adapter": adapter, "n_pairs": len(scores), "n_pairs_both_orders": len(scores),
                "accuracy": acc, "ci95": [lo, hi], "consistent_share": sum(v != 0.5 for v in scores.values()) / len(scores),
                "share_answered_A": 0.5, "n_nan": 0, "pair_scores": scores, "p_a": []}
    b, a = result(base_scores, None), result(adapter_scores, "x")
    a["vs_baseline"] = eval_judge.paired_diff(adapter_scores, base_scores)
    a["gate_pass"] = eval_judge.gate(a)
    (tmp_path / "base.json").write_text(json.dumps(b))
    (tmp_path / "adapter.json").write_text(json.dumps(a))


def test_good_judge_passes_and_agrees_with_eval_script(tmp_path):
    keys = [f"k{i}" for i in range(400)]
    base = {k: [1.0, 0.5, 0.5, 0.0][i % 4] for i, k in enumerate(keys)}      # accuracy 0.5
    trained = {k: [1.0, 1.0, 0.5, 0.0][i % 4] for i, k in enumerate(keys)}   # accuracy 0.625
    _write(tmp_path, base, trained)
    res = jg.run(tmp_path)
    t = res["arms"]["trained"]
    assert (t["right_both"], t["right_one"], t["right_neither"]) == (200, 100, 100)
    assert abs(t["consistent_share"] - 0.75) < 1e-12 and abs(t["accuracy_when_consistent"] - 200 / 300) < 1e-12
    assert res["paired_diff"]["trained_better"] == 100 and res["paired_diff"]["untrained_better"] == 0
    assert res["gate_pass"] and all(res["eval_script_agrees"].values())


def test_always_A_judge_fails_the_gate(tmp_path):
    keys = [f"k{i}" for i in range(400)]
    flip = {k: 0.5 for k in keys}  # right in exactly one order of every pair
    _write(tmp_path, flip, flip)
    res = jg.run(tmp_path)
    assert res["arms"]["trained"]["consistent_share"] == 0 and not res["gate_pass"]
    assert all(res["eval_script_agrees"].values())
