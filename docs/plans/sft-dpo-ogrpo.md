# Model track, simplified: SFT → DPO → O-GRPO

Where this fits: replaces the model-track steps in [status.md](../status.md) "Next steps" with the
HumorGen recipe (`references/md/2604-09629v3.md`, Sections 4–5 and Appendix H). User decision 2026-11-24.
O-GRPO here means **offline** GRPO, as in HumorGen. Detailed judge work stays in [DPO_training.md](DPO_training.md).

## The pipeline

One candidate pool, ranked once, feeds all three training stages.

| Step | What | Code | State |
|---|---|---|---|
| 0. Judge | Qwen2.5-7B + LoRA, DPO-trained to answer A/B on rJokes upvote pairs (LoL's frozen judge idea) | `explorer/train_dpo.py`, `explorer/eval_judge.py` | 🔨 last run stopped at 0.32 epoch, reward accuracy at chance; pair builder for `/scratch/tingyu.z/dpo/dpo_*.jsonl` not committed |
| 1. Pool | Own premises × 24 candidates (6 modes × 4). No rJokes or transcript lines as material | 🔨 `explorer/gen_candidates.py` | 🔨 |
| 2. Rank | Judge plays all pairs both orders per premise → Bradley–Terry θ | `judge/rank.py` + 🔨 `DPOJudge` in `judge/judges.py` (P(A) from A/B logits, as in `eval_judge.py`) | 🔨 |
| 3. Sets | SFT: top 10 per premise. DPO: 5 pairs, random top-5 vs bottom-5. O-GRPO: all 24 with A = group z-score of θ | 🔨 `explorer/build_sets.py` | 🔨 |
| 4. Train | SFT from base; DPO and O-GRPO each from the SFT checkpoint. O-GRPO = SFT loss weighted by softmax(A/T) over the group, T = 1 | 🔨 `train_sft.py`; `train_dpo.py` (reuse with `--data_dir`); 🔨 `train_ogrpo.py` | 🔨 |
| 5. Eval | Held-out premises, base vs SFT vs DPO vs O-GRPO, paired CIs, **a judge other than the one that built the sets** | 🔨 | 🔨 |

## Gates and known results

- **Judge gate:** the trained judge's 95% CI on rJokes test accuracy lies above the base model's and above 0.5.
  Otherwise steps 1–5 learn noise.
- HumorGen found DPO ≈ SFT and O-GRPO < SFT. A tie here replicates them; fix pass criteria in
  `decisions.md` before running.
- Open: with softmax weights every candidate gets a positive weight, so nothing is penalized, yet the paper
  explains O-GRPO's loss by weak jokes being penalized. Decide which form to replicate before building step 4.

## Later: ideas to try after the baseline

Only after SFT, DPO and O-GRPO have a baseline score. Each needs its own pass criterion in `decisions.md` first.

| Idea | Source | What | Why later |
|---|---|---|---|
| Answer rewriting | LoL, Section 2.2 and Figure 2(c) (`references/md/2024.md`) | An LLM rewrites each top-ranked line into a flat version. Use it as extra DPO pairs (funny = chosen, flat = rejected) and as SFT examples for flat → funny | Cheap, and the quality gap is guaranteed without the judge. Unmeasured until the baseline exists |

Considered and skipped: LoL's teacher–student loop (costly, multi-turn), automatic instruction expansion, and
rationale-mixed DPO (HumorGen's explainer trap, Section 6.3). Judgment call, not measured.
