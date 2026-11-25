# Model track, simplified: SFT → DPO → O-GRPO

Where this fits: replaces the model-track steps in [status.md](../status.md) "Next steps" with the
HumorGen recipe (`references/md/2604-09629v3.md`, Sections 4–5 and Appendix H). User decision 2026-11-24.
O-GRPO here means **offline** GRPO, as in HumorGen. Detailed judge work stays in [DPO_training.md](DPO_training.md).

## The pipeline

One candidate pool, ranked once, feeds all three training stages. **State 2026-11-25: all code built, smoke-tested
on Qwen2.5-0.5B and unit-tested; nothing run on the cluster yet.** Build record: [model-pipeline-log.md](model-pipeline-log.md).
Branch `model-pipeline`.

| Step | What | Code | State |
|---|---|---|---|
| 0. Judge | Qwen2.5-7B + LoRA, DPO-trained to answer A/B on rJokes upvote pairs (LoL's frozen judge idea); gate = Exp 17 | `build_judge_pairs.py`, `train_dpo.py --stage judge`, `eval_judge.py` | ✅ built. Pair builder reproduces the cluster files byte for byte. ⏳ run |
| 1. Pool | Premises (option A: teacher-written; 1 in 5 held out) × 24 candidates (6 modes × 4) from the Qwen2.5-32B teacher. No rJokes or transcript lines as material | `gen_premises.py`, `gen_candidates.py`, `joke_prompts.py` | ✅ built. ⏳ premise source to confirm, then run |
| 2. Rank | Safety rules (teens), then the judge scores all ordered pairs per premise → Bradley–Terry θ | `rank_pool.py`, `judge.judges.LogitJudge`, `judge.rank.rank_from_probs` | ✅ built. ⏳ run after the gate |
| 3. Sets | SFT top 10; DPO within-mode best vs worst (or `--no-mode`: HumorGen's top-5 vs bottom-5); O-GRPO all lines, A = group z-score, weight = softmax(A)·G | `build_sets.py` | ✅ built |
| 4. Train | SFT from base; DPO and O-GRPO each from the merged SFT model | `train_sft.py --stage sft/ogrpo`, `train_dpo.py --stage generator` | ✅ built. ⏳ run |
| 5. Eval | ~230 held-out premises, base vs SFT vs DPO vs O-GRPO, paired win rates, **Qwen2.5-32B judge, not the one that built the sets**; Exp 23 | `eval_generators.py generate / judge` | ✅ built. ⏳ run |

Proposed pass criteria for Exp 17 and Exp 23: decisions.md 2026-11-25 (**the user confirms them before any run**).

## Runbook (cluster, repo root, branch `model-pipeline`)

Each line is one job; wait for it to finish, then copy its outputs back to the laptop (course storage is deleted
after grades). `R=$COURSE_DIR/runs`. Every job resumes if resubmitted unchanged.

```bash
uv pip install -r explorer/requirements-cluster.txt --index-strategy unsafe-best-match   # adds trl, peft, httpx
# step 0: judge pairs, training, gate (Exp 17)
python -m explorer.build_judge_pairs --splits data/processed/rjokes --out /scratch/$USER/dpo     # CPU, ~1 min
sbatch explorer/train_dpo.sbatch --stage judge                                # ~1.6 h for 20k records
sbatch explorer/eval_judge.sbatch --out runs/judge_eval/base.json
sbatch explorer/eval_judge.sbatch --adapter $R/dpo_judge_v1/final --out runs/judge_eval/adapter.json \
    --baseline runs/judge_eval/base.json                                      # prints gate_pass
# step 1 (only after gate_pass is true); first use of the 32B: HF_HUB_OFFLINE=0 to download it (~65 GB)
HF_HUB_OFFLINE=0 sbatch explorer/gen_premises.sbatch                          # data/premises/premises-v1.jsonl
#   -> hand-check the premises file, then commit it (our model's text, not Reddit)
sbatch explorer/gen_candidates.sbatch                                         # runs/pool/candidates.jsonl
# step 2
sbatch explorer/rank_pool.sbatch --judge-adapter $R/dpo_judge_v1/final        # runs/pool/ranked.jsonl
# step 3 (CPU)
python -m explorer.build_sets                                                 # runs/sets/*, read stats.json
# step 4
sbatch explorer/train_sft.sbatch --stage sft
sbatch explorer/train_dpo.sbatch --stage generator --data_dir runs/sets --init_adapter $R/sft_v1/final
sbatch explorer/train_sft.sbatch --stage ogrpo --init_adapter $R/sft_v1/final
# step 5 (Exp 23)
sbatch explorer/eval_generators.sbatch generate --models base sft=$R/sft_v1/final \
    dpo=$R/dpo_generator_v1/final ogrpo=$R/ogrpo_v1/final
sbatch explorer/eval_generators.sbatch judge                                  # runs/eval/report.json
```

**Look before moving on:** after step 0, `share_answered_A` (position bias) and `consistent_share`; after step 2,
`stats.json`'s `judge_consistent_share_median` (low = the ranking is mostly noise), `unsafe_share`,
`dropped_flat_premises`; after step 1, the job log's "short cells" count.

## Data on hand: `NewlyData/` (teammate, 2026-11-24, gitignored)

| File | Rows | What |
|---|---|---|
| `dpo_state.jsonl` | 89,924 | The 90k labeled rJokes **train** jokes (same ids as `modes.jsonl`) with text, score, pct, mode, safety, plus room labels `venue`, `energy`, `age` inferred by Qwen2.5-7B |
| `state.jsonl` | 90,000 | The raw room-label output; 31 rows failed to parse (`raw`, e.g. venue `comedy-club_showcase`) |

How it fits:
- **Judge only (step 0).** rJokes may train and test judges, never supply joke material (decisions.md 2026-10-31,
  2026-11-01), so these jokes can't enter SFT, DPO or O-GRPO. All rows are from `train`, so the judge test on
  `test` pairs stays clean. Drop the 23,107 rows with any safety flag, as for the existing pairs.
- **Room labels are inferred, not measured, and skewed:** showcase 95%, high energy 85%, young adults 96%
  (company_event 555, low energy 398, teens 1). A room-conditioned judge would mostly learn "showcase". Train
  the plain judge first; add the room to the judge prompt only if a paired test on the bar and company_event
  rows can show a difference, which 4,322 and 555 rows may not support.
- Open: ask the teammate what `dpo_state` was built for (room-conditioned DPO pairs?) and with what prompt.

## Gates and known results

- **Judge gate:** the trained judge's 95% CI on rJokes test accuracy lies above the base model's and above 0.5.
  Otherwise steps 1–5 learn noise.
- HumorGen found DPO ≈ SFT and O-GRPO < SFT. A tie here replicates them; fix pass criteria in
  `decisions.md` before running.
- Built as HumorGen writes it (Appendix H, eq. 4–5): softmax weights, all positive, so weak lines are down-weighted,
  never pushed away. The paper's explanation (weak jokes "penalized") does not match its own formula; a variant
  with signed advantages would be a different method. Judgment call, not measured.

## Later: ideas to try after the baseline

Only after SFT, DPO and O-GRPO have a baseline score. Each needs its own pass criterion in `decisions.md` first.

| Idea | Source | What | Why later |
|---|---|---|---|
| Answer rewriting | LoL, Section 2.2 and Figure 2(c) (`references/md/2024.md`) | An LLM rewrites each top-ranked line into a flat version. Use it as extra DPO pairs (funny = chosen, flat = rejected) and as SFT examples for flat → funny | Cheap, and the quality gap is guaranteed without the judge. Unmeasured until the baseline exists |

Considered and skipped: LoL's teacher–student loop (costly, multi-turn), automatic instruction expansion, and
rationale-mixed DPO (HumorGen's explainer trap, Section 6.3). Judgment call, not measured.
