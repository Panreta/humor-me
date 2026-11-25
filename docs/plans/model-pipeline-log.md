# Model pipeline build log

Where this fits: the step-by-step record of building the SFT → DPO → O-GRPO pipeline in
[sft-dpo-ogrpo.md](sft-dpo-ogrpo.md). Newest session at the top. Each entry says what was done, why, how it
was checked, and what is left. Branch: `model-pipeline` (off `judge-api`, stacking approved by the user
2026-11-24).

---

## Session 1: overnight build, 2026-11-24 → 25 (Claude, unattended)

**Brief from the user:** write and test all code for steps 0–4 so the morning is only cluster submissions. No
GPU jobs (no SSH to Explorer from the laptop), no premise generation (source not decided), no joke text in
git, no PR.

### 1.0 Setup

- **Worktree:** `../humor-me-model-pipeline` on new branch `model-pipeline`, created from `judge-api` at
  `889d799`. The main checkout has the user's uncommitted edits (decisions.md, the notebooks moved to
  `scrolls/`), so a separate worktree keeps them untouched (CLAUDE.md: each session uses its own worktree).
- **Local Python:** a venv in Claude's scratch folder, not the system Python. Pinned to what the cluster runs
  so local tests mean something there: `transformers 4.57.6`, `trl 0.24.0`, `peft 0.17.1`, `torch 2.15`
  (CPU). The cluster has `torch 2.5.1`; nothing below relies on a newer torch.
- **Smoke model:** `Qwen/Qwen2.5-0.5B-Instruct`, downloaded to the scratch HF cache. Same tokenizer and chat
  template as the 7B, so the A/B token ids, prompt formatting and LoRA targets are tested for real; only the
  size differs.
- **Baseline tests before any change:** 455 passed, 3 skipped, **2 failed**:
  `tests/test_scripts_import.py` for `explorer.train_dpo` and `explorer.eval_judge`. Cause: both scripts parse
  arguments and read `os.environ["COURSE_DIR"]` at import time, so once `trl` is installed the import test
  runs them with pytest's arguments. On the user's laptop `trl` wasn't installed, so the test skipped and
  hid it. Fix: put each script's body in `main()` (step 1.2).

### 1.1 Found: the "missing" judge pair builder

The plan said the script that wrote `/scratch/tingyu.z/dpo/dpo_{train,dev,test}.jsonl` was never committed.
It exists on the laptop, next to its outputs: `data/DPO_set/data/processed/rjokes/make_dpo_pairs.py`
(gitignored folder). What it does:

1. Keep jokes with a known `year` and no keyword `flags`.
2. Cells = year × length quartile (quartile cuts computed from each split's own lengths).
3. In each cell, pair a high joke (`pct ≥ 0.8`) with a low one (`pct ≤ 0.3`), each joke used once.
4. Write each pair **twice**, once per order, so "A" and "B" are each the right answer half the time
   (45,761 each in train). Prompt: `Which joke is funnier?\nJoke A: …\nJoke B: …\nAnswer with A or B.`
5. Shuffle with seed 17.

Outputs on the laptop: train 91,522 records (45,761 pairs), dev 10,872, test 10,636 (5,318 pairs, the number
in DPO_training.md). The logged run (`dpo_judge.11878130`) used only 1,998 of them for 20 steps, which is
why it ended at chance.

**Checked:** `explorer/build_judge_pairs.py` (step 1.2) run on the laptop's splits reproduces all three files
**byte for byte** (`cmp`: train, dev, test identical). So the committed script is the one that made the data.

**Data checks on the pairs:**
- No test or dev joke appears verbatim in train. After lower-casing and stripping punctuation, 334 of the
  10,636 test jokes (3%) match a train joke: reposts. Small; left in, noted here. Judgment call, not measured.
- Every pair is in the file twice (both orders). The old `eval_judge.py` counted the 10,636 test records as
  independent, which makes its CI about √2 too narrow. Fixed in 1.2.

### 1.2 Step 0 code: judge training and the gate

| File | What changed | Why |
|---|---|---|
| `explorer/build_judge_pairs.py` | New: the laptop script, importable, `--splits/--out` flags | Rebuildable on the cluster; tested |
| `explorer/judge_model.py` | New: load a judge (base + optional adapter), `score(prompts) -> P(A)` in length-sorted batches, OOM split, NaN for a prompt that can't fit | One scoring path for the gate and for ranking the pool (step 2) |
| `explorer/eval_judge.py` | Rewritten: `main()`, batched, **scored per pair** (1 / 0.5 / 0), CI over pairs, `--baseline` gives the paired difference and `gate_pass`; pair keys hashed so results hold no joke text; `--n` replaced by `--n_pairs` | The gate as defined in the plan; result files safe to copy back |
| `explorer/train_common.py` | New: `load_policy` (merges an earlier stage's adapter), `lora`, precision pick, resume helper | Shared by all three trainers |
| `explorer/train_dpo.py` | `main()`; `--stage judge` (defaults = the 2026-11-14 run) or `--stage generator` (HumorGen Table 19: lr 5e-7 constant, warmup 0.1, 5 epochs, global batch 16, early stopping patience 2, needs `--init_adapter`) | One DPO script for both uses |
| `*.sbatch` | `python -m explorer.<name>` | New scripts import `judge/`, which needs the repo root on the path |

**Smoke runs** (Qwen2.5-0.5B-Instruct, CPU):
- `train_dpo --stage judge`, 16 records, 2 steps: trains, evaluates, saves `final/` with `run_args.json`.
  `logps/chosen ≈ -33` looked wrong for a one-letter answer; inspected the formatted example: the completion is
  `A<|im_end|>\n` (3 tokens), correct. The low value is the shared `<|im_end|>\n` suffix, which cancels in the
  DPO margin. Not a bug.
- `eval_judge` on 200 test pairs: the 0.5B model answers **"A" 99.75% of the time**, scores 0.5025, consistent
  on 0.5% of pairs. Exactly what an always-A judge should score, so per-pair scoring works. Position bias is
  the first thing to look at in the 7B baseline too.

**Bug caught by the smoke run:** the first version cut `--n` records, not pairs, so most pairs had one order
and counted as "consistent" (98% for an always-A judge). Fixed: `--n_pairs` keeps both orders; consistency
counts only pairs with both. Covered by `test_first_pairs_keeps_both_orders` and
`test_always_A_judge_scores_exactly_chance_and_never_consistent`.

**Tests:** `tests/test_model_pipeline.py` (pair rule, determinism, scoring, gate, stage presets). The two
baseline import failures now pass.

**Morning note, judge run length:** 91,522 records ÷ 32 per step = 2,860 steps. The logged H200 run took
about 9 s per step, so a full epoch is about 7 h, longer than `train_dpo.sbatch`'s 4 h. The default stays at
20,000 records (625 steps, about 1.6 h); resubmitting resumes from the last checkpoint if you want more.
