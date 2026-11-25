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

### 1.3 Steps 1–4 code: premises, candidates, ranking, sets, SFT and O-GRPO

| Step | File | What it does |
|---|---|---|
| 1 | `explorer/joke_prompts.py` | The teacher prompt (one per mode, technique after HumorGen Appendix J, recast as the one robot identity, asks for N lines as JSON) and the **student** prompt the 7B sees everywhere (identity + rules + `Mode: …` + `Premise: …`, or premise only). `MODES` text = the live generator's (a test keeps them equal). `PROMPT_ID = joke-v1` |
| 1a | `explorer/gen_premises.py` | Option A premises: the teacher lists premises over 20 areas; near-duplicates and unsafe ones dropped; stable id = hash of the normalized text; ~10% `eval` by id |
| 1b | `explorer/gen_candidates.py` | One call per (premise, mode) for 4 lines; short cells retried up to 2 times, never padded; resumable per premise; refuses to mix teachers or prompt versions in one file |
| — | `explorer/hf_generate.py`, `explorer/pool_io.py` | Batched sampling with OOM split; resumable JSONL with torn-line repair (same rule as `label_modes.py`) |
| — | `explorer/safety_rules.py` | The live show's rules layer (`backend/app/safety.py check_rules`) under **teens** rules, the strictest, so no training line is one the show would refuse in any room |
| 2 | `explorer/rank_pool.py`, `judge/judges.py LogitJudge`, `judge/rank.py rank_from_probs, ordered_pairs` | Screen (safety, exact dups), score all n(n-1) ordered pairs in one batched call per chunk of premises, fit Bradley–Terry with the existing `judge/` code. Reports per premise: first-shown win share, order consistency, unsafe lines with reasons |
| 3 | `explorer/build_sets.py` | SFT top 10; DPO pairs; O-GRPO with group z-score advantage and softmax weight stored as `w·G`; drops premises whose θ are all equal; ~5% of train premises as dev; eval premises only in `eval_prompts.jsonl`; `stats.json` |
| 4 | `explorer/train_sft.py` | `--stage sft` (HumorGen SFT settings) and `--stage ogrpo` (weighted loss, from the SFT adapter). Custom `WeightedTrainer` |
| — | `*.sbatch` | One per step, H200 on `gpu` like `train_dpo.sbatch`, resumable with `--requeue` |
| — | `explorer/requirements-cluster.txt` | Added `trl==0.24.0`, `peft==0.17.1`, `datasets`, `accelerate`, `httpx` (versions tested locally) |

**Design decision for you (default chosen, easy to flip): the mode in the student prompt.** The live show's
director picks a mode, so the trained model should take one. But a DPO pair shares one prompt, and HumorGen's
"random top-5 vs bottom-5" would often pair lines of different modes under one mode's prompt. Default: mode in
the prompt, and DPO pairs are made **within a mode**, that mode's best line vs its worst, up to 6 pairs per
premise. `build_sets --no-mode` gives HumorGen's exact form (premise-only prompt, global pairs) for a
faithful replication. Judgment call, not measured.

**Smoke runs** (0.5B model, CPU, 3 hand-written premises):
- `gen_candidates`: 54 of 72 lines in 43 s; 12 cells short after retries (the 0.5B often ignores the JSON
  format; the 32B should not, check the "short cells" count in the job log).
- `rank_pool`: 306/272/240 calls for 18/17/16 lines, as n(n-1) says. 3 lines rejected by the safety rules
  (`body_or_appearance:jeans`, `teens_sexual:sleep with`, `named_person:Why`). The last is a false positive:
  the show's rule treats a capital letter mid-sentence as a possible name. It fails closed by design, so it will
  over-reject some lines (brand names too). Left as is; `stats.json` reports the unsafe share.
- **Found: a position-biased judge makes a silent, meaningless ranking.** The smoke judge always says "A", so
  every pair splits 1–1, every θ is 0 and the "ranking" is input order (`consistent_share` 0.0). Guard added:
  `build_sets` drops premises whose θ are all equal and reports the median consistency. If the real judge's
  median consistency is low, the pool ranking is mostly noise: look at it before training.
- `build_sets` on a synthetic ranking (random judge scores): O-GRPO weights per premise sum to exactly G (17.0
  for 17 lines); DPO pairs share one mode's prompt; the eval premise went only to `eval_prompts`.
- `train_sft --stage sft` 3 steps: dev loss 1.98 → 1.88. `--stage ogrpo` and `train_dpo --stage generator`
  from that adapter: both run and save.
- **Checked: "start from SFT" really starts from SFT.** The merged model (`load_policy`) and base + SFT adapter
  unmerged give the same logits (max difference 0.0004, float rounding); both differ from the plain base by up to
  1.0. So DPO's reference and O-GRPO's start are the SFT model.

**Caught before it ran: a 4× learning rate in the weighted trainer.** transformers 4.57 skips dividing the loss
by the gradient-accumulation steps when the model's `forward` takes `**kwargs` (Qwen's does), assuming a
custom `compute_loss` normalizes by `num_items_in_batch` itself. Ours doesn't, so with accumulation 4 the
effective learning rate would have been 4× the setting. Fixed by setting `model_accepts_loss_kwargs = False`
(the Trainer's own docstring says to); a test checks it.

**Caught before it ran: 32B teacher would not fit the job's RAM.** `from_pretrained(...).to("cuda")` loads on
the CPU first: ~65 GB for the 32B, more than `--mem=64G`. Now loads with `device_map` straight to the GPU.

**Tests:** 25 in `tests/test_model_pipeline.py`. A mutation check: breaking the O-GRPO weight (`w` instead of
`w·G`) and the ranking's winner rule each made a test fail, so the tests bite. Full suite: **494 passed,
3 skipped**. The rule that explorer scripts carry `from __future__ import annotations` (Python 3.9) now covers
the new files.

**My own slip, fixed:** restoring a file after the mutation check, `git checkout judge/rank.py` reverted it to
the last commit and wiped the uncommitted `rank_from_probs`. Re-applied the same code and re-ran the full suite.
