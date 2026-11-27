# Manual work on rJokes `test.jsonl`: picking the DPO judge (Exp 17)

Where this fits: the DPO pairs for the joke model come from a **judge** that compares two candidate jokes
(decisions.md 2026-11-01, "Trained joke model and DPO move into M1"). Before any pair is generated, we have to
know which judge prompt tracks human taste best. rJokes `test.jsonl` is the human-taste yardstick for that
comparison. This page lists the hand work that has to happen first, why, and what comes after.

No joke text is on this page. Reddit terms allow local use only (decisions.md 2026-10-31); every file with
text lives in `data/processed/rjokes/judge_check/`, which is gitignored.

## Why `test.jsonl`

| Reason | Detail |
|---|---|
| It's the only allowed source | rJokes may test judges but is never joke material (decisions.md 2026-10-31, 2026-11-01). Transcripts can't be used to write lines at all (2026-11-08). |
| It's clean | The 90k labeled rows in `modes.jsonl` all come from `train`. Test hasn't been used by any experiment, so a judge can't have been tuned to it. |
| It has a human signal | Every joke has an upvote percentile within its year (`pct`), the only human preference we have before live shows. |
| It's big enough | 29,660 test jokes are unflagged and have a year. Pairing them gives 5,318 matched pairs. We need 1,600. |
| `dev` keeps the few-shot examples separate | Few-shot examples come from `dev.jsonl`, so the test score isn't inflated by showing the judge test jokes. |

## How the pairs were built (already done)

Rule, fixed before any judge sees them:

1. Keep test jokes with a known year and no keyword flag (`flags == []`). Flagged jokes are 25% of the data and
   the judge's safety labels can't be trusted (Exp 9), so unflagged only. Judgment call, not measured.
2. Cell = year × length quartile (cuts at 13 / 17 / 25 words), the same control as Exp 10. Length and year both
   move upvotes, so a pair only compares jokes of similar length from the same year.
3. Within a cell, pair a **high** joke (`pct ≥ 0.8`) with a **low** joke (`pct ≤ 0.3`). Each joke is used once.
   Mean pct gap: 0.72.
4. Shuffle with seed 17. The first 100 pairs are the hand-check sample, the next 1,500 are the judge set.
   No joke appears in both.

| File (in `data/processed/rjokes/judge_check/`) | Rows | Use |
|---|---|---|
| `test_hand_check.csv` | 100 | **You fill this in.** Blind: A/B order is random, nothing says which joke was upvoted |
| `test_hand_check_KEY.csv` | 100 | Answer key: ids, which side was upvoted, pct, cell. **Don't open it until the sheet is done** |
| `test_judge_pairs.jsonl` | 1,500 | Ids only. The judge run joins the text at run time |
| `dev_fewshot_candidates.csv` | 30 | **You fill this in.** Candidates for the judge's few-shot examples (`pct ≥ 0.9` vs `≤ 0.2`) |

## Manual task 1: blind hand check of 100 test pairs (about 40 min)

Open `test_hand_check.csv`. For each pair, fill in:

| Column | Values | Meaning |
|---|---|---|
| `valid` | `y` / `n` | Is this a fair comparison of two jokes? |
| `problem` | `dup`, `not_joke`, `needs_context`, `unsafe`, `garbled`, `other` | Only when `valid = n`. `dup`: same or near-same joke. `needs_context`: depends on a Reddit thread, a username or an edit note. `unsafe`: bodies, protected groups or sexual content the keyword flags missed |
| `my_pick` | `A` / `B` / `tie` | Which one you find funnier. Decide before reading the next pair; don't go back |
| `notes` | free text | Anything odd |

**Why this is needed:**

- **It sets the ceiling.** Upvotes are noisy, so even a perfect judge can't agree with them 100% of the time.
  How often *you* agree with upvotes is the best estimate of how high a judge can reasonably score. A judge at
  0.58 means very different things if you scored 0.60 or 0.85. (CLAUDE.md "measure before cut" rule 2: size
  the prize first.)
- **It shows how much of the pair set is usable.** If many pairs are invalid, they drag every judge toward
  0.5. The invalid rate tells us whether to filter the 1,500 judge pairs (and how) before the run.
- **It catches safety misses.** These rows will also go to the judge, and `unsafe` counts tell us whether
  keyword flags are enough to filter them.

With 100 pairs, your agreement rate has a 95% CI of about ±0.10. That's enough to place the ceiling, not to
rank judges. Ranking is the 1,500-pair set's job.

## Manual task 2: choose few-shot examples from dev (about 15 min)

One of the three judges is shown example pairs, and decisions.md 2026-11-01 says **examples must pass a hand
check first**. Open `dev_fewshot_candidates.csv`. Here you can see which joke was upvoted; the question is
whether the pair *teaches* something.

- `keep = y` only if: both are real jokes, the upvoted one is clearly funnier to you, the pair is safe to show,
  and it isn't a near-duplicate of another kept pair.
- `why`: one line on what the pair shows (sharper wording, better twist, less obvious punchline...).
- Keep **6 to 8**. Spread them over length and style; don't pick only puns.

## Manual task 3: fix the pass criteria in decisions.md (before the run)

The project rule is that criteria are written down before a judge is run. Proposed, for you to edit:

- **Judges compared (J0–J2):** J0 plain prompt; J1 "human taste" prompt; J2 J1 + the kept dev examples.
  Each judge sees every pair in both orders (3,000 calls per judge).
- **Scoring:** per pair, a judge that agrees in both orders scores 1 or 0; an order-inconsistent answer scores
  0.5. Accuracy is averaged over pairs marked valid, or over all pairs if task 1 finds < 5% invalid.
- **C1, the judge tracks upvotes:** the best judge's 95% CI on accuracy lies above 0.5. If not, no judge tracks
  human taste, and DPO from judge pairs would learn arbitrary taste: stop and discuss before generating pairs.
- **C2, choosing between judges:** the best judge wins only if its paired advantage over the next one is larger
  than the paired CI's half-width. With 1,500 pairs this half-width is about ±0.03, depending on how often two
  judges disagree. If it's smaller, the judges tie and we keep the **simplest** (J0 before J1 before J2). If the
  tie matters, add pairs: about 3,700 more matched test pairs are unused.
- **C3, position bias:** report the share of "first shown" wins. It doesn't block a judge, because both orders
  are asked, but a judge outside 0.40–0.60 wastes calls on inconsistent answers.
- **Reported alongside:** your hand-check agreement (the ceiling), and accuracy by length bin and year.

**The number that would change the decision:** the best judge's accuracy against 0.5 (C1) and against the next
judge (C2). If every outcome led to the same judge, this run wouldn't be needed. It's needed because J1/J2
cost extra tokens per call on 72k DPO calls, so they only stay in if they win.

## After the manual work

1. Write the criteria above into `docs/decisions.md` ("Judge choice for DPO pairs: pass criteria, fixed before
   running").
2. Move the pair builder into `explorer/` and add Exp 17 to a new `notebooks/decisions-model.ipynb` as a
   `run(**params)` through `sim.cache.run_or_load`, with ✅/❌ claim checks and a `COVERAGE` entry in
   `notebooks/decision-plot.ipynb`.
3. Run J0–J2 on `test_judge_pairs.jsonl` (cluster or Gemini), score against the key, and log the outcome in
   `docs/decisions.md` and `docs/status.md`.
4. The chosen judge then labels the generated candidates: own premises → 6-mode candidates → judge
   all pairs both orders → Bradley–Terry → (top, bottom) pairs for DPO.

## Limits

- Upvotes are Reddit taste, from 2015 to 2019. A live room may differ: Exp 10 found the judge and upvotes agree
  only in part. The report must say the DPO model learns the judge's taste.
- Dropping flagged jokes removes some of the edgier humor, so the judge is tested on milder material than
  some of its candidates will be.
- `pct ≤ 0.3` includes many score-0 jokes, whose percentile only reflects ties. Some "low" jokes may just
  never have been seen. Task 1's ceiling absorbs this noise.
