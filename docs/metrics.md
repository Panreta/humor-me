# Metrics log

One section per merged PR: the numbers that justified merging it. The report's Results section
is assembled from here, so every number names the command that produced it.

## PR #1: simulated audience env (2026-10-27)

`python -m sim.evaluate_env` (1,000 episodes per cell, 40 jokes per set, context
`<venue>/normal/mixed`). Raw output: `reports/metrics/sim_env.json`.

Laughs per set = sum of 0–3 laugh levels over 40 jokes.

| Venue | Oracle | Best fixed (mode) | Fixed neurotic | Random | Best vs runner-up gap | Jokes to separate |
|---|---|---|---|---|---|---|
| bar | 45.4 | 32.4 (cynic) | 25.9 | 35.8 | cynic − absurdist = 0.052 | 551 |
| company_event | 41.9 | 28.9 (observer) | 25.9 | 35.5 | observer − neurotic = 0.031 | 1,476 |
| showcase | 43.6 | 30.0 (absurdist) | 28.6 | 38.6 | absurdist − neurotic = 0.014 | 7,063 |

Best mode over all 27 contexts: absurdist, cynic, neurotic or observer (4 distinct).

**Findings**
1. There is something to learn: the oracle beats random by 5–10 laughs per set, and the best
   mode depends on context.
2. **Random beats every fixed mode.** Fatigue (−0.15 per use in a 5-joke window, so −0.75
   for a fixed mode) outweighs the largest venue offset (+0.4). In this setting, variety
   matters more than picking the right mode.
3. Telling the top two modes apart takes 551–7,063 jokes against a 40-joke set. Learning
   within one set can't find the venue's best mode; only learning across sets can.

**Open (decides PR 2's setup):** retune fatigue vs venue offsets, or keep them and require
policies that see recent history. See PR #1 discussion.
