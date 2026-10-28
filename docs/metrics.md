# Metrics log

One section per merged PR: the numbers that justified merging it. The report's Results section
is assembled from here, so every number names the command that produced it.

## PR #1: simulated audience env (2026-10-27)

### v1: one decision per joke (superseded)

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

### v2: one decision per bit (merged version)

Decision (decisions.md 2026-10-27): the policy picks the mode once per bit (8 bits × 5 jokes),
like a comedian committing to one premise and attitude; fatigue counts uses in the last 2 bits
(fixed mode at steady state: −0.3, below the largest venue offset of +0.4).

| Venue | Oracle | Best fixed (mode) | Fixed neurotic | Random | Best vs runner-up gap | Bits (shows) to separate |
|---|---|---|---|---|---|---|
| bar | 50.4 | 46.7 (cynic) | 39.0 | 38.1 | cynic − absurdist = 0.052 | 111 (13.9) |
| company_event | 46.9 | 42.6 (observer) | 39.0 | 37.7 | observer − neurotic = 0.031 | 296 (37.0) |
| showcase | 49.1 | 43.9 (absurdist) | 42.3 | 40.8 | absurdist − neurotic = 0.014 | 1,414 (176.8) |

**Findings (v2 vs v1)**
1. **Fit to the room now matters:** the best fixed mode beats random by 3–9 laughs per set
   (v1: random won everywhere). The venue's mode is worth learning.
2. **Variety still matters, but less:** the oracle beats the best fixed mode by 4–5 laughs, so
   a policy that both fits the room *and* rotates should beat any single persona.
3. **"Always Neurotic"** (HumorGen's best judge-rated persona) trails the venue's best mode by
   1.6–7.7 laughs: a fixed persona chosen by an LLM judge is not the best choice for every room.
4. **Learning must span shows:** separating the top two modes takes ~14 shows (bar) to ~177
   (showcase). Per-bit rewards average 5 jokes, which cut the samples needed 5× vs v1, but one
   show is still far too short. Priors and pooling across contexts carry the live case.
