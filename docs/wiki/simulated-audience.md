# Simulated audience

**Where it fits:** the primary RL evaluation for M1. The live demo shows the same policy in the
real world, but the graded result comes from thousands of reproducible simulated shows.
**Used in:** `sim/env.py` (PR #1), every policy comparison and regret curve, the
[metrics log](../metrics.md).

## Why simulate

Live audiences are slow (one show = 10 min), noisy, and not reproducible. A simulator answers
"does the *algorithm* adapt?" in seconds. It cannot answer "are the jokes funny?". The
report must say so: the venue preferences are made up.

## The model

```text
latent  = base[mode] + venue[mode] + age[mode] + energy      ← fixed for the show
        + novelty · e^(−t/τ)                                  ← "a robot is talking!"
        − fatigue · (uses of this mode in the last 5 jokes)   ← room tires of repetition
        + shift[mode] (after shift_at)                        ← room changes its mind mid-set
level   = thresholds cleared by latent + N(0, noise)          ← 0, 1, 2 or 3
reward  = level / 3
```

- **Ordinal laugh level** (0–3) matches what live laugh detection and transcript labels give.
  Scaling to [0, 1] keeps samplers simple.
- **Exact oracle:** E[level] = Σ Φ((latent − θₖ)/σ), so regret carries no sampling noise.
- **Gym-style but numpy-only:** `reset(context, seed)` / `step(mode)`, no gymnasium dependency.

## What the first metrics showed (PR #1)

From [metrics.md](../metrics.md):
1. There is something to learn: the oracle beats random by 5–10 laughs per 40-joke set.
2. **Random beat every fixed mode**: fatigue (−0.75 at steady state) outweighs the venue
   offsets (≤ +0.4). The sim was rewarding variety more than fit to the room.
3. Telling the top two modes apart takes 551–7,063 jokes, so within one set a learner can't
   find the venue's best mode. That's an argument for priors and for learning across shows.

Open design decision: fatigue strength, and whether the policy sees recent modes or decides
per bit.

## Further reading

- [Sutton & Barto, ch. 2.5 (non-stationary problems)](http://incompleteideas.net/book/the-book-2nd.html):
  why fatigue and mid-set shifts break "average all past rewards".
- [Li et al. 2010, section 4 (offline evaluation)](https://arxiv.org/abs/1003.0146): how to
  evaluate bandits on logged data, which is relevant once we have real session logs.
