# judge: pairwise ranking of candidate jokes

The JUDGE step of the live loop ([design.md](../docs/design.md)): the generator writes K
candidate jokes for one prompt, and the judge ranks them from pairwise "which is funnier?"
calls. Each call goes to a `Judge` backend (an LLM later, a simulated judge now); a
Bradley–Terry model turns the answers into one strength per joke, and an adaptive (LUCB)
rule picks which pair to ask about next, so calls go to the boundaries that matter (top-k,
bottom-k) instead of to every pair. Terms: [glossary](../docs/wiki/glossary.md#judge-ranking-candidate-jokes).

Standard library + numpy only. No network calls: `LLMJudge` is a stub.

```python
import numpy as np
from judge import SimulatedJudge, rank, write_judgments

jokes = [f"candidate joke {i}" for i in range(24)]
theta_true = np.random.default_rng(0).normal(size=24)           # hidden funniness
judge = SimulatedJudge(theta_true, jokes, seed=0, first_bias=0.5)  # likes whatever it reads first

r = rank("airport security", jokes, judge, method="adaptive", budget=80,
         top_cutoffs=(5, 10), bottom_cutoffs=(5,), seed=0)
print(r.n_calls, r.stop_reason)       # e.g. 80 budget
print(r.top(5), r.bottom(5))          # joke indices: funniest first / least funny first
print(r.theta.round(2), r.se.round(2))
write_judgments("judgments.jsonl", r.comparisons)   # one JSON object per call
```

| File | What it holds |
|---|---|
| `types.py` | `Comparison` (one judge call), `Ranking` (fit + every comparison) |
| `backends.py` | `Judge` protocol, `SimulatedJudge`, `LLMJudge` stub |
| `bt.py` | `fit_bt`, `bt_se`, `borda_scores` |
| `select.py` | `warm_start_pairs`, `all_pairs`, `adaptive_next` |
| `rank.py` | `rank()`, the one place the judge is called; `write_judgments` |

Tests: `tests/test_judge.py`.

## Model

Each joke $i$ has a latent funniness $\theta_i$, with $\pi_i = e^{\theta_i}$, and

$$
P(i \text{ beats } j) = \sigma(\theta_i - \theta_j) = \frac{\pi_i}{\pi_i + \pi_j}.
$$

## Fit (MM algorithm, Hunter 2004) with a prior

Every joke gets $\alpha$ virtual wins and $\alpha$ virtual losses against a dummy item with
$\pi_0 = 1$. With $W_i$ the real wins of $i$ and $n_{ij}$ the number of comparisons between
$i$ and $j$, iterate until $\max_i |\log \pi_i^{\text{new}} - \log \pi_i^{\text{old}}| < \text{tol}$:

$$
\pi_i \leftarrow \frac{W_i + \alpha}{\displaystyle\sum_{j \ne i} \frac{n_{ij}}{\pi_i + \pi_j} + \frac{2\alpha}{\pi_i + 1}}
$$

Return $\theta = \log \pi$, centered to mean 0. The prior keeps every $\theta$ finite when a joke
has zero wins, and fixes the scale (no renormalization step is needed).

## Standard errors

Diagonal Fisher approximation, with $p_{ij} = \sigma(\theta_i - \theta_j)$ and
$p_{i0} = \sigma(\theta_i)$, using the *uncentered* $\theta$ from the fit (the dummy item pins
the location):

$$
I_i = \sum_{j \ne i} n_{ij}\, p_{ij} (1 - p_{ij}) + 2\alpha\, p_{i0} (1 - p_{i0}),
\qquad
\mathrm{se}_i = \frac{1}{\sqrt{I_i}}
$$

## Empirical Borda score

$$
b_i = \frac{1}{|O_i|} \sum_{j \in O_i} \frac{\text{wins of } i \text{ over } j}{n_{ij}}
$$

where $O_i$ is the set of opponents $i$ has played. Jokes with no games get NaN. Reported next
to $\theta$ as a model-free check; it isn't used for selection.

## Adaptive selection (top-k identification, LUCB style)

$$
\mathrm{LCB}_i = \theta_i - \beta\, \mathrm{se}_i, \qquad \mathrm{UCB}_i = \theta_i + \beta\, \mathrm{se}_i
$$

Each cut-off is a position $m$: "the current top $m$ by $\theta$". A top cut-off $k$ gives
$m = k$; a bottom cut-off $k$ gives $m = n - k$. For each $m$, with $T$ the current top $m$:

$$
h = \arg\min_{i \in T} \mathrm{LCB}_i, \qquad l = \arg\max_{i \notin T} \mathrm{UCB}_i
$$

The cut-off is **separated** if $\mathrm{LCB}_h > \mathrm{UCB}_l$; its overlap is
$\mathrm{UCB}_l - \mathrm{LCB}_h$. If every cut-off is separated, stop. Otherwise compare the
pair $(h, l)$ of the cut-off with the largest overlap, refit, and repeat.

**Warm start:** `warm_rounds` random perfect matchings of the $n$ jokes (if $n$ is odd, one
random joke sits out each round), so every joke has a few games before LUCB starts.

## Position randomization

LLM judges tend to prefer the joke they read first (or second). Selection only returns an
unordered pair $(a, b)$; for every call, `rank()` flips a fair coin from its seeded RNG to
decide which is shown first, calls `judge.compare`, maps the answer back to `a`/`b`, and
records `shown_first` in the `Comparison`. This is the only place order is randomized, so a
bias averages out of $\theta$ and can be measured from the log (win rate of `shown_first`).

## Stopping

| `stop_reason` | When |
|---|---|
| `separated` | adaptive: every cut-off is separated |
| `budget` | `n_calls` reached `budget` first |
| `exhausted` | all_pairs: every pair compared once |

`method="all_pairs"` compares every unordered pair once in random order (`budget=None` = no
limit). `method="adaptive"` requires a budget, since exactly tied jokes never separate.
