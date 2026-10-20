# Decision Log

Append-only. Format: date — decision — why — alternatives rejected.

## 2026-10-20 — Audio-only realtime loop
Why: latency, and removes appearance-based joke risk by construction.
Rejected: continuous facial-expression tracking (noisy, slow, privacy cost).
Keep: one optional async scene snapshot at session start → text only.

## 2026-10-20 — Per-joke laughter is the reward; episode rating is the outcome
Why: episode rating is sparse (1 scalar / ~20 jokes) and peak-end biased; raw laugh counts
are gameable. Normalize laugh features to room baseline.

## 2026-10-20 — Contextual bandit over HumorGen persona modes, fixed core identity
Why: consistent personality with adaptive material; venue is a context feature/prior.
Rejected: switching full personas (incoherent); full RL online (too little signal per show).

## 2026-10-20 — LLM-as-judge first, learned reward model later
Why: no data at start. Training mode collects ratings + pairwise votes to replace the judge (H2).

## 2026-10-20 — Data sources
Primary: own logs. Cold start: rJokes, HumorGen/LoL methods. Deferred: scrapsfromtheloft
(copyright), HumorGen dataset (anonymous repo, no license yet). See data-sources.md.

## 2026-10-20 — Research framing
H1: adaptive bandit beats best fixed mode and random mode live. H2: learned RM beats LLM judge.

## 2026-10-20 — M1 = Dec 15 (industry showcase + RL course final project)
Budget ~100–110 h. RL contribution proven in a simulated audience env (regret curves vs
baselines); live demo shows the same policy acting live. Cut for M1: VLM snapshot, commute,
learned RM, DPO, live H1 experiment. See roadmap.md.

## 2026-10-20 — Context = three operator-set features, not one venue label
venue (bar, company_event, showcase, commute) × energy (low, normal, high) × age (teens,
young_adults, mixed). Why: original list mixed place, mood, and demographic; audio-only can't
infer age. Commute deferred (transit noise, unclear audience).

## 2026-10-20 — Teens = stricter safety filter (hard rule)
Why: minors. Not left to the bandit.
