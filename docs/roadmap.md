# Roadmap

## M1 — Dec 15, 2026 (industry showcase + RL course final project)

Budget: 8 weeks × 12–15 h ≈ **100–110 h**. Two audiences: course graders (need RL rigor) and
showcase visitors (need a live demo that's fast, safe, funny).

**Principle:** the RL contribution is proven in **simulation** (reproducible, thousands of
episodes); the live demo shows the same policy acting in the real world. Build the sim first
so the grade isn't hostage to audio/latency problems.

| Tier | Scope |
|---|---|
| Must | Simulated audience env; bandit algos + baselines + regret curves; voice loop + waveform UI; core identity × 6 modes generation; safety filter (teens hard rule); laughter detection with operator-button fallback; laugh-aware turn-taking; episode logging |
| Should | Banter/roast handling via ASR; training-mode episode rating; 3–5 pilot sessions |
| Could | LLM-judge rerank of K=3 candidates; pairwise vote UI |
| Cut (post-M1) | VLM snapshot (operator sets context), commute venue, learned RM, DPO, distillation, H1 live experiment |

| Wk | Dates | Deliverable | Exit criterion |
|---|---|---|---|
| 1 | Oct 20–26 | Stack scaffold (FastAPI + Next.js), log schema, **latency spike** (streaming LLM → streaming TTS), sim env spec | Measured end-to-end latency; pick providers |
| 2 | Oct 27–Nov 2 | **Sim env** (gym-style, parametric audience) + ε-greedy, LinUCB, Thompson + fixed/random baselines | Regret curves for all policies across 3 venue contexts |
| 3–4 | Nov 3–16 | Voice loop, waveform UI, operator panel (venue/energy/age + laugh button), mode-conditioned generation, safety filter | 10-min generated set plays end-to-end, logged |
| 5 | Nov 17–23 | VAD + laughter detector + hold-during-laugh; bandit wired to live reward | Detector vs operator button agreement measured; no talking over laughs |
| 6 | Nov 24–30 | ASR banter/roast handling; episode rating UI; final sim experiments + plots (switching penalty ablation) | Report-ready figures |
| 7 | Dec 1–7 | Pilot sessions (friends/classmates), fix top issues, report draft | ≥ 3 logged live sessions |
| 8 | Dec 8–15 | **Feature freeze.** Rehearse demo in noisy room, finalize report | Demo runs 3× in a row without intervention |

Fallbacks: detector unreliable → operator button is the reward. Bandit live-wiring slips →
live demo uses operator-chosen mode; RL results stand on sim.

---

## Long-term plan (post-M1)

Estimates assume **one developer, part-time (~15 h/week)**. Full-time ≈ divide by 2.5.
Stage 5 is calendar-bound (depends on getting audiences), not effort-bound.

| # | Stage | Deliverable | Exit criteria | Est. |
|---|---|---|---|---|
| 0 | Foundations | Repo docs, stack choice, license checks | Stack + log schema draft agreed in decisions.md | 1 wk |
| 1 | Voice loop MVP | Prepared set → TTS → waveform UI; human "laughed" button; episode logging | 10-min set plays end-to-end; every joke logged with timestamps | 3 wks |
| 2 | Hearing the room | VAD + laughter detector + laugh-aware turn-taking | Detector agrees with human button ≥ 80% on recorded sessions; bot never talks over a laugh | 3 wks |
| 3 | Generation | Core identity + 6 modes, K-candidate gen, LLM pairwise judge, safety filter, venue snapshot, ASR banter + roast handling | Judge rerank beats random pick on rJokes-style held-out pairs; safety filter passes red-team list; next line ready ≤ 1s after laugh decay | 4 wks |
| 4 | Director | Contextual bandit (Thompson/LinUCB) + switching penalty; simulated-audience harness | Bandit converges to the right mode for each simulated venue; mode switches stay bounded | 3 wks |
| 5 | Training mode + collection | End-of-episode rating + pairwise vote UI; run sessions across venues | ≥ 30 sessions across ≥ 3 venue classes | 6–10 wks calendar (overlaps 6) |
| 6 | Experiments (H1) | Run the condition table in design.md; analysis | Pre-registered metrics reported with CIs, win or lose | 4 wks |
| 7 | Learned reward + DPO (H2) | RM from logs/votes; compare to LLM judge; DPO small generator | RM vs judge on held-out live reactions; distilled model meets latency budget | 5 wks |

**Total: ~6–8 months part-time** (~29 wks effort; Stage 5 overlaps Stage 6 prep).

## Biggest schedule risks

- Latency (Stage 3) — may force switching TTS/LLM providers or early distillation.
- Laugh detector in noisy venues (bars) (Stage 2) — may need venue-specific thresholds.
- Session count (Stage 5) — H1 needs enough sessions per condition; walk-up audiences are noisy.

## Open questions

- Stack: Python backend + web UI? Which LLM/TTS providers?
- Per-condition session count needed for H1 (do a power estimate after first 10 sessions).
- Consent flow for recording in training mode.
- Venue taxonomy: fixed list vs free-text from VLM mapped to classes.
