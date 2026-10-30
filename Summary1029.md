# humor-me: project analysis (2026-10-29)

A snapshot of where the project stands, what's been shown, and where the risks are. Sources:
`docs/status.md`, `docs/decisions.md`, `docs/roadmap.md`, `docs/design.md`, `docs/explore-next.md`,
`sim/`, `reports/metrics/`, and git history (83 commits, PRs #1–#13).

## 1. What the project is

A voice-only stand-up comedy agent, shown as an animated waveform. It performs a set, listens
for laughter, and uses reinforcement learning to choose which persona mode (Neurotic, Cynic,
Observer, Wordsmith, Optimist, Absurdist, taken from HumorGen) to use for the next bit. It has
one fixed identity, a self-aware and slightly anxious robot, and the modes are flavors of that
identity, not separate characters.

- **Milestone M1:** Dec 15, 2026, the RL course final project and class demo.
- **Core research claim:** a contextual bandit, and later a stateful director, can adapt its
  material to a room from per-joke laugh feedback. This is proven in a **simulated audience**
  so the grade doesn't depend on live audio working.
- **Hard rules** (CLAUDE.md): no jokes about bodies or protected attributes, audio only (no
  faces), one core identity, take roasts gracefully, laugh-per-joke is the training reward (not
  raw laugh count), and log everything needed to replay a show.

## 2. Architecture (planned)

```
mic → VAD → laughter detector ─┐
         └→ ASR → banter/roast ─┴→ room state → DIRECTOR (bandit picks mode per bit)
                                                   → GENERATOR (identity × mode, K candidates)
                                                   → JUDGE (pairwise rerank) → safety filter
                                                   → TTS → waveform UI (hold during laughs)
```

It learns on two timescales: within a show, the bandit updates (no gradient steps live); across
shows, a reward model trained on logged sessions plus DPO (post-M1).

## 3. Component status

| Component | State | Notes |
|---|---|---|
| Transcript pipeline S0–S6 | Built, **paused** (2026-10-26) | Bits F1 0.80 vs 0.21 baseline on Bargatze; threads 0.63; callbacks 0/3. Kept for the report appendix and aggregate structure stats |
| rJokes prep (R1) | Done | 432k → 403k jokes after dedupe and length filter; era-normalized percentiles; train/dev/test 324k/40k/40k; 75% have no safety flag |
| Explorer (HPC) mode/safety labeling | Scripts ready, **not run** | Blocks replacing the sim's `base` payoffs with human-derived values |
| Simulated audience (`sim/env.py`) | Built, tested | 8 bits × 5 jokes; latent = base + venue + age + energy + novelty − fatigue + shift |
| Policies (`sim/policies.py`) | Built | Fixed, Random, ε-greedy, LinUCB, LinTS; optional reaction and show-state features |
| Decision notebook | Built | `notebooks/decisions.ipynb`, one re-runnable section per experiment with claim checks |
| Backend | Scaffold | FastAPI `/health` and an echo websocket |
| Frontend | Scaffold | Default Next.js page |
| Set planner / bit writer | Not started | Sprint step 6 |
| Voice, TTS, laughter detection, safety filter | Not started | Weeks 3–5 |

## 4. What the experiments have shown

All results come from the simulator, whose audience preferences are **made up**, apart from
`base`, which comes from HumorGen judge win rates. The results test whether the *algorithm*
adapts, not whether any joke is funny.

| # | Question | Result | Decision |
|---|---|---|---|
| 1 | Choose a mode per joke or per bit? | Per joke, the sim rewarded rotation (an artifact). Per bit, the best fixed mode beats random by 3–9 laughs and the oracle beats best-fixed by 4–5 | **Per bit** |
| 2 | Which policy and prior? | LinUCB + uniform prior: 728 ± 64 laughs lost over 300 shows vs 868–926 for the other linear models and 2,457+ for ε-greedy; 97% of the oracle once experienced. The prior made no significant difference | **LinUCB, uniform prior** |
| 3 | Does a reaction feature help after a mid-set taste shift? | +0.2–0.4 laughs, within the CIs. All learners drop from ~95% to 80–83% of the oracle and recover no faster than random | Not adopted |
| 4 | Do longer sets let it recover? | No. At 8, 16, 32 and 52 bits, after-shift performance stays at 80–85% of the oracle | Keep 8-bit sets |
| 5 | Show state A (tonight's per-mode record as features) | Missed both pre-registered criteria in every cell: it hedges rather than reading the turn, and steady shows pay for it | Not adopted, so try step B |

**What this adds up to:** pooling across rooms is what makes adapting pay. The director handles
*between-room* differences well but has no notion of "this room, tonight". That is the opening
for the project's real RL contribution: a stateful or MDP director, built together with
callbacks (the delayed-reward decision).

## 5. Strengths

- **Rigor and traceability.** Every decision has a decision-log entry, a notebook section and
  claim checks. Pass criteria are fixed *before* runs (Exp 7), and negative results are
  recorded honestly. This is strong material for graders.
- **Sound de-risking.** Proving the RL result in simulation first means the grade doesn't
  depend on latency or the laughter detector. Fallbacks are defined: an operator laugh button,
  and operator-chosen modes.
- **Good modeling calls.** Per-bit decisions remove the need for a switching penalty, reduce
  reward noise, and match how comedians actually work. Recovery is now measured with
  choice-quality curves instead of the luck-prone "bits to recover" metric.
- **Legal and ethical care.** Transcripts stay local, there's a memorization guard, safety is a
  hard rule (stricter for teens) rather than something the bandit learns, and no faces are
  used or stored.
- **Documentation.** A wiki, a glossary, a dev log of problems hit and an explore-next backlog
  make it easy to onboard a new contributor or a new session.

## 6. Risks and gaps

1. **Schedule: the live 60% hasn't started.** The Oct 26 – Nov 2 sprint targets about 40% of
   M1 "Must", and the sim and bandit parts are on track. But weeks 3–5 (voice loop, TTS,
   operator panel, generation, safety filter, VAD and laughter detector) all start from an
   echo-websocket scaffold, with about 6.5 weeks left. The live demo is the biggest risk.
2. **Show-state work could expand without limit.** Steps A → B → C plus the 5b variant are
   interesting but open-ended. Without a timebox, they'll eat into the live-loop weeks.
3. **Callbacks aren't modeled in the sim.** They're what justifies sequential RL over a bandit
   (decisions.md 2026-10-21). Without them, the "beyond bandits" part of the report is thin.
4. **The sim's priors are still placeholders.** The Explorer labeling run hasn't happened, so
   `base` still comes from an LLM judge, not from people.
5. **No generation yet.** Nothing yet produces a long-form set: no set planner, no bit writer,
   no mode-conditioned prompting, and no safety filter on real outputs.
6. **LLM quota.** The Gemini free tier is a bottleneck (one Bargatze call took 1,145 s), and
   the paid-tier or Claude API question is still open. The live loop will need a dependable
   provider.
7. **Docs have drifted.** `docs/status.md` (dated 2026-10-27) still says "policies next" and
   "45 tests passing", but PRs #4–#13 have since merged policies and the shift, set-length and
   show-state experiments. `explore-next.md` lists several "Done" rows as "(this PR)" instead
   of PR numbers.
8. **Setup is Windows-only.** The README and status commands use `.venv/Scripts/python`. On
   this macOS checkout there is no `.venv`, so the test suite wasn't run for this analysis.

## 7. Recommendations (in priority order)

1. **Timebox show state:** run step B (career estimate plus a fast per-show correction) with
   pre-set criteria by about Nov 5. If it passes, fold it into the MDP with callbacks. If it
   fails, write it up as a documented limitation and move on.
2. **Start the live loop now, in parallel:** streaming LLM → streaming TTS → waveform, plus the
   operator laugh button and episode logging. This is the critical path for the demo, and the
   button keeps it working even without a detector.
3. **Prototype the set planner and bit writer** (text only, one 10-minute set per venue) so
   that generation and the safety filter can be tested before voice is added.
4. **Run the Explorer labeling job** so the sim's `base` and explore-next #1 (human mode
   preference via Bradley–Terry) can rest on human data instead of an LLM judge's.
5. **Settle the LLM provider** for the live loop, choosing between a paid tier and the Claude
   API, and keep a local 8B model as an offline fallback.
6. **Housekeeping:** update `status.md`, replace "(this PR)" with PR numbers in
   `explore-next.md`, and add macOS/Linux setup lines (`.venv/bin/python`) to the README.

## 8. Bottom line

The research half is in good shape. The simulator, policy comparison and decision trail are
rigorous, and they already support a clear finding: a contextual bandit reads *rooms* but not
*a room turning*, which motivates a stateful director. The product half (voice, generation,
laughter) is mostly unbuilt, with about six weeks to M1. The main job now is to keep the
research thread timeboxed and put most of the effort into a demo that can run end to end,
using the operator-button fallback.
