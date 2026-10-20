# Design

## Goal

A voice agent (rendered as an animated waveform) that performs stand-up to whoever is in
front of it: sets the stage from the venue, delivers material, handles banter and roasts,
and adapts to the room in real time based on laughter. Research first; a working demo is the
vehicle.

## Architecture

```
[session start]  optional camera snapshot ─► VLM ─► scene text (venue, size, vibe) ─► discard image
                                                          │
mic ─► VAD ─┬─► laughter detector ─► laugh features ──────┼─► room state
            └─► ASR ─► banter / roast classifier ─────────┘        │
                                                                   ▼
                                     DIRECTOR: contextual bandit picks persona mode
                                                                   │
                         GENERATOR: core identity × mode ─► K candidates (thought-leap prompting)
                                                                   │
                                     JUDGE: pairwise rerank (LLM judge → later learned RM)
                                                                   │
                                     safety filter ─► TTS (with timing marks) ─► waveform UI
                                                                   │
                                     laugh-aware turn-taking: hold during laugh, resume on decay
```

### Two learning timescales

| Timescale | Mechanism | Updates |
|---|---|---|
| Within a set (seconds–minutes) | Contextual bandit over persona modes | Bandit posterior, in-context "what landed" memory |
| Across sets (days–weeks) | Reward model from logged laughs + votes → DPO on generator | Model weights, offline |

No gradient updates during a live show.

### Persona model

- **Core identity** (fixed): self-aware, slightly anxious robot that knows it's on display.
- **Modes** (from HumorGen): Neurotic, Cynic, Observer, Wordsmith, Optimist, Absurdist.
- Contextual bandit chooses modes; add a **switching penalty** so the voice stays coherent.
  The bandit picks a mode; it is the fixed core identity that keeps the voice consistent.

### Contextual bandit

- Arms: 6 modes (optionally × bit type: prepared / crowd work / callback).
- Context (operator-set at session start — no CV, so age can't be inferred):
  - `venue`: bar | company_event | showcase | commute (commute deferred: noise, unclear audience)
  - `energy`: low | normal | high ("hustle Monday" = low)
  - `age`: teens | young_adults | mixed
  - live features: laugh baseline, time into set, recent banter/roast flag, last-mode outcome
- `age=teens` triggers a **stricter safety filter** (no alcohol/sexual content). Hard rule,
  not a bandit decision.
- Venue is a **prior**, not ground truth — live signal must be able to override it.
- Algorithm: start with Thompson sampling or LinUCB. Simple and inspectable.
- Debug offline against simulated audiences before live use.

### Simulated audience environment (primary RL evaluation for M1)

Gym-style env, no LLM in the loop, so thousands of episodes run in seconds:
- Each context `(venue, energy, age)` has a hidden laugh probability per mode, plus noise,
  fatigue (repeating a mode decays its payoff) and a novelty bump early in the set.
- Priors seeded from HumorGen per-persona win rates + stated assumptions per venue.
  **The audience preferences are made up**; the report must say so. The sim tests whether
  the *algorithm* adapts, not whether the jokes are funny.
- Metrics: cumulative regret, laughs/episode, mode-switch count, adaptation speed after a
  mid-set preference shift.
- Policies: fixed best mode, random, ε-greedy, LinUCB, Thompson; ± switching penalty.
- Optional: LLM-persona audiences for qualitative checks only.

## Reward design

| Signal | Granularity | Role |
|---|---|---|
| Laugh occurred / duration / onset latency / intensity | Per joke | **Training reward** (normalized to room baseline) |
| Laughs per minute | Per episode | Reported metric |
| End-of-episode rating (1–5) | Per episode | Reported outcome; calibrates laugh signal |
| Pairwise votes ("bit A or B?") | Per episode, training mode | Preference data for RM (Christiano et al.) |

Known biases to control: room-size loudness, novelty ("a robot is talking!"), peak-end bias
in ratings, drunk-audience inflation at bars, delayed laughs misattributed to wrong line.
Expect high noise: HumorGen reports 31.7% inter-annotator agreement, 58.3% judge–human match.

## Modes of operation

- **Show mode**: optimize for performance; log everything.
- **Training mode**: same, plus explicit end-of-episode rating and pairwise votes UI.

## Research hypothesis

H1: Adapting persona mode to live laughter (contextual bandit) yields higher normalized
laughs/min and episode rating than (a) the best fixed mode (Neurotic, per HumorGen) and
(b) random mode per joke.

| Condition | Purpose |
|---|---|
| Plain LLM "be funny" | Floor |
| Fixed Neurotic | Strong static baseline |
| Random mode | Controls for variety |
| **Contextual bandit** | Method |
| Bandit without judge rerank | Ablation: does reranking matter live? |

H2 (later): a reward model trained on logged laughs + votes beats the LLM judge at predicting
live audience response.

## Latency budget

Target: ≤ 1s from end-of-laugh to next line. Achieve by generating/judging the next
candidates *while* the bot is speaking or the room is laughing. Distill to a small model later.

## Safety

- Hard filter on output: appearance, bodies, protected attributes, named individuals' traits.
- Banter responses target the content of what was said.
- Camera consent notice on UI; no persistence of images/audio of identifiable people beyond
  derived features unless explicit consent in training mode.
