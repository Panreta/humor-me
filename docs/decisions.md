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

## 2026-10-20 — M1 = Dec 15 (RL course final project, school demo)
Budget ~100–110 h. RL contribution proven in a simulated audience env (regret curves vs
baselines); live demo shows the same policy acting live. Cut for M1: VLM snapshot, commute,
learned RM, DPO, live H1 experiment. See roadmap.md.

## 2026-10-20 — Context = three operator-set features, not one venue label
venue (bar, company_event, showcase, commute) × energy (low, normal, high) × age (teens,
young_adults, mixed). Why: original list mixed place, mood, and demographic; audio-only can't
infer age. Commute deferred (transit noise, unclear audience).

## 2026-10-20 — Teens = stricter safety filter (hard rule)
Why: minors. Not left to the bandit.

## 2026-10-20 — Model output = next bit, conditioned on room state
State: context (venue/energy/age), bits so far (mode, premise, laugh outcome), open
callbacks, laugh trend, elapsed time. RL planner picks the next bit spec; generator writes it.
Why: fits a sequential RL framing (MDP over bits), and keeps the set coherent.

## 2026-10-20 — Transcripts are for structure + reward, never as imitation text
Uses: (1) laugh predictor from normalized audience markers (reward model / sim reward),
(2) set structure (bit order, segues, callbacks) as planner demonstrations, (3) bit skeletons
→ synthetic sets in the robot's voice → SFT. Why: SFT on raw transcripts blends 19 voices
into a generic one and memorizes lines. Corpus stays local in data/ (gitignored).
Crowd-work is its own bit type.

## 2026-10-20 — Laugh labels: lexicon-mapped levels, binary primary
Levels 0–4 (none/chuckle/laugh/big laugh/applause break) from a versioned markers.yaml;
unknown marker strings stop the pipeline. Unattributed chuckles/sighs = performer, not
audience. Binary (≥2) is the primary target; ordinal only within-set. Groan/gasp are a
separate channel; groan is mode-dependent reward (positive for Wordsmith).
Why: transcribers differ in vocabulary and density; most never write audience chuckles.

## 2026-10-20 — Opener/closer are positions, not bit types
Bit types: prepared, crowd_work, callback, segue. Position assigned by index in set.

## 2026-10-20 — Density gate for laugh labels; Birbiglia replaces Bamford in held-out
Only 11/20 sets are densely tagged (≤150 words per audience laugh). Sparse/unmarked sets
are structure-only, since absent markers would be read as "no laugh". Bamford is sparse, so
Birbiglia (also Neurotic) takes her held-out slot. 8 training sets remain for the laugh
predictor: thin; more *densely tagged* transcripts are the highest-value data to add.

## 2026-10-21 — Applause alone is not a laugh
Applause/cheer markers without a laugh get level 0 + `applause` flag; level 4 requires a laugh
(same marker or merged with adjacent applause). Why: eyeball check found mid-set cheers for
hometown mentions labeled as applause breaks. Level-4 counts dropped (e.g. Notaro 23 → 0).

## 2026-10-21 — Two datasets, split logically at the output
`laugh_labeled` vs `structure_only`, decided by the density gate (overridable per set via
`dataset:` in body_overrides.yaml) and written to `data/processed/datasets.json`. Stages 3+
write each to its own folder. Why: lets each grow independently and makes it impossible for
the laugh predictor to train on sparse sets. Rejected: separate raw/ folders, since density
is only known after markers run.

## 2026-10-21 — Callbacks are first-class: hooks in set memory, callback(k) planner action
Each bit stores a paraphrased `hook` and how hard it landed; the planner can call back a landed
hook ≥ 2 bits later. Sim callback payoff parameters are fit from transcript callback stats
(S4–S5). Why: callbacks are the delayed-reward decision that justifies sequential RL over a
bandit, and the cross-bit context the user wants the agent to use.

## 2026-10-21 — Laugh predictor trains on lines, not beats
`lines.jsonl`: last line before a reaction gets its level, setup lines 0, each with 150 words of
preceding context (bit-scoped after S4). Beats stay for structure. Why: beat length alone
predicted the beat label (77% positive; 23 vs 74 median words).

## 2026-10-21 — Gemini (free tier) for S4–S5, with minimal excerpts
User's call: small, non-commercial corpus; annotation quality outweighs the exposure. Mitigations:
TextTiling runs locally and only windows around uncertain boundaries are sent; responses must
paraphrase (premises, hooks), never quote; small batches with backoff for free-tier rate limits.
Considered: paid Gemini / Claude API (no training on API data, ~$3–6 batched), local 8B via
Ollama on the RTX 4060 (private, weaker annotation). Revisit local 8B as an offline fallback
for the live demo.

## 2026-10-21 — S4: Gemini directly on whole sets (supersedes "minimal excerpts" above)
One Gemini call per set returns bit starts (as line_idx) plus annotations (premise paraphrase,
mode, type, callback_to, hook) in one pass, halving calls against free-tier limits. Output
references lines by index only, never quotes. TextTiling deferred: add later only as a report
baseline. Hand labels for Bargatze + Birbiglia via `pipeline.labels` (indices committed, text local).
