# Decisions

What we decided and why, in plain words. One row per decision; the ID links to the full record
(options, numbers, evidence) in [decisions-detail.md](decisions-detail.md).

Adding a decision: write the full entry in decisions-detail.md, then add its row here with the same ID.
A check in `tests/test_docs.py` fails if the IDs in the two files don't match. Rewording a row here is fine;
changing what was decided needs a new entry.

| ID | Date | Decision |
| --- | --- | --- |
| [D-01](decisions-detail.md#d-01) | 2026-10-20 | Audio-only realtime loop |
| [D-02](decisions-detail.md#d-02) | 2026-10-20 | Per-joke laughter is the reward; episode rating is the outcome |
| [D-03](decisions-detail.md#d-03) | 2026-10-20 | Contextual bandit over HumorGen persona modes, fixed core identity |
| [D-04](decisions-detail.md#d-04) | 2026-10-20 | LLM-as-judge first, learned reward model later |
| [D-05](decisions-detail.md#d-05) | 2026-10-20 | Data sources |
| [D-06](decisions-detail.md#d-06) | 2026-10-20 | Research framing |
| [D-07](decisions-detail.md#d-07) | 2026-10-20 | M1 = Dec 15 (RL course final project, school demo) |
| [D-08](decisions-detail.md#d-08) | 2026-10-20 | Context = three operator-set features, not one venue label |
| [D-09](decisions-detail.md#d-09) | 2026-10-20 | Teens = stricter safety filter (hard rule) |
| [D-10](decisions-detail.md#d-10) | 2026-10-20 | Model output = next bit, conditioned on room state |
| [D-11](decisions-detail.md#d-11) | 2026-10-20 | Transcripts are for structure + reward, never as imitation text |
| [D-12](decisions-detail.md#d-12) | 2026-10-20 | Laugh labels: lexicon-mapped levels, binary primary |
| [D-13](decisions-detail.md#d-13) | 2026-10-20 | Opener/closer are positions, not bit types |
| [D-14](decisions-detail.md#d-14) | 2026-10-20 | Density gate for laugh labels; Birbiglia replaces Bamford in held-out |
| [D-15](decisions-detail.md#d-15) | 2026-10-21 | Applause alone is not a laugh |
| [D-16](decisions-detail.md#d-16) | 2026-10-21 | Two datasets, split logically at the output |
| [D-17](decisions-detail.md#d-17) | 2026-10-21 | Callbacks are first-class: hooks in set memory, callback(k) planner action |
| [D-18](decisions-detail.md#d-18) | 2026-10-21 | Laugh predictor trains on lines, not beats |
| [D-19](decisions-detail.md#d-19) | 2026-10-21 | Gemini (free tier) for S4–S5, with minimal excerpts |
| [D-20](decisions-detail.md#d-20) | 2026-10-21 | S4: Gemini directly on whole sets (supersedes "minimal excerpts" above) |
| [D-21](decisions-detail.md#d-21) | 2026-10-21 | Bit = smallest standalone unit; same-topic bits form a thread |
| [D-22](decisions-detail.md#d-22) | 2026-10-26 | Pause the transcript pipeline; rJokes + simulator first |
| [D-23](decisions-detail.md#d-23) | 2026-10-26 | Long-form sets are planned hierarchically, not generated in one pass |
| [D-24](decisions-detail.md#d-24) | 2026-10-26 | Northeastern Explorer for GPU work; laptop for everything else |
| [D-25](decisions-detail.md#d-25) | 2026-10-27 | The director picks a mode per bit, not per joke |
| [D-26](decisions-detail.md#d-26) | 2026-10-27 | Director policy: LinUCB with a uniform prior |
| [D-27](decisions-detail.md#d-27) | 2026-10-28 | Keep the director's features; within-show adaptation needs show state |
| [D-28](decisions-detail.md#d-28) | 2026-10-28 | Keep 8-bit sets; longer sets don't let the director read a turn |
| [D-29](decisions-detail.md#d-29) | 2026-10-28 | Show state step A (tonight's record as features): not adopted; go to B |
| [D-30](decisions-detail.md#d-30) | 2026-10-31 | Show state step B (career + tonight's correction): not adopted; C needs exploration and fatigue planning |
| [D-31](decisions-detail.md#d-31) | 2026-10-31 | Sim `base`: HumorGen win rates mapped linearly (retroactive record) |
| [D-32](decisions-detail.md#d-32) | 2026-10-31 | rJokes is for mode statistics only, never joke material |
| [D-33](decisions-detail.md#d-33) | 2026-10-31 | Quote a few rJokes rows in docs; never the dataset |
| [D-34](decisions-detail.md#d-34) | 2026-11-01 | Human vs judge mode ranking: design and pass criteria (fixed before running) |
| [D-35](decisions-detail.md#d-35) | 2026-11-01 | Human `base` for the sim: decile win rates |
| [D-36](decisions-detail.md#d-36) | 2026-11-01 | Human vs judge mode ranking: outcome |
| [D-37](decisions-detail.md#d-37) | 2026-11-01 | Joke length vs upvotes: weak correlation, clear trend |
| [D-38](decisions-detail.md#d-38) | 2026-11-01 | rJokes correlation map: design and rule (fixed before running) |
| [D-39](decisions-detail.md#d-39) | 2026-11-01 | rJokes correlation map: outcome, Exp 10 stands |
| [D-40](decisions-detail.md#d-40) | 2026-11-01 | Correlation-map rule misses a weak link paired with a strong one |
| [D-41](decisions-detail.md#d-41) | 2026-11-01 | Two tastes in the sim: pass criteria (fixed before running) |
| [D-42](decisions-detail.md#d-42) | 2026-11-01 | Two tastes in the sim: outcome, the director choice holds |
| [D-43](decisions-detail.md#d-43) | 2026-11-01 | Demo voice: Kokoro, local, on the CPU |
| [D-44](decisions-detail.md#d-44) | 2026-11-01 | M1 target moves to Dec 8; presentation and report on Dec 14 |
| [D-45](decisions-detail.md#d-45) | 2026-11-01 | Cut banter and roast handling from M1 |
| [D-46](decisions-detail.md#d-46) | 2026-11-01 | Split show state step C: keep the MDP, defer callbacks |
| [D-47](decisions-detail.md#d-47) | 2026-11-01 | Show state step C1: method and pass criteria (fixed before running) |
| [D-48](decisions-detail.md#d-48) | 2026-11-01 | Show state step C1, first run: fails, and the run doesn't test planning fairly |
| [D-49](decisions-detail.md#d-49) | 2026-11-01 | Show state step C1b: fix the scale and targets (fixed before running) |
| [D-50](decisions-detail.md#d-50) | 2026-11-01 | Show state step C1b, bug fixed: one more run (fixed before running) |
| [D-51](decisions-detail.md#d-51) | 2026-11-01 | rJokes may also test judges, still never joke material |
| [D-52](decisions-detail.md#d-52) | 2026-11-01 | Trained joke model and DPO move into M1, from our own premises |
| [D-53](decisions-detail.md#d-53) | 2026-11-01 | Oracle vs planning optimum: keep the oracle as the yardstick, report the true ceiling beside it |
| [D-54](decisions-detail.md#d-54) | 2026-11-01 | Every experiment gets a chart or a stated reason |
| [D-55](decisions-detail.md#d-55) | 2026-11-08 | Live loop: one process, with an append-only log as the contract |
| [D-56](decisions-detail.md#d-56) | 2026-11-08 | Director state is rebuilt from logs at show start |
| [D-57](decisions-detail.md#d-57) | 2026-11-08 | Live reward comes from operator keys, on frozen laugh levels |
| [D-58](decisions-detail.md#d-58) | 2026-11-08 | Explore at ε = 0.2 and pick candidates uniformly, with logged probabilities |
| [D-59](decisions-detail.md#d-59) | 2026-11-08 | Record raw show audio, with consent |
| [D-60](decisions-detail.md#d-60) | 2026-11-08 | Laugh detector arrives in stages, with buttons as the fallback |
| [D-61](decisions-detail.md#d-61) | 2026-11-08 | M1 shows validate the judge and DPO; training waits for about 30 shows |
| [D-62](decisions-detail.md#d-62) | 2026-11-08 | PR standard: one task, reviewed before it reaches the user |
| [D-63](decisions-detail.md#d-63) | 2026-11-08 | Safety filter: rules first, classifier only if they miss (Exp 19) |
| [D-64](decisions-detail.md#d-64) | 2026-11-08 | Scraps from the Loft permission, on record |
| [D-65](decisions-detail.md#d-65) | 2026-11-08 | One decisions notebook per project section |
| [D-66](decisions-detail.md#d-66) | 2026-11-08 | Joke bank gets bit structure, transcript statistics stay undecided |
| [D-67](decisions-detail.md#d-67) | 2026-11-08 | Exp 18: exploring at ε = 0.2 misses its bar in one of six rooms |
| [D-68](decisions-detail.md#d-68) | 2026-11-08 | Keep ε = 0.2; the cost bar no longer covers the doubled-judge stress room |
| [D-69](decisions-detail.md#d-69) | 2026-11-08 | Safety filter: rules plus an LLM check |
| [D-70](decisions-detail.md#d-70) | 2026-11-08 | Safety filter: rules plus an LLM check (Exp 19 re-run) |
| [D-71](decisions-detail.md#d-71) | 2026-11-08 | Safety check latency bar moves to 3 s at p95; red-team list stays in the repo |
| [D-72](decisions-detail.md#d-72) | 2026-11-08 | Safety bar: at least 75% on fresh indirect lines; no pre-approval of scripts |
| [D-73](decisions-detail.md#d-73) | 2026-11-08 | Safety bar met on the second fresh set; T8 accepted |
| [D-74](decisions-detail.md#d-74) | 2026-11-14 | Generator: one call per bit, render-ahead, bridge lines and a bank fallback (Exp 20) |
| [D-75](decisions-detail.md#d-75) | 2026-11-14 | Bank fallback lines run through the rules only (user) |
| [D-76](decisions-detail.md#d-76) | 2026-11-14 | A bit plays one candidate thread (user) |
| [D-77](decisions-detail.md#d-77) | 2026-11-14 | Session log v2 adds a bridge event (user) |
| [D-78](decisions-detail.md#d-78) | 2026-11-14 | Exp 20: the generated show meets its timing bars on three shows |
| [D-79](decisions-detail.md#d-79) | 2026-11-15 | Consent-gated raw audio over the websocket (T10) |
| [D-80](decisions-detail.md#d-80) | 2026-11-15 | Exp 21: the click-alignment bar re-tuned to the spread around a constant (user) |
| [D-81](decisions-detail.md#d-81) | 2026-11-15 | Operator UI waveform: full-width bars with a solid idle line, lilac-tinted mono theme (user) |
| [D-82](decisions-detail.md#d-82) | 2026-11-15 | First live show uses Gemini (GENERATOR=llm) (user) |
| [D-83](decisions-detail.md#d-83) | 2026-11-15 | Exp 22: rating after show_end; the acceptance scorer is built, the shows are the user's |
| [D-84](decisions-detail.md#d-84) | 2026-11-15 | Exp 22: review round 1 changes to the rating and the scorer |
| [D-85](decisions-detail.md#d-85) | 2026-11-15 | Exp 22: the acceptance run passes; Stage 0 is done |
| [D-86](decisions-detail.md#d-86) | 2026-11-24 | Split the decision log: plain log for humans, detail log for agents (user) |
