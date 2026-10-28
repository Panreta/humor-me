# humor-me: a stand-up agent that reads the room (report)

Status: **outline**. Each section lists the claim it must support and where the evidence comes
from: a row in [metrics.md](metrics.md), a decision, or a wiki page. A section is "done" when
every claim cites a number from a merged PR. Sections marked *(user)* are written by the
author in their own voice; the rest are drafted by the AI (Claude) and edited by the author.

## 1. Overview *(user)*

One paragraph: what was built (a voice-only stand-up agent that adapts its comedic mode to
laughter), the RL question (does adapting to the room beat a fixed persona?), the headline
result, and the demo.

## 2. Why LLM jokes differ from a stand-up comedian

- Claim: one-shot LLM humor is generic, deaf to the room, untimed and unstructured.
- Evidence: HumorGen mode collapse and persona win rates; rJokes noise ceiling (repost spread).
- Wiki: [LLM humor vs stand-up](wiki/llm-humor-vs-standup.md).

## 3. How a comedian writes, and how that maps onto an agent

- Premise → setup/punch → tags → bit → set → callback.
- Mapping: bit ↔ mode choice; tag-or-move-on ↔ director; callback ↔ delayed reward;
  show-to-show rewriting ↔ learning across episodes.
- Evidence: transcript pipeline (bits F1 0.80 vs 0.21 baseline on the tuning set; held-out
  Birbiglia score if run).
- Wiki: [Stand-up structure](wiki/standup-structure.md).

## 4. From bits to a contextual bandit (and where it stops being one)

- Arms = modes, context = venue/energy/age (+ recent history), reward = laugh level / 3.
- Why a bandit first; why fatigue and callbacks push toward an MDP.
- Wiki: [Contextual bandits](wiki/contextual-bandits.md).

## 5. Method

5.1 Simulated audience: model, made-up priors (stated plainly), exact oracle.
5.2 Policies: fixed, random, ε-greedy, LinUCB, Thompson; ± switching penalty.
5.3 Long-form generation: set planner → bit writer → director.
5.4 Data: rJokes prep and mode labels; what transcripts were (and weren't) used for.
5.5 Live system: voice loop, laugh detection, operator fallback, safety filter.

## 6. Key decisions and trade-offs

One row per D-xx entry that changed the result (options, trade-off, choice). Candidates so far:
pause transcripts for rJokes; hierarchical long-form generation; reward = level / 3; fatigue
strength vs venue offsets; learn within vs across shows; Explorer for GPU jobs only.

## 7. Results

| Experiment | Metric | Source |
|---|---|---|
| Sim sanity: is there anything to learn? | Oracle vs fixed vs random laughs/set | metrics.md, PR #1 |
| Policies within one show | Cumulative regret, laughs/set, 3 venues | PR 2 |
| Policies across shows | Regret vs number of shows per venue | PR 2 |
| Switching penalty ablation | Laughs/set vs mode switches | PR 2 |
| Mid-set preference shift | Jokes until recovery | PR 2 |
| Callbacks: greedy vs sequential | Late-set laugh rate, regret | PR 2b |
| rJokes priors vs hand priors | Regret change | PR 3 |
| Long-form sets | Word budget, closer present, planned callbacks used | PR 4 |
| Live pilots | Detector vs operator agreement; episode rating | Weeks 5–7 |

## 8. Limitations

Made-up audience preferences; Reddit taste ≠ a room; laugh level as a noisy proxy; tiny live
sample; LLM judge ≠ humans (HumorGen: 58.3% judge–human agreement).

## 9. Future work

Learned reward model from logged sessions; DPO on the generator; judge-as-audience (idea B);
H1 live experiment with enough sessions for confidence intervals.

## 10. Acknowledgements and AI use

State exactly what the AI (Claude) wrote: code (by module), docs, wiki pages, report sections,
and what the author wrote or decided. Cite datasets (rJokes), papers (HumorGen, LoL,
Christiano et al. 2017, LinUCB, Thompson tutorial) and the Explorer cluster (NURC).
