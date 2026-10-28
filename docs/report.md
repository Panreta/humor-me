# humor-me: a stand-up agent that reads the room (report)

Status: **outline, partly backed** (results from #1 and #4 are in). Each section lists the claim
it must support and where the evidence comes from: a row in [metrics.md](metrics.md), a decision,
or a wiki page. Terms (bit, mode, prior, regret, …) are used as defined in the
[glossary](wiki/glossary.md). A section is "done" when
every claim cites a number from a merged PR. Sections marked *(user)* are written by the
author in their own voice; the rest are drafted by the AI (Claude) and edited by the author.

## 1. Overview *(user)*

LLM can do almost anything. Models are intelligent enough to reason, implement, plan a trip, write code and many more. There's something that's still relevant and where we humans prefer a human touch. Writing, Illustration, Design anything that has a "taste". You can argue that LLM does generate joke, what's different? Give it a list of jokes, or add a RAG and it will fetch a similar joke. But that restricts humor to known set of stale one liner jokes. This is also one of the reasons that though LLM can write stories, but readers notice. In 2026 two novels were pulled by their publishers after readers flagged the prose as machine-like ([1](https://www.nytimes.com/2026/03/19/books/shy-girl-book-ai.html), [2](https://www.chroniclejournal.com/news/national/quebec/publisher-cuts-ties-with-quebec-author-after-saying-it-found-evidence-of-ai-use/article_41f60777-6d02-5670-8ce7-aee3546f7bee.html)). Both authors deny it and AI detectors are unreliable, but people sensed something was off before anyone proved anything. We can easily make out the monotonous language and the lack of depth. I think humor is something very close to this.

LLM lacks taste, Humor needs it. If we observe, the way LLMs generate jokes is sometimes just retrieval. Same list of dad jokes, that fall flat. But, how does a comedian acquire this taste? What sets them apart from a generic one-liner punchlines? How can they come up with "new bits/jokes" spontaneously? That's the million dollar question this project is trying to find.

This repo tries to relate comic generation with RL topics and see if we can mimic a stand-up comedian on a set. The goal is ambitious and this is just an exploratory interest. I believe LLMs can't do stand up. At least, not any good ones. I'm just trying to find evidence how bad it can get.

A new stand-up comedian is not a writer telling a lengthy story. They write [bits](wiki/glossary.md) - short relevant scenes that are funny. They collect these, perform at couple of venues, measure which bits did well, and which ones fell flat. It's an iterative process.

That loop is reinforcement learning. A comedian tries a bit (an action), the room laughs or
doesn't (a reward), and next time they choose differently. They don't learn it in one night
either: it takes many shows, in many rooms, to know that the cynical stuff kills at a bar and
dies at a company party. Taste, at least part of it, is what that loop leaves behind.

So that's what this project builds: a voice-only stand-up agent, a slightly anxious robot,
that performs a set one bit at a time. Before each bit a **director** picks one of six comedic
**modes** (Neurotic, Cynic, Observer, Wordsmith, Optimist, Absurdist, from
[HumorGen](wiki/llm-humor-vs-standup.md)) based on the room (venue, energy, age) and how the
last bit landed, and it carries what it learned into the next show
([glossary](wiki/glossary.md)). The question: **does adapting to the room beat performing one
fixed persona?**

In [simulation](wiki/simulated-audience.md), it does. After a few hundred shows the adaptive
director ([LinUCB](wiki/contextual-bandits.md)) gets 97% of the laughs of a director that knows
exactly what the room likes, and 18% more than always performing as Neurotic, the persona an
LLM judge rates funniest. What made the difference wasn't the starting guess; it was carrying
lessons from one room to similar rooms ([results](metrics.md)).

That's not proof an LLM can do stand-up. The simulated room's tastes are made up, and the
director only picks the attitude; it doesn't make a joke any funnier. What it shows is that the
comedian's loop, the part that builds taste, can be learned. Whether the jokes underneath are
any good, and how bad it gets in front of real people, is what the live sessions are for.

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

- The director picks a **mode once per bit** (not per joke): a comedian commits to an attitude
  for a whole bit and varies between bits. Per joke, the sim rewarded rotation over fit
  (decisions.md 2026-10-27; metrics.md #1 v1 vs v2).
- Arms = 6 modes; context = venue/energy/age + last bit's mode and reward; reward = mean laugh
  level of the bit's jokes ÷ 3.
- Why a bandit first; why fatigue and callbacks push toward an MDP.
- Wiki: [Contextual bandits](wiki/contextual-bandits.md).

## 5. Method

5.1 Simulated audience: the room's hidden truth (`sim/audience.yaml`, made up except `base`,
    stated plainly), 8 bits × 5 jokes, fatigue, novelty, exact oracle.
5.2 Policies: fixed, random, ε-greedy, LinUCB, Thompson (LinTS). Director's prior: HumorGen,
    uniform, or wrong (shuffled). No switching penalty: per-bit decisions keep the voice coherent.
5.3 Long-form generation: set planner → bit writer → director.
5.4 Data: rJokes prep and mode labels; what transcripts were (and weren't) used for.
5.5 Live system: voice loop, laugh detection, operator fallback, safety filter.

## 6. Key decisions and trade-offs

One row per decision that changed the result (to become D-xx entries: options, trade-off, choice).

| Decision | Status | Evidence |
|---|---|---|
| Pause transcripts; rJokes + simulator first | Decided 2026-10-26 | decisions.md |
| Long-form sets planned hierarchically (planner → bit writer → director) | Decided 2026-10-26 | decisions.md |
| Reward = laugh level 0–3, scaled to [0, 1] | Decided 2026-10-27 | sim/README.md |
| Director decides per bit, not per joke | Decided 2026-10-27 | metrics.md #1 v1 vs v2 |
| Director policy: LinUCB + uniform prior | Decided 2026-10-27 | metrics.md #4 |
| Learn across shows (one show is too short) | Decided 2026-10-27 | metrics.md #1: 14–177 shows to separate top modes |
| Explorer for GPU jobs only | Decided 2026-10-26 | decisions.md |
| Human mode preference: rJokes upvotes vs HumorGen judge | Open | explore-next.md #1 |

## 7. Results

| Experiment | Metric | Status | Headline so far |
|---|---|---|---|
| Is there anything to learn? | Oracle vs best fixed vs random, laughs/show | Done (#1) | Oracle 47–50, best fixed 43–47, random 38–41 |
| Policies × priors across a career | Laughs lost vs oracle over 300 shows; cold start; experienced | Done (#4) | LinUCB: 97% of oracle, +18% vs always-Neurotic; ε-greedy loses 3.4× more; prior makes no significant difference |
| Mid-set preference shift | Bits until recovery | Next (explore-next #4) | |
| Callbacks: greedy vs sequential | Late-set laugh rate, laughs lost | Planned (explore-next #5) | |
| Prior strength sweep | Laughs lost vs prior weight | Planned (explore-next #7) | |
| Human vs LLM-judge mode ranking | Rank correlation, rJokes upvotes vs HumorGen | Blocked on Explorer labels | |
| Room's `base` from human data | Change in policy ranking | After the above | |
| Long-form sets | Word budget, closer present, planned callbacks used | Planned | |
| Live pilots | Detector vs operator agreement; episode rating | Weeks 5–7 | |

Dropped: switching-penalty ablation (per-bit decisions made it unnecessary).

## 8. Limitations

Made-up audience preferences; Reddit taste ≠ a room; laugh level as a noisy proxy; tiny live
sample; LLM judge ≠ humans (HumorGen: 58.3% judge–human agreement).

## 9. Future work

Learned reward model from logged sessions; DPO on the generator; judge-as-audience (idea B);
H1 live experiment with enough sessions for confidence intervals.

## 10. Acknowledgements and AI use

Overview: paragraphs 1–4 written by the author; paragraphs 5–8 (RL loop → what was built →
result → limits) drafted by the AI (Claude) at the author's request, then edited by the author.
State exactly what the AI (Claude) wrote: code (by module), docs, wiki pages, report sections,
and what the author wrote or decided. Cite datasets (rJokes), papers (HumorGen, LoL,
Christiano et al. 2017, LinUCB, Thompson tutorial) and the Explorer cluster (NURC).

## References

Numbered in order of first use. Notes say what each source does and doesn't support.

1. Alter, A. (2026, March 19). Horror novel "Shy Girl" canceled over suspected A.I. use.
   *The New York Times*. https://www.nytimes.com/2026/03/19/books/shy-girl-book-ai.html
   Note: readers flagged the prose first (Reddit, Jan 2026; a YouTube essay with 1.2M views);
   Hachette cancelled the US edition and discontinued the UK one. The author denies using AI
   (blames a freelance editor); the AI-detector evidence was later disputed. Article not read
   directly (paywall); details cross-checked with Wikipedia ("Shy Girl") and Publishers
   Marketplace (2026-03).
2. Glorieux-Stryckman, M. (2026, October 21). Publisher cuts ties with Quebec author after
   saying it found evidence of AI use. *The Canadian Press* (via The Chronicle-Journal).
   https://www.chroniclejournal.com/news/national/quebec/publisher-cuts-ties-with-quebec-author-after-saying-it-found-evidence-of-ai-use/article_41f60777-6d02-5670-8ce7-aee3546f7bee.html
   Note: suspicion began on social media; Boréal cited document metadata and an external
   manuscript analysis, won't reprint. The author denies it; the publisher says it "will likely
   never have a definitive answer."

