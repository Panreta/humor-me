# LLM humor vs stand-up

**Where it fits:** the motivation for the whole project and the first chapter of the report.
**Used in:** generator prompts (core identity × modes), the [set planner](standup-structure.md),
the reranker, and the reason a [bandit](contextual-bandits.md) picks modes at all.

## What goes wrong when you ask an LLM to be funny

| Gap | Why it happens | What a comedian does instead |
|---|---|---|
| **Average jokes** | Decoding favors high-probability text, and humor lives in surprising-but-fitting continuations. HumorGen calls this mode collapse | Writes many versions and keeps the one that surprises |
| **No audience** | The model never hears the room | Adjusts on the next line from the laugh (or silence) to the last one |
| **No timing** | Text has no pauses and doesn't wait for laughs | Holds through the laugh and comes back in as it fades |
| **No structure** | One-pass output is a list of jokes, or a story that drifts | Premise → tags → bits → set, with callbacks planned |
| **No memory across shows** | Every prompt starts fresh | Keeps what killed, cuts what bombed, show after show |

## What the papers we build on add

- **HumorGen** ([anonymous review repo](https://anonymous.4open.science/r/HumorGen-2697)):
  six *cognitive personas* (Neurotic, Cynic, Observer, Wordsmith, Optimist, Absurdist), each tied
  to a humor theory, push generation away from the generic middle. An LLM judge ranks candidates
  (Elo), and the winners are distilled into a small model. Their per-persona win rates seed our
  sim's `base` values. In our project the personas become **modes** of one identity (invariant 3).
- **LoL** ([arXiv 2410.10370](https://arxiv.org/abs/2410.10370)):
  "thought-leap" prompting (connect distant ideas on purpose) plus a frozen pairwise judge for
  online DPO. It does *not* classify personas.

Neither paper has an audience, timing, or set structure. That gap is this project.

## In this project

- Generation = core identity × mode, K candidates, reranked.
- Structure = hierarchical planning ([stand-up structure](standup-structure.md)).
- Audience = laugh signal → [bandit](contextual-bandits.md) over modes, tested first in a
  [simulated room](simulated-audience.md).

## Further reading

- [HumorGen (anonymous ACL submission; code and data)](https://anonymous.4open.science/r/HumorGen-2697) and
  [LoL, Wang et al., ICLR 2025](https://arxiv.org/abs/2410.10370): read Table 1 and Table 4 of HumorGen for the personas
  and win rates. Markdown conversions live in `references/md/` (local only, not committed).
- [The rJokes Dataset (LREC 2020)](https://aclanthology.org/2020.lrec-1.753/): shows that
  predicting how funny a joke is from text alone is hard; useful for setting expectations.
