# Beats, bits, sets (S3, S5)

Code: [`pipeline/beats.py`](../beats.py) (S3, built) · `pipeline/annotate.py` (S5, planned) · Schemas: [docs/pipeline.md](../../docs/pipeline.md#schemas)

## The three units

```text
set   ─ the whole special (opener … closer)
 └ bit  ─ one premise, explored with tags and act-outs ("airport security is theater")
    └ beat ─ the text between two audience reactions: a setup, a punch, a tag
```

Stand-up isn't a list of jokes. A comic picks a **premise**, gets a laugh, then **tags** it
(extra punchlines on the same idea), maybe **acts it out**, and moves on with a **segue**.
Later bits **call back** to earlier ones. To learn *structure* (which RL needs for "what bit
next?"), the data must keep these levels.

## Beats (S3), rule-based, no LLM

- A beat = text from the previous audience reaction to the next one.
- Label = level of the reaction that follows (0 if none before the next beat), plus flags,
  word count and whether the laugh interrupted a sentence.
- Very long reaction-free spans are split into sentence groups so no "beat" is a whole story.
- This is what the laugh predictor trains on: *(beat text, context) → laugh level*.

## Lines (S3): the laugh predictor's input

Beats end at laughs, so ~77% of them are positive and the few negatives are long chunks: beat
length alone predicts the label (median 23 vs 74 words). Training on beats would teach "short =
funny". So each beat is split into sentences:

- the **last line** before a reaction gets its level (a final line under 3 words, like "Yeah.",
  is folded into the punchline before it);
- earlier lines in the beat (setup) get 0;
- each line carries `context`: the 150 words before it (becomes bit-scoped after S4).

Result on the 11 labeled sets: 13,395 lines, laugh rate 0.15, median length 7 vs 6 words, so
length no longer gives the label away. Laugh rate varies by set (0.08–0.25) with transcriber
density, so evaluate per set or on held-out comedians.

## Bits (S4 boundaries, S5 annotation)

- Boundaries come from [TextTiling](texttiling.md), refined by an LLM.
- Rules: a bit is the **smallest unit that works on its own** without earlier setup (one
  premise + its tags); consecutive same-topic bits are marked `continues` and form a **thread**;
  a callback is a *new* bit with `callback_to` at least 2 bits back; crowd work is its own type.
- Annotation (LLM): paraphrased `premise`, persona `mode` (Neurotic, Observer, Absurdist,
  Cynic, Optimist, Wordsmith), `techniques`, `type`, `callback_to`, `segue_from_prev`.
- `position` (opener / body / closer) comes from the bit's index, not from the LLM.
- Premises are **paraphrased**; annotations never quote the transcript.

## Sets

- An ordered list of bits plus metadata (`laugh_labeled`, split).
- A set is a **trajectory**: state (what's been done, how it landed) → action (next bit's mode
  and type) → reward (laughs). That's the demonstration data for the RL planner.

## What each unit feeds

| Unit | Consumer |
|---|---|
| beats | laugh predictor (reward model), sim audience timing priors |
| bits | skeletons → synthetic robot-voice sets → generator fine-tuning |
| sets | planner behavior-cloning prior, set-arc statistics |
