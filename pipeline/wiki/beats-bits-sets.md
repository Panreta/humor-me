# Beats, bits, sets (S3, S5)

Code: *planned* (`pipeline/beats.py`, `pipeline/annotate.py`) · Schemas: [docs/pipeline.md](../../docs/pipeline.md#schemas)

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

## Bits (S4 boundaries, S5 annotation)

- Boundaries come from [TextTiling](texttiling.md), refined by an LLM.
- Rules: a bit is **one premise**; tags stay inside the bit; a callback is a *new* bit with
  `callback_to`; crowd work is its own bit type.
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
