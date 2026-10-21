# TextTiling (S4)

Code: *planned* (`pipeline/bits.py`)

## What it is

TextTiling (Hearst, 1997, *"TextTiling: Segmenting Text into Multi-paragraph Subtopic
Passages"*) splits a long text into topical segments **without supervision**. The idea is
lexical cohesion: sentences about the same topic share vocabulary, so a sudden drop in
similarity between neighbouring stretches of text marks a topic shift.

## The original algorithm

1. **Tokenize** into fixed-size pseudo-sentences (e.g. 20 words), drop stopwords, stem.
2. **Gap similarity.** At every gap between pseudo-sentences, compare the block of *k*
   pseudo-sentences before the gap with the block after it (cosine similarity of word counts).
3. **Smooth** the similarity curve to remove jitter.
4. **Depth score** at each valley: `(left peak − valley) + (right peak − valley)`. A deep valley
   means similar text on both sides with a sharp dip in between, i.e. a topic change.
5. **Boundaries** where depth > `mean − σ/2` (a cutoff relative to the text's own scores),
   snapped to the nearest paragraph or sentence break.

## Our variant: embedding TextTiling

Stand-up is a poor fit for word counts. A bit about "my wife" and a bit about "marriage" share
few exact words, and comics repeat filler ("right?", "you know") everywhere. So:

| Step | Original | Ours |
|---|---|---|
| Unit | 20-word pseudo-sentences | sentences from `clean.txt`, with beats as natural breaks |
| Representation | stemmed word counts | sentence embeddings (small local model) |
| Similarity | cosine of counts | cosine of mean embeddings of *k* sentences each side |
| Boundary snap | paragraph | nearest **beat boundary** (a laugh is a natural end point) |
| Final call | threshold | threshold proposes, then an **LLM confirms or moves** each boundary |

Why keep TextTiling at all if an LLM is in the loop? It's cheap, deterministic and
reproducible. It gives the LLM a short list of candidates instead of a 10,000-word transcript,
and it's a **baseline** the LLM refinement must beat on [Pk / WindowDiff](segmentation-evaluation.md).

## Stand-up specific pitfalls

- **Callbacks** reuse an earlier bit's vocabulary, so similarity *rises* at a real boundary.
  The LLM pass handles these and marks `callback_to`.
- **Tags** sometimes drift in vocabulary while staying on the same premise, which over-splits.
  Snapping to beat boundaries and the "one premise" rule limit this.
- **Crowd work** is short and lexically unrelated to everything around it. It's detected as
  its own bit type, not as noise.

## Parameters to tune (on the 2 hand-labeled sets only)

`k` (block size, ~3–6 sentences), smoothing width, depth cutoff, minimum bit length. Tune on
the hand-labeled sets, then freeze before running on the rest. Otherwise the evaluation is
optimistic.
