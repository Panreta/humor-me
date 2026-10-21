# Segmentation evaluation (S6)

Code: [`pipeline/validate.py`](../validate.py) · run `python -m pipeline.validate` after importing hand labels.

Baseline in place of TextTiling (deferred): the same number of bits spaced evenly. The LLM
should clearly beat it; if it doesn't, boundaries are no better than chance-shaped guesses.

## Why not plain precision/recall?

If the true bit boundary is after sentence 40 and we predict sentence 41, exact-match
precision counts that as a miss *and* a false alarm, the same as predicting sentence 200.
Segmentation metrics give partial credit for near misses.

## Pk (Beeferman, Berger & Lafferty, 1999)

Slide a window of width *k* across the text (k = half the average reference segment length).
At each position, ask: "are the two ends of the window in the **same** segment?" Answer it
for the reference and for the prediction. **Pk = fraction of positions where the answers
disagree.**

- 0 = perfect; lower is better.
- Near misses only disagree at a few window positions, so they cost little.
- Known flaws: it penalizes false negatives more than false positives and ignores the number
  of boundaries inside a window.

## WindowDiff (Pevzner & Hearst, 2002)

Same sliding window, but compare the **number** of boundaries inside it:
`WindowDiff = fraction of windows where count_ref ≠ count_pred`.

- Fixes Pk's asymmetry and catches "two boundaries where there should be one".
- Report both; they're the standard pair in segmentation papers.

## Our evaluation protocol

| What | How | Target |
|---|---|---|
| Bit boundaries | Hand-label bits in **2 sets** (one dense, one sparse). Score TextTiling alone vs TextTiling + LLM. | LLM refinement beats TextTiling on both metrics |
| Marker mapping | 50 markers stratified by level, labeled by hand vs lexicon | report agreement |
| Annotations | spot-check 30 bits for `mode` and `premise` | report accuracy, list systematic errors |

Rules:
- Hand-label **before** looking at the model's boundaries, so they don't anchor you.
- Tune TextTiling parameters only on the hand-labeled sets, and report that honestly; with 2
  sets there's no separate test split. That's a stated limitation, not a hidden one.
- Use the same unit (sentences) for reference and prediction.
