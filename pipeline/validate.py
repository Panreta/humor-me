"""S6: score predicted bits (S4) against hand labels.

Usage: python -m pipeline.validate [<set_id> ...]     # default: every set with data/labels/<set>.yaml

Metrics (units = lines, the same units the labels and predictions use):
- Pk (Beeferman et al. 1999) and WindowDiff (Pevzner & Hearst 2002): lower is better, 0 = perfect.
- Boundary precision/recall/F1 with a ±TOLERANCE-line window: easier to read than Pk.
- Callback precision/recall: a predicted callback counts if both its bit start and its target
  start land within the tolerance of a hand-labeled callback.
- Baseline: the same number of bits spaced evenly. The LLM should beat it clearly.
"""
import json
import sys

import yaml

from pipeline.common import PROCESSED, ROOT

LABELS = ROOT / "data" / "labels"
TOLERANCE = 2


def segment_ids(starts: list[int], n: int) -> list[int]:
    """Position i -> id of the segment containing it. `starts` are positions (0-based), first is 0."""
    ids, seg, s = [], -1, set(starts)
    for i in range(n):
        seg += i in s
        ids.append(seg)
    return ids


def window_k(ref_starts: list[int], n: int) -> int:
    return max(2, round(n / max(len(ref_starts), 1) / 2))


def pk(ref: list[int], hyp: list[int], n: int) -> float:
    k = window_k(ref, n)
    r, h = segment_ids(ref, n), segment_ids(hyp, n)
    trials = n - k
    return sum((r[i] == r[i + k]) != (h[i] == h[i + k]) for i in range(trials)) / trials


def window_diff(ref: list[int], hyp: list[int], n: int) -> float:
    k = window_k(ref, n)
    r, h = segment_ids(ref, n), segment_ids(hyp, n)
    trials = n - k
    return sum((r[i + k] - r[i]) != (h[i + k] - h[i]) for i in range(trials)) / trials


def boundary_prf(ref: list[int], hyp: list[int], tol: int = TOLERANCE) -> tuple[float, float, float]:
    # Position 0 is always a start; scoring it would inflate every metric.
    ref_b, hyp_b = [b for b in ref if b > 0], [b for b in hyp if b > 0]
    unmatched, tp = list(ref_b), 0
    for b in hyp_b:
        hit = next((r for r in unmatched if abs(r - b) <= tol), None)
        if hit is not None:
            unmatched.remove(hit)
            tp += 1
    p = tp / len(hyp_b) if hyp_b else 0.0
    rc = tp / len(ref_b) if ref_b else 0.0
    return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)


def callback_pr(ref: list[tuple[int, int]], hyp: list[tuple[int, int]], tol: int = TOLERANCE):
    def match(a, b):
        return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol
    tp_h = sum(any(match(h, r) for r in ref) for h in hyp)
    tp_r = sum(any(match(r, h) for h in hyp) for r in ref)
    return (tp_h / len(hyp) if hyp else None), (tp_r / len(ref) if ref else None)


def even_baseline(n_bits: int, n: int) -> list[int]:
    return sorted({round(i * n / n_bits) for i in range(n_bits)})


def load(set_id: str):
    manifest = json.loads((PROCESSED / "datasets.json").read_text(encoding="utf-8"))
    dataset = "laugh_labeled" if set_id in manifest["laugh_labeled"] else "structure_only"
    with open(PROCESSED / dataset / "lines.jsonl", encoding="utf-8") as f:
        order = [r["line_idx"] for r in map(json.loads, f) if r["set_id"] == set_id]
    pos = {li: i for i, li in enumerate(order)}

    lab = yaml.safe_load((LABELS / f"{set_id}.yaml").read_text(encoding="utf-8"))
    ref = sorted(pos[b["line"]] for b in lab["bit_starts"])
    ref_cb = [(pos[b["line"]], pos[b["callback_to"]]) for b in lab["bit_starts"] if "callback_to" in b]

    bits_path = PROCESSED / dataset / "bits.jsonl"
    with open(bits_path, encoding="utf-8") as f:
        bits = [b for b in map(json.loads, f) if b["set_id"] == set_id]
    if not bits:
        sys.exit(f"{set_id}: no predicted bits; run `python -m pipeline.bits {set_id}`")
    hyp = [pos[b["start_line"]] for b in bits]
    hyp_cb = [(pos[b["start_line"]], pos[bits[b["callback_to"]]["start_line"]])
              for b in bits if b["callback_to"] is not None]
    return len(order), ref, hyp, ref_cb, hyp_cb


def score(set_id: str) -> dict:
    n, ref, hyp, ref_cb, hyp_cb = load(set_id)
    base = even_baseline(len(hyp), n)
    out = {"set_id": set_id, "lines": n, "ref_bits": len(ref), "hyp_bits": len(hyp)}
    for name, starts in (("llm", hyp), ("even_baseline", base)):
        p, r, f = boundary_prf(ref, starts)
        out[name] = {"pk": round(pk(ref, starts, n), 3), "windowdiff": round(window_diff(ref, starts, n), 3),
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
    cp, cr = callback_pr(ref_cb, hyp_cb)
    out["callbacks"] = {"ref": len(ref_cb), "hyp": len(hyp_cb), "precision": cp, "recall": cr}
    return out


def main() -> None:
    sets = sys.argv[1:] or sorted(p.stem for p in LABELS.glob("*.yaml"))
    if not sets:
        sys.exit("no hand labels yet: data/labels/<set>.yaml (see `python -m pipeline.labels`)")
    results = [score(s) for s in sets]
    print(f"{'set':42} {'bits ref/llm':>12}  {'method':14} {'Pk':>6} {'WinDiff':>7} {'P':>6} {'R':>6} {'F1':>6}")
    for r in results:
        for name in ("llm", "even_baseline"):
            m = r[name]
            label = f"{r['ref_bits']}/{r['hyp_bits']}" if name == "llm" else ""
            print(f"{r['set_id'][:42] if name == 'llm' else '':42} {label:>12}  {name:14} "
                  f"{m['pk']:6.3f} {m['windowdiff']:7.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")
        c = r["callbacks"]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        print(f"{'':42} {'':>12}  callbacks      ref={c['ref']} llm={c['hyp']} "
              f"precision={fmt(c['precision'])} recall={fmt(c['recall'])}")
    out = PROCESSED / "eval_bits.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\n(tolerance ±{TOLERANCE} lines; lower Pk/WindowDiff is better) -> {out}")


if __name__ == "__main__":
    main()
