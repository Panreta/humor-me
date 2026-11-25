"""Step 2: rank each premise's candidates with the judge (docs/plans/sft-dpo-ogrpo.md).

Usage (cluster, repo root):
    python -m explorer.rank_pool --candidates runs/pool/candidates.jsonl --out runs/pool/ranked.jsonl \
        --judge-adapter $COURSE_DIR/runs/dpo_judge_v1/final

Per premise: drop lines the live safety rules reject (teens rules, explorer/safety_rules.py) and exact
duplicates, then score every ordered pair (n(n-1) prompts; 552 for 24 lines) with the judge's A/B logits and fit
Bradley-Terry (judge.rank.rank_from_probs). Premises left with fewer than --min-lines are skipped and reported.
Scoring is batched across --chunk premises at a time, and the output is resumable by premise.
Output rows: premise_id, premise, split, judge, n_calls, first_shown_win_share, consistent_share,
lines [{text, mode, k, theta, se, borda}] sorted best first, unsafe [{text, mode, reason}].
"""
import argparse
import time

from explorer.pool_io import append_jsonl, chunks, read_jsonl


def group(rows: list[dict]) -> list[list[dict]]:
    """Candidate rows grouped by premise, in first-seen order."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["premise_id"], []).append(r)
    return list(out.values())


def screen(lines: list[dict], verdict) -> tuple[list[dict], list[dict]]:
    """(kept, unsafe): unsafe lines with their reason; exact duplicate texts dropped (first kept)."""
    kept, unsafe, seen = [], [], set()
    for r in lines:
        ok, reason = verdict(r["text"])
        if not ok:
            unsafe.append({"text": r["text"], "mode": r["mode"], "reason": reason})
        elif r["text"] not in seen:
            seen.add(r["text"])
            kept.append(r)
    return kept, unsafe


def ranked_row(lines: list[dict], unsafe: list[dict], p_first: dict, judge_name: str) -> dict:
    from judge.rank import rank_from_probs

    n = len(lines)
    rk = rank_from_probs(n, p_first, judge_name, prompt_id=lines[0]["premise_id"])
    winners: dict[tuple[int, int], list[int]] = {}
    for (i, j), p in p_first.items():
        winners.setdefault((min(i, j), max(i, j)), []).append(i if p > 0.5 else j)
    both = [w for w in winners.values() if len(w) == 2]  # consistent: the same joke wins in both orders
    order = rk.order()
    return {"premise_id": lines[0]["premise_id"], "premise": lines[0]["premise"], "split": lines[0]["split"],
            "judge": judge_name, "n_calls": rk.n_calls,
            "first_shown_win_share": sum(p > 0.5 for p in p_first.values()) / max(len(p_first), 1),
            "consistent_share": sum(w[0] == w[1] for w in both) / max(len(both), 1),
            "lines": [{"text": lines[i]["text"], "mode": lines[i]["mode"], "k": lines[i]["k"],
                       "theta": float(rk.theta[i]), "se": float(rk.se[i]), "borda": float(rk.borda[i])}
                      for i in order],
            "unsafe": unsafe}


def rank_chunk(groups: list[list[dict]], judge, verdict, min_lines: int) -> tuple[list[dict], list[str]]:
    """Ranked rows for these premises (one judge.score call for all their pairs), and skipped premise ids."""
    from judge.rank import ordered_pairs

    jobs, prompts, skipped = [], [], []
    for lines in groups:
        kept, unsafe = screen(lines, verdict)
        if len(kept) < min_lines:
            skipped.append(lines[0]["premise_id"])
            continue
        pairs = ordered_pairs(len(kept))
        jobs.append((kept, unsafe, pairs, len(prompts)))
        prompts += [judge.message(kept[0]["premise"], kept[i]["text"], kept[j]["text"]) for i, j in pairs]
    scores = judge.score(prompts) if prompts else []
    rows = [ranked_row(kept, unsafe, {pair: scores[start + t] for t, pair in enumerate(pairs)}, judge.name)
            for kept, unsafe, pairs, start in jobs]
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidates", default="runs/pool/candidates.jsonl")
    ap.add_argument("--out", default="runs/pool/ranked.jsonl")
    ap.add_argument("--judge-base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--judge-adapter", default=None, help="the step 0 judge; leave out only for a smoke test")
    ap.add_argument("--min-lines", type=int, default=8, help="skip premises with fewer safe lines")
    ap.add_argument("--chunk", type=int, default=20, help="premises per scoring call and checkpoint")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    groups = group(read_jsonl(args.candidates))
    done = {r["premise_id"] for r in read_jsonl(args.out)}
    todo = [g for g in groups if g[0]["premise_id"] not in done]
    print(f"{len(done)} premises ranked, {len(todo)} to go", flush=True)
    if not todo:
        return

    from explorer.judge_model import load, make_scorer
    from explorer.safety_rules import verdict
    from judge.judges import LogitJudge

    tok, model = load(args.judge_base, args.judge_adapter, args.dtype)
    name = f"logit:{args.judge_base}" + (f"+{args.judge_adapter}" if args.judge_adapter else "")
    judge = LogitJudge(make_scorer(tok, model, args.batch_size), name=name)
    t0, n_done, n_skipped = time.time(), 0, 0
    for batch in chunks(todo, args.chunk):
        rows, skipped = rank_chunk(batch, judge, verdict, args.min_lines)
        append_jsonl(args.out, rows)
        n_done, n_skipped = n_done + len(batch), n_skipped + len(skipped)
        if skipped:
            print(f"skipped (under {args.min_lines} safe lines): {skipped}", flush=True)
        print(f"{len(done) + n_done}/{len(groups)} premises, {n_done / (time.time() - t0) * 60:.1f}/min, "
              f"skipped {n_skipped}", flush=True)


if __name__ == "__main__":
    main()
