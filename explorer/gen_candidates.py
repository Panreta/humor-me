"""Step 1: the candidate pool. For every premise, the teacher writes --per-mode lines in each of the 6 modes
(HumorGen: 4 x 6 = 24 per prompt). docs/plans/sft-dpo-ogrpo.md.

Usage (cluster, repo root):
    python -m explorer.gen_candidates --premises data/premises/premises-v1.jsonl --out runs/pool/candidates.jsonl
    python -m explorer.gen_candidates ... --max-premises 5 --out /tmp/smoke.jsonl        # smoke test

One call per (premise, mode) asks for --per-mode different lines as a JSON list (joke_prompts.TEACHER_SYSTEM).
A call that returns fewer usable lines is asked again, up to --retries times; whatever is still missing is
reported, not invented. Resumable: premises are written in chunks, all 6 modes at once, and a rerun skips
premises already in --out. Both train and eval premises get candidates; build_sets.py trains on train only.
Output rows: premise_id, premise, split, mode, k, text, teacher, prompt_id.
"""
import argparse
import time

from explorer.joke_prompts import MODES, PROMPT_ID, parse_lines, teacher_messages
from explorer.pool_io import append_jsonl, chunks, read_jsonl


def fill(premises: list[dict], generate, per_mode: int, retries: int) -> tuple[list[dict], int]:
    """Candidate rows for these premises, and how many (premise, mode) cells ended short."""
    cells = [(p, m) for p in premises for m in MODES]
    got: dict[int, list[str]] = {i: [] for i in range(len(cells))}
    todo = list(range(len(cells)))
    for _ in range(1 + retries):
        if not todo:
            break
        replies = generate([teacher_messages(cells[i][0]["premise"], cells[i][1], per_mode) for i in todo])
        for i, reply in zip(todo, replies):
            for line in parse_lines(reply):
                if line not in got[i] and len(got[i]) < per_mode:
                    got[i].append(line)
        todo = [i for i in todo if len(got[i]) < per_mode]
    rows = [{"premise_id": p["id"], "premise": p["premise"], "split": p["split"], "mode": m, "k": k,
             "text": text, "prompt_id": PROMPT_ID}
            for i, (p, m) in enumerate(cells) for k, text in enumerate(got[i])]
    return rows, len(todo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--premises", default="data/premises/premises-v1.jsonl")
    ap.add_argument("--out", default="runs/pool/candidates.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct", help="the teacher")
    ap.add_argument("--per-mode", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-premises", type=int, default=-1, help="first n premises only (-1 = all)")
    ap.add_argument("--chunk", type=int, default=40, help="premises per checkpoint")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    premises = read_jsonl(args.premises)
    if args.max_premises > 0:
        premises = premises[:args.max_premises]
    existing = read_jsonl(args.out)
    mixed = {(r["teacher"], r["prompt_id"]) for r in existing} - {(args.model, PROMPT_ID)}
    if mixed:
        raise SystemExit(f"{args.out} holds candidates from {sorted(mixed)}; use another --out")
    done = {r["premise_id"] for r in existing}
    todo = [p for p in premises if p["id"] not in done]
    print(f"{len(done)} premises done, {len(todo)} to go, model {args.model}", flush=True)
    if not todo:
        return

    from explorer.hf_generate import make_generator
    generate = make_generator(args.model, dtype=args.dtype, batch_size=args.batch_size,
                              temperature=args.temperature, max_new_tokens=400, seed=args.seed)
    t0, n_done, short = time.time(), 0, 0
    for batch in chunks(todo, args.chunk):
        rows, n_short = fill(batch, generate, args.per_mode, args.retries)
        append_jsonl(args.out, [{**r, "teacher": args.model} for r in rows])
        n_done, short = n_done + len(batch), short + n_short
        rate = n_done / (time.time() - t0)
        print(f"{len(done) + n_done}/{len(premises)} premises, {rate * 60:.1f}/min, "
              f"short cells so far {short}", flush=True)


if __name__ == "__main__":
    main()
