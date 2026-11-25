"""Label the R2 sample (the jokes in modes.jsonl) with the room each joke suits best: venue, energy,
age, in the sim's vocabulary (a test keeps them equal; sim isn't imported so the cluster venv needs no yaml). For DPO experiments; the labels are the LLM's guess,
not anything measured in a room.

Usage (on an Explorer GPU node, from the repo root; -m so the explorer package imports):
    python -m explorer.label_state --n 50 --out /tmp/state_smoke.jsonl     # smoke test
    python -m explorer.label_state                                          # all of modes.jsonl
    python -m explorer.label_state --export data/processed/rjokes/dpo_state.jsonl   # join, CPU only

Resumable like label_modes.py. The label file holds no joke text; --export writes text + mode + state
in one file to hand over, which stays local (rJokes text is never committed).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from explorer.label_modes import JSON_OBJ, chunks, load_done, make_generator

VENUE = {
    "bar": "late-night bar crowd, a few drinks in, came for a night out",
    "company_event": "colleagues at a work event, bosses in the room, must stay office-safe",
    "showcase": "comedy-club showcase, people who came for comedy and like clever or odd material",
}
ENERGY = {
    "low": "tired or quiet room: the joke must land easily, gentle, short setup",
    "normal": "an ordinary attentive room",
    "high": "lively, loud room ready for fast, edgy or big material",
}
AGE = {
    "teens": "13 to 17 years old: nothing sexual, no drugs or alcohol",
    "young_adults": "18 to 30",
    "mixed": "all ages together, including older people",
}

SYSTEM = "You classify jokes for a research dataset. Answer with one JSON object and nothing else."


def _block(d: dict) -> str:
    return "\n".join(f'- "{k}": {v}' for k, v in d.items())


PROMPT = f"""Joke:
<<<{{joke}}>>>

A comedian wants to tell this joke on stage. Which room would it work best in?
1. venue, pick one:
{_block(VENUE)}
2. energy of the room, pick one:
{_block(ENERGY)}
3. audience age, pick one:
{_block(AGE)}

Return JSON: {{"venue": "...", "energy": "...", "age": "..."}}"""

FIELDS = {"venue": VENUE, "energy": ENERGY, "age": AGE}


def parse(text: str) -> dict | None:
    m = JSON_OBJ.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {k: str(d.get(k, "")).lower().strip() for k in FIELDS}
    return out if all(out[k] in FIELDS[k] for k in FIELDS) else None


def sample_ids(modes_path: pathlib.Path) -> list[str]:
    """R2's ids in their labeling order, which is the seed's shuffle, so --n takes a uniform prefix."""
    return [json.loads(line)["id"] for line in open(modes_path, encoding="utf-8")]


def export(modes_path, state_path, input_path, out_path) -> tuple[int, int]:
    """One row per joke with a valid mode and state label. Returns (written, skipped)."""
    states = {r["id"]: r for r in map(json.loads, open(state_path, encoding="utf-8")) if "venue" in r}
    modes = {r["id"]: r for r in map(json.loads, open(modes_path, encoding="utf-8")) if r.get("mode")}
    keep = states.keys() & modes.keys()
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for line in open(input_path, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in keep:
                continue
            m, s = modes[r["id"]], states[r["id"]]
            f.write(json.dumps({
                "id": r["id"], "text": r["text"], "score": m["score"], "pct": m["pct"],
                "mode": m["mode"], "mode2": m.get("mode2"), "safety": m.get("safety"),
                "venue": s["venue"], "energy": s["energy"], "age": s["age"],
            }) + "\n")
            written += 1
    return written, len(modes.keys() | states.keys()) - written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/rjokes/train.jsonl")
    ap.add_argument("--modes", default="data/processed/rjokes/modes.jsonl")
    ap.add_argument("--out", default="data/processed/rjokes/state.jsonl")
    ap.add_argument("--n", type=int, default=None, help="label only the first n of the R2 sample")
    ap.add_argument("--export", default=None, help="write text + mode + state to this path and exit")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--dtype", default="auto", help="auto | half | bfloat16 (use half on V100)")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if args.export:
        written, skipped = export(args.modes, args.out, args.input, args.export)
        print(f"wrote {written} rows to {args.export}; {skipped} without both labels left out")
        return

    out = pathlib.Path(args.out)
    done, models = load_done(out)
    if models - {args.model}:
        sys.exit(f"{out} was labeled by {sorted(models)}; refusing to mix in {args.model}. Use another --out.")
    ids = sample_ids(pathlib.Path(args.modes))[:args.n]
    want = {i for i in ids if i not in done}
    text = {}
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in want:
            text[r["id"]] = r["text"]
    todo = [i for i in ids if i in text]
    print(f"{len(done)} already labeled; {len(todo)} to go", flush=True)
    if not todo:
        return

    generate = make_generator(args)
    t0, labeled, failed = time.time(), 0, 0
    for batch in chunks(todo, args.chunk):
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": PROMPT.replace("{joke}", text[i])}] for i in batch]
        outs = generate(convs)
        with open(out, "a", encoding="utf-8") as f:
            for i, reply in zip(batch, outs):
                rec = {"id": i, "model": args.model, "seed": args.seed}
                lab = parse(reply)
                if lab:
                    rec.update(lab)
                else:
                    failed += 1
                    rec["raw"] = reply[:200]
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        labeled += len(batch)
        print(f"{len(done) + labeled}/{len(ids)} labeled, {labeled / (time.time() - t0):.1f}/s, "
              f"parse failures {failed}", flush=True)


if __name__ == "__main__":
    main()
