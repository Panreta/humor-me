"""R2: label a uniform sample of rJokes with a HumorGen mode + safety flags, using a local LLM (vLLM).

Usage (on an Explorer GPU node, from the repo root):
    python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n 30000
    python explorer/label_modes.py --input ... --n 50 --out /tmp/smoke.jsonl     # smoke test

Uniform (not stratified) sample so per-mode score distributions need no reweighting.
Writes one JSON line per joke: id, pct, score, mode, mode2, safety{...}, raw on parse failure.
No joke text in the output, so results can be copied back and committed if useful.
"""
import argparse
import json
import random
import re
import time

MODES = {
    "neurotic": "internal anxiety, overthinking, social insecurity (tension release)",
    "cynic": "hypocrisy, biting sarcasm, moral contradictions (superiority, social critique)",
    "observer": "mundane minutiae, unwritten awkward social norms (incongruity)",
    "wordsmith": "puns, double meanings, sound play (linguistic ambiguity)",
    "optimist": "wholesome reinterpretation of something negative (benign violation)",
    "absurdist": "non-sequiturs, dream logic, broken cause and effect (surrealism)",
}
SAFETY = {
    "targets_group": "the punchline depends on a race, religion, nationality, gender, sexuality or disability",
    "about_bodies": "mocks appearance, weight, or body parts",
    "sexual": "sexual content or innuendo",
    "alcohol_drugs": "alcohol or drug use",
    "slur": "contains a slur",
}

SYSTEM = "You classify jokes for a research dataset. Answer with one JSON object and nothing else."
TEMPLATE = """Joke:
<<<{joke}>>>

1. Which comedic mode does this joke mainly use? Pick one, and optionally a second:
{modes}
Use "none" if no mode fits.

2. Safety flags, true or false:
{safety}

Return JSON: {{"mode": "...", "mode2": "...|null", {keys}}}"""

PROMPT = TEMPLATE.replace("{modes}", "\n".join(f'- "{k}": {v}' for k, v in MODES.items())) \
    .replace("{safety}", "\n".join(f'- "{k}": {v}' for k, v in SAFETY.items())) \
    .replace("{keys}", ", ".join(f'"{k}": false' for k in SAFETY))
JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse(text: str) -> dict | None:
    m = JSON_OBJ.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    mode = str(d.get("mode", "")).lower()
    if mode not in MODES and mode != "none":
        return None
    mode2 = str(d.get("mode2") or "").lower()
    return {
        "mode": mode,
        "mode2": mode2 if mode2 in MODES and mode2 != mode else None,
        "safety": {k: bool(d.get(k, False)) for k in SAFETY},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/processed/rjokes/modes.jsonl")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # imported late so --help works without a GPU

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n]

    llm = LLM(model=args.model, max_model_len=args.max_model_len, seed=args.seed)
    params = SamplingParams(temperature=0.0, max_tokens=120)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": PROMPT.replace("{joke}", r["text"])}] for r in rows]

    t0 = time.time()
    outs = llm.chat(convs, params)
    dt = time.time() - t0

    failed = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            lab = parse(text)
            rec = {"id": r["id"], "score": r["score"], "pct": r["pct"], "model": args.model}
            if lab:
                rec.update(lab)
            else:
                failed += 1
                rec["raw"] = text[:200]
            f.write(json.dumps(rec) + "\n")
    print(f"{len(rows)} jokes in {dt:.0f}s ({len(rows) / dt:.1f}/s), parse failures {failed}")


if __name__ == "__main__":
    main()
