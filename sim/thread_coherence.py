"""Exp 27 (D-111): does the connected prompt (gen-v5) write threads whose lines connect, more often than the current
prompt (gen-v4)?

Usage (Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`, `ANTHROPIC_API_KEY` set for the judge):
    python -m sim.thread_coherence run       # generate both arms, judge every thread -> reports/metrics/thread_coherence.json
    python -m sim.thread_coherence review    # you check 10 threads the judge called connected (y/n), arm hidden
    python -m sim.thread_coherence score     # recompute the numbers with your answers in (no model calls)

20 seeded prompts (ids `c00` to `c19`; another seed than Exp 24's `p..` and Exp 26's `t..`). To look like a later bit of a
show, each carries 0 to 4 premises already told (a fixed list of bland topics; `premises_already_told` in the request).
For each prompt both arms write one bit through the show's own path: `generate.writer_client()` and `ThreadGenerator`
(3 parallel threads), after `generate.warm_up`. Arm A is the prompt the show runs, arm B the connected one.

Judge: Claude Haiku 5.5, thinking off (the safety filter's own client settings), one call per thread, blind to the arm,
lines shown in plain punctuation. Question: one stand-up bit, or separate jokes on a shared topic? Verdicts are cached by
the thread's text hash, so a rerun does not pay twice. Generated text lives only under `data/thread_coherence/`
(gitignored); the metrics hold numbers and reason classes only.

Bar, fixed before any run: the mean over prompts of (B's connected rate minus A's), paired by prompt, is above 0 and the
paired bootstrap 95% CI's lower bound is above 0. Spot-check: you answer "is this one bit?" for 10 threads the judge
called connected; if you say no to 4 or more, the judge is not trusted and Exp 27 is inconclusive whatever the CI says.
Reported beside it, with no bar: threads delivered and lost per arm (by reason), `bad_replies`, bit wall time p50/p95.
"""
import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import generate, llm  # noqa: E402
from sim import cache  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim.blind import plain  # noqa: E402
from sim.env import AGES, ENERGIES, VENUES  # noqa: E402

NAME = "thread_coherence"
N_PROMPTS = 20
SEED = "exp27"
ARMS = {"A": "gen-v4", "B": "gen-v5"}
N_BOOT = 10_000
BOOT_SEED = 27
DATA = ROOT / "data" / "thread_coherence"
MAX_TOLD = 4
REVIEW_N = 10
REVIEW_SEED = 27
VOID_AT_NO = 4  # this many "no" answers or more and the judge is not trusted

# Bland placeholder premises, only here to make a prompt look like a later bit of a show.
TOPICS = ("a printer that jams", "waiting on hold", "a gym membership", "assembling furniture", "airport security",
          "a group chat", "the office microwave", "a password reset", "self-checkout machines", "a broken umbrella",
          "parallel parking", "a software update", "a long elevator ride", "meal prepping", "a smart thermostat",
          "autocorrect", "a loyalty card", "a rainy commute")

JUDGE_SYSTEM = """You judge stand-up material. You get five lines in the order they are performed, written as one bit.

Decide whether it is ONE stand-up bit or a list of separate jokes. It is one bit when it has one premise and each line builds on the line before it (a tag on the last punchline, an escalation, a twist), so that the order matters and the lines could not be shuffled or dropped without losing something. It is a list of separate jokes when the lines share a topic but each stands alone, like one-liners on the same subject.

Judge the structure only, never whether it is funny.

Reply with JSON only: {"connected": true} for one bit, {"connected": false} for a list of separate jokes."""
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["connected"],
                "properties": {"connected": {"type": "boolean"}}}


# --- prompts, digests, folders -----------------------------------------------------------------------------------

def prompts(n: int = N_PROMPTS, seed: str = SEED) -> list[dict]:
    """Seeded like Exp 24's: modes round-robin, each context from its own stream. The number of premises already told
    cycles 0..4 so every length is covered; which topics, and their order, are seeded."""
    modes = tuple(generate.MODES)
    out = []
    for i in range(n):
        r = random.Random(f"{seed}:{i}")
        out.append({"id": f"c{i:02d}", "mode": modes[i % len(modes)],
                    "context": {"venue": r.choice(VENUES), "energy": r.choice(ENERGIES), "age": r.choice(AGES)},
                    "told": r.sample(TOPICS, i % (MAX_TOLD + 1))})
    return out


def judge_model() -> tuple[str, str, bool]:
    return cg.SAFETY_PROVIDER, cg.SAFETY_MODEL, cg.SAFETY_THINKING


def writer_model() -> str:
    return llm.setting("JOKE_WRITER_MODEL") or llm.OLLAMA_MODEL


def digest(n: int = N_PROMPTS, seed: str = SEED) -> str:
    """What the generated bits are made of: both prompts, the reply shapes, timing, model and the prompts' contents. A
    change starts a new run folder. The judge has its own digest (`judge_digest`), as its verdicts can be redone alone."""
    parts = [json.dumps({a: [i, generate.PROMPTS[i]] for a, i in ARMS.items()}, sort_keys=True), generate.TEENS_RULE,
             json.dumps(generate.MODES, sort_keys=True), json.dumps(generate.REPLY_SCHEMA, sort_keys=True),
             json.dumps(generate.COUNTED_REPLY_SCHEMA, sort_keys=True),
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.RETRIES,
                  generate.WRITER_CONTEXT, generate.HANG_GRACE_S, llm.MIN_ATTEMPT_S)),
             writer_model(), json.dumps(llm.OLLAMA_NO_THINK, sort_keys=True), "json_schema=True", "reask_bad_shape_once",
             json.dumps(prompts(n, seed), sort_keys=True), seed, str(n)]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8]


def judge_digest() -> str:
    return hashlib.sha256("\x00".join([JUDGE_SYSTEM, json.dumps(JUDGE_SCHEMA, sort_keys=True),
                                       *map(str, judge_model())]).encode()).hexdigest()[:8]


def run_dir(n: int = N_PROMPTS, seed: str = SEED, root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or DATA) / f"run-{digest(n, seed)}"


def params(n: int = N_PROMPTS, seed: str = SEED) -> dict:
    return {"n_prompts": n, "seed": seed, "digest_": digest(n, seed), "judge_digest_": judge_digest()}


def thread_hash(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(plain(x) for x in lines).encode()).hexdigest()[:16]


def _write(path: pathlib.Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(path)


def _read(path: pathlib.Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# --- generating --------------------------------------------------------------------------------------------------

def bit_path(directory: pathlib.Path, pid: str, arm: str) -> pathlib.Path:
    return directory / "bits" / f"{pid}-{arm}.json"


def write_bit(client, prompt: dict, arm: str) -> dict:
    """One bit through the show's own path. The thread text is kept in the record (the run folder is local); the bit's
    counters are deltas of the client's, exact because bits run one at a time."""
    gen = generate.ThreadGenerator(client, prompt_id=ARMS[arm])
    gen._told = list(prompt["told"])  # what a later bit of a show sends: the premises of the threads played so far
    bad0 = client.bad_replies
    start = time.perf_counter()
    threads: list[list[str]] = []
    lost: list[str] = []
    failure = None
    try:
        batches = [gen.generate(prompt["mode"], 0, j, prompt["context"]) for j in range(generate.JOKES)]
        threads = [[b.texts[i] for b in batches] for i in range(len(batches[0].texts))]
        lost = list(batches[0].threads_lost)
    except Exception as e:  # noqa: BLE001 - a bit with no thread is a result, not a crash
        failure = f"{type(e).__name__}"
        lost = list(getattr(e, "threads_lost", [])) or [f"{type(e).__name__}: {e}"[:100]]
    return {"prompt": prompt["id"], "arm": arm, "wall_s": time.perf_counter() - start, "threads": threads,
            "lost": lost, "failure": failure, "bad_replies": client.bad_replies - bad0}


def generate_bits(client, bits: list[dict], directory: pathlib.Path, http=None) -> list[dict]:
    """Writes every missing (prompt, arm) bit, resumable: a finished one is read back. The writer is warmed up and
    checked (100% on the GPU, context 8192) only when something is left to write. The arm order alternates by prompt,
    so a slow stretch of the machine does not fall on one arm."""
    todo = [(p, a) for p in bits for a in ARMS if _read(bit_path(directory, p["id"], a)) is None]
    if todo:
        probe = generate.ThreadGenerator(client)
        generate.warm_up(probe, http) if http else generate.warm_up(probe)
    out = {}
    for i, p in enumerate(bits):
        for arm in (ARMS if i % 2 == 0 else reversed(list(ARMS))):
            path = bit_path(directory, p["id"], arm)
            rec = _read(path)
            if rec is None:
                rec = write_bit(client, p, arm)
                _write(path, rec)
            out[(p["id"], arm)] = rec
    return [out[(p["id"], a)] for p in bits for a in ARMS]


# --- judging -----------------------------------------------------------------------------------------------------

def make_judge_client():
    """The safety filter's client settings (Haiku 5.5, thinking off, its timing and retries); only the prompt differs."""
    timeout, deadline = generate.FILTER_TIMING
    provider, model, thinking = judge_model()
    return llm.from_env(timeout, role="safety_check", provider=provider, model=model, retries=generate.FILTER_RETRIES,
                        deadline_s=deadline, thinking=thinking)


def judge_user(lines: list[str]) -> str:
    return "\n".join(f"{i}. {plain(x)}" for i, x in enumerate(lines, 1))


def parse_verdict(reply: str) -> bool:
    v = json.loads(llm.strip_fence(reply))["connected"]
    if not isinstance(v, bool):
        raise ValueError("connected is not a boolean")
    return v


def verdict_path(directory: pathlib.Path, h: str) -> pathlib.Path:
    return directory / "judge_cache" / judge_digest() / f"{h}.json"


def judge_thread(client, lines: list[str], directory: pathlib.Path) -> tuple[bool | None, bool]:
    """(verdict or None when the judge could not answer, whether a call was made). Cached by the text's hash; an
    outage is not cached, it is not the judge's answer."""
    path = verdict_path(directory, thread_hash(lines))
    cached = _read(path)
    if cached is not None:
        return cached["connected"], False
    try:
        verdict = parse_verdict(client.chat(JUDGE_SYSTEM, judge_user(lines), json_reply=JUDGE_SCHEMA, validate=parse_verdict))
    except (llm.LLMError, ValueError, KeyError, TypeError, AttributeError):
        return None, True
    _write(path, {"connected": verdict})
    return verdict, True


def judge_all(client, records: list[dict], directory: pathlib.Path) -> dict[str, bool | None]:
    verdicts: dict[str, bool | None] = {}
    for rec in records:
        for lines in rec["threads"]:
            h = thread_hash(lines)
            if h not in verdicts:
                verdicts[h], _ = judge_thread(client, lines, directory)
    return verdicts


# --- the spot-check ----------------------------------------------------------------------------------------------

def review_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "review.json"


def connected_threads(records: list[dict], verdicts: dict) -> dict[str, list[str]]:
    """hash -> lines, for the threads the judge called connected, in hash order."""
    found = {thread_hash(t): t for r in records for t in r["threads"] if verdicts.get(thread_hash(t)) is True}
    return dict(sorted(found.items()))


def review_picks(found: dict[str, list[str]]) -> list[str]:
    """The threads to check: REVIEW_N of the judged-connected ones, a fixed random draw (so a rerun asks the same ones),
    pooled over both arms and shown with no arm."""
    hashes = list(found)
    return random.Random(REVIEW_SEED).sample(hashes, min(REVIEW_N, len(hashes)))


def read_answers(directory: pathlib.Path) -> dict[str, str]:
    return (_read(review_path(directory)) or {}).get("answers", {})


def review(directory: pathlib.Path, records: list[dict], verdicts: dict, read=input, out=print) -> int:
    """Asks the user y/n for each pick not answered yet, saving after every answer. Returns how many are answered."""
    picks = review_picks(connected_threads(records, verdicts))
    found = connected_threads(records, verdicts)
    answers = read_answers(directory)
    for n, h in enumerate(picks, 1):
        if h in answers:
            continue
        out(f"\nThread {n} of {len(picks)}\n" + judge_user(found[h]))
        while True:
            a = read("Is this ONE bit (one premise, each line builds on the one before)? [y/n/q] ").strip().lower()
            if a in ("y", "n", "q"):
                break
        if a == "q":
            break
        answers[h] = a
        _write(review_path(directory), {"answers": answers})
    return sum(h in answers for h in picks)


def spot_check(records: list[dict], verdicts: dict, answers: dict[str, str]) -> dict:
    """Numbers only. `void` is the fixed rule: VOID_AT_NO "no" answers or more. `complete` is whether every pick has an
    answer; an unfinished check cannot clear the judge, but it can void it early."""
    picks = review_picks(connected_threads(records, verdicts))
    got = [answers[h] for h in picks if h in answers]
    no = got.count("n")
    return {"asked": len(picks), "answered": len(got), "yes": got.count("y"), "no": no,
            "agreement": got.count("y") / len(got) if got else None, "void_at_no": VOID_AT_NO,
            "void": no >= VOID_AT_NO, "complete": len(got) == len(picks) and bool(picks)}


# --- scoring -----------------------------------------------------------------------------------------------------

def per_prompt(records: list[dict], verdicts: dict) -> dict[str, dict[str, dict]]:
    """prompt id -> arm -> {delivered, judged, connected, rate}. The rate is over the threads the judge answered, None
    when there are none (that prompt then has no rate for the arm and drops out of the pairing)."""
    out: dict[str, dict[str, dict]] = {}
    for r in records:
        judged = [verdicts.get(thread_hash(t)) for t in r["threads"]]
        judged = [v for v in judged if v is not None]
        out.setdefault(r["prompt"], {})[r["arm"]] = {
            "delivered": len(r["threads"]), "judged": len(judged), "connected": sum(judged),
            "rate": sum(judged) / len(judged) if judged else None}
    return out


def paired_diffs(rates: dict[str, dict[str, dict]]) -> tuple[list[str], np.ndarray]:
    """B minus A for every prompt where BOTH arms have a rate, matched by prompt id, in id order."""
    ids = sorted(p for p, arms in rates.items()
                 if arms.get("A", {}).get("rate") is not None and arms.get("B", {}).get("rate") is not None)
    return ids, np.array([rates[p]["B"]["rate"] - rates[p]["A"]["rate"] for p in ids], dtype=float)


def bar(diffs: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Mean paired difference above 0 with the paired bootstrap 95% CI's lower bound above 0 (resampling prompts)."""
    if len(diffs) < 2:
        return {"n_paired": int(len(diffs)), "mean_diff": None, "ci95": None, "lower_above_zero": False, "passes": False}
    mean = float(diffs.mean())
    ci = cg.boot_ci(diffs, cg.boot_idx(len(diffs), n_boot, seed))
    return {"n_paired": int(len(diffs)), "mean_diff": mean, "ci95": ci, "lower_above_zero": ci[0] > 0,
            "passes": mean > 0 and ci[0] > 0}


def arm_stats(records: list[dict], arm: str, rates: dict) -> dict:
    mine = [r for r in records if r["arm"] == arm]
    reasons: dict[str, int] = {}
    for r in mine:
        for x in r["lost"]:
            reasons[x] = reasons.get(x, 0) + 1
    walls = [r["wall_s"] for r in mine]
    judged = sum(v[arm]["judged"] for v in rates.values() if arm in v)
    return {"prompt": ARMS[arm], "bits": len(mine), "threads_delivered": sum(len(r["threads"]) for r in mine),
            "threads_lost": sum(len(r["lost"]) for r in mine), "lost_by_reason": dict(sorted(reasons.items())),
            "bits_with_no_thread": sum(not r["threads"] for r in mine), "bad_replies": sum(r["bad_replies"] for r in mine),
            "threads_judged": judged, "threads_connected": sum(v[arm]["connected"] for v in rates.values() if arm in v),
            "bit_p50_s": cg.percentile(walls, 50), "bit_p95_s": cg.percentile(walls, 95)}


def decide(b: dict, check: dict) -> str:
    """`keep_current`: the bar failed. `adopt`: the bar passed and the spot-check is complete and did not void it.
    `inconclusive_judge`: the spot-check voided the judge, whatever the CI says. `pending_review`: the bar passed and
    the check is not finished."""
    if check["void"]:
        return "inconclusive_judge"
    if not b["passes"]:
        return "keep_current"
    return "adopt" if check["complete"] else "pending_review"


def summarize(records: list[dict], verdicts: dict, answers: dict[str, str], n_boot: int = N_BOOT) -> dict:
    rates = per_prompt(records, verdicts)
    ids, diffs = paired_diffs(rates)
    b = bar(diffs, n_boot)
    check = spot_check(records, verdicts, answers)
    return {"n_prompts": len(rates), "arms": {a: arm_stats(records, a, rates) for a in ARMS},
            "per_prompt": {p: {a: rates[p][a] for a in ARMS if a in rates[p]} for p in sorted(rates)},
            "paired_prompts": ids, "bar": b, "spot_check": check,
            "judge_failures": sum(v is None for v in verdicts.values()), "verdict": decide(b, check)}


# --- power -------------------------------------------------------------------------------------------------------

def simulate_power(delta: float, p_a: float = 0.3, n: int = N_PROMPTS, threads: int = 3, spread: float = 4.0,
                   reps: int = 400, n_boot: int = 1000, seed: int = BOOT_SEED) -> float:
    """How often the bar passes if B's true connected rate is `delta` above A's. Made up, not measured: each prompt has
    its own rate (Beta around `p_a`, concentration `spread`, shared by both arms and shifted by `delta` for B), each arm
    delivers `threads` threads per prompt, each thread connected with the prompt's rate. A prompt effect shared by the
    arms makes the pairing help; a bigger `spread` means a weaker one."""
    rng = np.random.default_rng(seed)
    idx = cg.boot_idx(n, n_boot, seed)
    hit = 0
    for _ in range(reps):
        base = rng.beta(p_a * spread, (1 - p_a) * spread, n)
        a = rng.binomial(threads, base) / threads
        b = rng.binomial(threads, np.clip(base + delta, 0, 1)) / threads
        d = b - a
        hit += d.mean() > 0 and cg.boot_ci(d, idx)[0] > 0
    return hit / reps


# --- running -----------------------------------------------------------------------------------------------------

def gather(directory: pathlib.Path, bits: list[dict]) -> list[dict] | None:
    """The finished bits in the run folder, or None while any is missing."""
    recs = [_read(bit_path(directory, p["id"], a)) for p in bits for a in ARMS]
    return None if any(r is None for r in recs) else recs


def run(n_prompts: int = N_PROMPTS, seed: str = SEED, digest_: str = "", judge_digest_: str = "") -> dict:
    assert digest_ == digest(n_prompts, seed), "the prompts, timing or request changed since this call was built"
    assert judge_digest_ == judge_digest(), "the judge's prompt or model changed since this call was built"
    client = generate.writer_client()
    bits, directory = prompts(n_prompts, seed), run_dir(n_prompts, seed)
    if gather(directory, bits) is None and not getattr(client, "local", False):
        raise SystemExit("Exp 27 writes with the local writer: set JOKE_WRITER_PROVIDER=ollama")
    records = generate_bits(client, bits, directory)
    verdicts = judge_all(make_judge_client(), records, directory)
    return summarize(records, verdicts, read_answers(directory))


def score(n_prompts: int = N_PROMPTS, seed: str = SEED) -> dict:
    """The numbers again from the run folder, with the user's answers, and no model call. Raises when a bit or a
    verdict is missing: finish `run` first."""
    bits, directory = prompts(n_prompts, seed), run_dir(n_prompts, seed)
    records = gather(directory, bits)
    if records is None:
        raise SystemExit("not all bits are written yet: run `python -m sim.thread_coherence run` first")
    verdicts = {}
    for r in records:
        for t in r["threads"]:
            c = _read(verdict_path(directory, thread_hash(t)))
            verdicts[thread_hash(t)] = None if c is None else c["connected"]
    return summarize(records, verdicts, read_answers(directory))


def stored() -> dict | None:
    return cache.load(NAME, params())


def table(res: dict) -> str:
    b, c = res["bar"], res["spot_check"]
    ci = "n/a" if b["ci95"] is None else f"{b['ci95'][0]:+.3f} to {b['ci95'][1]:+.3f}"
    mean = "n/a" if b["mean_diff"] is None else f"{b['mean_diff']:+.3f}"
    rows = ["| Check | Result | Bar |", "|---|---|---|",
            f"| Mean paired difference B - A (over {b['n_paired']} prompts) | {mean}, CI {ci} | above 0, CI lower bound above 0 |",
            f"| Spot-check: you said no | {c['no']} of {c['answered']} answered ({c['asked']} asked) | fewer than {c['void_at_no']} |",
            f"| Verdict | {res['verdict']} | |"]
    for a, s in res["arms"].items():
        rows.append(f"| Arm {a} ({s['prompt']}): delivered / lost / connected | {s['threads_delivered']} / {s['threads_lost']}"
                    f" / {s['threads_connected']} of {s['threads_judged']} judged | beside the bar |")
        rows.append(f"| Arm {a}: bad replies, bit wall p50 / p95 | {s['bad_replies']}, {cg.fmt(s['bit_p50_s'])} s / "
                    f"{cg.fmt(s['bit_p95_s'])} s | beside the bar |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("run", "review", "score"))
    ap.add_argument("--rerun", action="store_true", help="run: recompute from the run folder even if a record exists")
    a = ap.parse_args()
    os.environ.setdefault("JOKE_WRITER_PROVIDER", "ollama")
    if a.command == "run":
        res = cache.run_or_load(NAME, run, rerun=a.rerun, **params())
    elif a.command == "review":
        bits, directory = prompts(), run_dir()
        records = gather(directory, bits)
        if records is None:
            raise SystemExit("run `python -m sim.thread_coherence run` first")
        verdicts = {thread_hash(t): (_read(verdict_path(directory, thread_hash(t))) or {}).get("connected")
                    for r in records for t in r["threads"]}
        done = review(directory, records, verdicts)
        print(f"{done} of {len(review_picks(connected_threads(records, verdicts)))} answered; now `score`.")
        return
    else:
        res = score()
        cache.save(NAME, params(), res)
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
