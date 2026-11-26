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
paired bootstrap 95% CI's lower bound is above 0. Spot-check, required whatever the bar says: you answer "is this one
bit?" for 10 threads, 5 the judge called connected and 5 it called not connected, spread over both arms (the other
group fills in when one is short); you are not told which is which. If you and the judge disagree on 4 or more of the 10
(40%), the judge is not trusted, and Exp 27 is inconclusive whatever the CI says. So is any run with fewer than 10
threads to check. Reported beside it, with no bar: threads delivered and lost per arm (by reason, a failed re-ask named),
`bad_replies`, bit wall time p50/p95 (also split by whether a bad shape was re-asked in the bit), and the writer's GPU
placement after the run.

Code changes elsewhere (the generator, the client) are NOT in the digest: rerun with `--rerun` after one.
"""
import argparse
import hashlib
import json
import math
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
from sim.blind import PLAIN, plain  # noqa: E402
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
VOID_DISAGREE = 0.4  # you and the judge disagree on at least this share of the threads checked: the judge is not trusted

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
    shown = judge_user(["a \u2019b\u2019 \u2014 c", "d"])  # how the lines are rendered, plain punctuation included
    return hashlib.sha256("\x00".join([JUDGE_SYSTEM, json.dumps(JUDGE_SCHEMA, sort_keys=True), shown,
                                       repr(sorted(PLAIN.items())), *map(str, judge_model())]).encode()).hexdigest()[:8]


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
    # `told` is what a later bit of a show sends: the premises of the threads played so far
    gen = generate.ThreadGenerator(client, prompt_id=ARMS[arm], told=prompt["told"])
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
    if todo:  # parallel calls take more KV cache: read where Ollama kept the writer, as Exp 26 does after its run
        share, context, others = generate.placement(client, http) if http else generate.placement(client)
        _write(placement_path(directory), {"gpu_share": share, "context_length": context, "others_loaded": others})
    return [out[(p["id"], a)] for p in bits for a in ARMS]


def placement_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "placement_after_run.json"


def writer_on_gpu(directory: pathlib.Path) -> bool | None:
    """Whether the writer was 100% on the GPU at the context the show uses after the run; None when not recorded."""
    p = _read(placement_path(directory))
    if p is None:
        return None
    return p["gpu_share"] is not None and p["gpu_share"] >= 1.0 and p["context_length"] in (None, generate.WRITER_CONTEXT)


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


def draw(records: list[dict], verdicts: dict) -> dict:
    """The threads to check, a fixed seeded draw (a rerun asks the same ones). REVIEW_N threads: half the judge called
    connected, half not connected, each half spread over the two arms (round robin, so 5 is 3 and 2), the other half
    filling in when one is short. Threads the judge could not answer are not drawn. Returns the picks in a shuffled
    order (so the order does not say which half a thread is from) and what was drawn, for the record."""
    rng = random.Random(REVIEW_SEED)
    judged: dict[str, tuple[str, bool]] = {}
    for r in records:
        for t in r["threads"]:
            h = thread_hash(t)
            if verdicts.get(h) is not None and h not in judged:
                judged[h] = (r["arm"], verdicts[h])
    pools = {(arm, v): sorted(h for h, (a, x) in judged.items() if (a, x) == (arm, v)) for arm in ARMS for v in (True, False)}
    for pool in pools.values():
        rng.shuffle(pool)

    def take(v: bool, k: int) -> list[str]:
        got: list[str] = []
        while len(got) < k and any(pools[(a, v)] for a in ARMS):
            for a in ARMS:
                if len(got) < k and pools[(a, v)]:
                    got.append(pools[(a, v)].pop())
        return got
    half = REVIEW_N // 2
    yes, no = take(True, half), take(False, half)
    yes += take(True, REVIEW_N - len(yes) - len(no))  # a half that came up short is filled from the other
    no += take(False, REVIEW_N - len(yes) - len(no))
    picks = yes + no
    rng.shuffle(picks)
    return {"picks": picks, "asked": len(picks), "connected": sum(judged[h][1] for h in picks),
            "not_connected": sum(not judged[h][1] for h in picks),
            "arm_A": sum(judged[h][0] == "A" for h in picks), "arm_B": sum(judged[h][0] == "B" for h in picks)}


def threads_by_hash(records: list[dict]) -> dict[str, list[str]]:
    return {thread_hash(t): t for r in records for t in r["threads"]}


def read_answers(directory: pathlib.Path) -> dict[str, str]:
    return (_read(review_path(directory)) or {}).get("answers", {})


def review(directory: pathlib.Path, records: list[dict], verdicts: dict, read=input, out=print) -> int:
    """Asks the user y/n for each pick not answered yet, saving after every answer (and what was drawn, once). The
    user sees the lines only: not the judge's verdict, the arm or the half. Returns how many picks are answered."""
    d = draw(records, verdicts)
    text = threads_by_hash(records)
    answers = read_answers(directory)
    _write(review_path(directory), {"answers": answers, "drawn": {k: v for k, v in d.items() if k != "picks"}})
    for n, h in enumerate(d["picks"], 1):
        if h in answers:
            continue
        out(f"\nThread {n} of {len(d['picks'])}\n" + judge_user(text[h]))
        while True:
            a = read("Is this ONE bit (one premise, each line builds on the one before)? [y/n/q] ").strip().lower()
            if a in ("y", "n", "q"):
                break
        if a == "q":
            break
        answers[h] = a
        _write(review_path(directory), {"answers": answers, "drawn": {k: v for k, v in d.items() if k != "picks"}})
    return sum(h in answers for h in d["picks"])


def spot_check(records: list[dict], verdicts: dict, answers: dict[str, str]) -> dict:
    """Numbers only. A disagreement is a thread where your answer differs from the judge's verdict. `void` is the fixed
    rule: disagreement on VOID_DISAGREE of the threads to check or more, counted against the number asked so an early
    run of disagreements already voids it. `short` means fewer than REVIEW_N threads could be drawn: the check cannot
    clear the judge. `complete` means every pick is answered."""
    d = draw(records, verdicts)
    got = {h: answers[h] for h in d["picks"] if h in answers}
    disagree = sum((a == "y") != verdicts[h] for h, a in got.items())
    return {"asked": d["asked"], "answered": len(got), "drawn_connected": d["connected"],
            "drawn_not_connected": d["not_connected"], "drawn_arm_A": d["arm_A"], "drawn_arm_B": d["arm_B"],
            "yes": sum(a == "y" for a in got.values()), "no": sum(a == "n" for a in got.values()),
            "disagree": disagree, "disagreement": disagree / len(got) if got else None,
            "void_at_share": VOID_DISAGREE, "short": d["asked"] < REVIEW_N,
            "void": d["asked"] > 0 and disagree >= VOID_DISAGREE * d["asked"] - 1e-9,
            "complete": len(got) == d["asked"] and d["asked"] > 0}


def p_void(error: float, n: int = REVIEW_N) -> float:
    """The chance the check voids a judge that is wrong on each thread independently with probability `error`: the
    chance of at least ceil(0.4 n) disagreements out of n. Exact (binomial). The independence is an assumption."""
    need = math.ceil(VOID_DISAGREE * n - 1e-9)
    return sum(math.comb(n, k) * error ** k * (1 - error) ** (n - k) for k in range(need, n + 1))


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
    re = [r for r in mine if r["bad_replies"] > 0]  # a bad reply is re-asked unless the budget is gone: close enough
    rest = [r for r in mine if r["bad_replies"] == 0]
    judged = sum(v[arm]["judged"] for v in rates.values() if arm in v)
    return {"prompt": ARMS[arm], "bits": len(mine), "threads_delivered": sum(len(r["threads"]) for r in mine),
            "threads_lost": sum(len(r["lost"]) for r in mine), "lost_by_reason": dict(sorted(reasons.items())),
            "bits_with_no_thread": sum(not r["threads"] for r in mine), "bad_replies": sum(r["bad_replies"] for r in mine),
            "threads_judged": judged, "threads_connected": sum(v[arm]["connected"] for v in rates.values() if arm in v),
            "bit_p50_s": cg.percentile(walls, 50), "bit_p95_s": cg.percentile(walls, 95),
            "bits_reasked": len(re), "bit_p50_s_reasked": cg.percentile([r["wall_s"] for r in re], 50),
            "bit_p95_s_reasked": cg.percentile([r["wall_s"] for r in re], 95),
            "bit_p50_s_no_reask": cg.percentile([r["wall_s"] for r in rest], 50),
            "bit_p95_s_no_reask": cg.percentile([r["wall_s"] for r in rest], 95)}


def decide(b: dict, check: dict) -> str:
    """The spot-check is required whatever the bar says. `inconclusive_judge`: it voided the judge, or fewer than
    REVIEW_N threads could be drawn, whatever the CI says. `pending_review`: not all picks are answered yet. `adopt`:
    the bar passed, and `keep_current`: it did not, each with a complete check that did not void the judge."""
    if check["void"] or check["short"]:
        return "inconclusive_judge"
    if not check["complete"]:
        return "pending_review"
    return "adopt" if b["passes"] else "keep_current"


def summarize(records: list[dict], verdicts: dict, answers: dict[str, str], n_boot: int = N_BOOT,
              on_gpu: bool | None = None) -> dict:
    rates = per_prompt(records, verdicts)
    ids, diffs = paired_diffs(rates)
    b = bar(diffs, n_boot)
    check = spot_check(records, verdicts, answers)
    return {"n_prompts": len(rates), "arms": {a: arm_stats(records, a, rates) for a in ARMS},
            "per_prompt": {p: {a: rates[p][a] for a in ARMS if a in rates[p]} for p in sorted(rates)},
            "paired_prompts": ids, "bar": b, "spot_check": check, "writer_on_gpu_after_run": on_gpu,
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
    return summarize(records, verdicts, read_answers(directory), on_gpu=writer_on_gpu(directory))


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
    return summarize(records, verdicts, read_answers(directory), on_gpu=writer_on_gpu(directory))


def stored() -> dict | None:
    return cache.load(NAME, params())


def table(res: dict) -> str:
    b, c = res["bar"], res["spot_check"]
    ci = "n/a" if b["ci95"] is None else f"{b['ci95'][0]:+.3f} to {b['ci95'][1]:+.3f}"
    mean = "n/a" if b["mean_diff"] is None else f"{b['mean_diff']:+.3f}"
    rows = ["| Check | Result | Bar |", "|---|---|---|",
            f"| Mean paired difference B - A (over {b['n_paired']} prompts) | {mean}, CI {ci} | above 0, CI lower bound above 0 |",
            f"| Spot-check: you and the judge disagree on | {c['disagree']} of {c['answered']} answered ({c['asked']} asked: "
            f"{c['drawn_connected']} judged connected, {c['drawn_not_connected']} not; {c['drawn_arm_A']} from A, {c['drawn_arm_B']} from B) "
            f"| under {c['void_at_share']:.0%}, and all {REVIEW_N} asked |",
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
        print(f"{done} of {draw(records, verdicts)['asked']} answered; now `score`.")
        return
    else:
        res = score()
        cache.save(NAME, params(), res)
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
