"""Exp 27 (D-111): does the connected prompt (gen-v5) write threads whose lines connect, more often than the current
prompt (gen-v4)?

Usage (Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`, `ANTHROPIC_API_KEY` set for the judge):
    python -m sim.thread_coherence run [--env-file backend/.env]  # generate both arms, judge every thread -> reports/metrics/thread_coherence.json
    python -m sim.thread_coherence review    # you check 10 threads the judge called connected (y/n), arm hidden
    python -m sim.thread_coherence score     # recompute the numbers with your answers in (no model calls)
    python -m sim.thread_coherence rescore --judge v2   # D-114: re-judge the stored threads with judge v2, only if Exp 30 trusted it

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

Extension to 100 prompts (D-111 and D-114 addenda, the user's call 2026-11-26; fixed before any new data). The 20-prompt
result above is kept as it is. Judge v2 re-scored it: mean B - A +0.133, CI [0.0, 0.25], so the bar failed at exactly 0.
Three looks already exist (v1 over 15, v2 over 15, v2 over 20 after five bits were rewritten) and the 20 are what
triggered the extension, so a pooled test would be biased toward a pass (a simulated 17% false alarm given a first 20 as
good as this one). So ids `c20` to `c99` are added (`prompts(100)[:20] == prompts(20)`, a test), no new user labels, and
**the deciding bar uses only the 80 fresh prompts c20 to c99**:
    python -m sim.thread_coherence run --n 100 --env-file backend/.env
    python -m sim.thread_coherence rescore --judge v2 --n 100 --env-file backend/.env
`run --n 100` writes only the bits that are missing (c00 to c19 are read back, never rewritten; the run folder is keyed
on the first 20 prompts, not on n) and judges nothing: v2 decides, v1 is not asked for the new threads and its spot-check
is superseded (D-114), so `review` refuses at every n and `score` prints that it is superseded. It also rewrites, once and
automatically, every fresh bit that has no thread and lost them all to transport timeouts, in both arms alike
(`bits_rewritten`). `--n` is 20 or 100. `rescore --n 100` judges all 100 prompts' threads with v2 (c00 to c19 are cached)
and stores `thread_coherence_rescore_n100.json`.
Bar: judge v2's paired mean B - A over the 80 fresh prompts (a prompt where an arm has no judged thread drops out;
`n_paired` is reported) is above 0 and the paired bootstrap 95% CI's lower bound is above 0. The pooled 100 is reported
beside it (`pooled_bar`) and never decides. Adopting gen-v5 also needs D-114's conditions (`judge_calibration.adoptable`,
which reads the fresh-80 bar of the n=100 record).
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
N_PROMPTS = 20  # Exp 27's own prompts; the extension (N_EXT) keeps these as its first 20
N_EXT = 100
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

# Judge v2 (D-114, Exp 30): written from the 10 threads the user checked in Exp 27, so those 10 are its training set and
# cannot test it. Same reply schema; only the criterion differs. v1 above must stay byte-identical (Exp 27's verdicts
# and `judge_digest` depend on it; a test pins its hash).
JUDGE_SYSTEM_V2 = """You judge stand-up material. You get five lines in the order they are performed, written as one bit.

Decide whether a listener would hear ONE bit or SEPARATE jokes.

It is ONE bit when a listener can follow one situation or one line of thought from the first line to the last. At least one of these holds: the same scene or the same people carry through the lines; each line follows from the one before it (because, so, but, which means); or the last line returns to the opening (a callback).

It is SEPARATE jokes when the lines jump between unrelated situations, even if they share a theme. A shared theme alone is not a bit: a robot joking about its battery, then its overheating, then its anxiety is three jokes on one subject.

Judge the structure only, never whether it is funny.

Reply with JSON only: {"connected": true} for one bit, {"connected": false} for separate jokes."""
JUDGES = {"v1": JUDGE_SYSTEM, "v2": JUDGE_SYSTEM_V2}
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
    return generation_digest(prompts(n, seed), seed, n)


def gen_digest(n: int = N_PROMPTS, seed: str = SEED) -> str:
    """Keys the run folder and the records. Beyond Exp 27's 20 prompts it is the 20-prompt digest, so the extension
    reads and adds to the folder (and keeps the digests) Exp 27's record was made under; what the first 20 bits are
    made of is the same either way. Below 20 it is `digest(n)`."""
    return digest(min(n, N_PROMPTS), seed)


def prompt_hash(prompt: dict) -> str:
    return hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()[:16]


def prompts_digest(n: int, seed: str = SEED) -> str:
    """What the prompts c00..c(n-1) are. The folder key covers only the first 20, so the extension's own prompts are
    pinned here (in the records' params) and in each bit's `prompt_hash`."""
    return hashlib.sha256(json.dumps(prompts(n, seed), sort_keys=True).encode()).hexdigest()[:16]


def metric_name(base: str, n: int) -> str:
    """The 20-prompt records keep their names; another n gets its own file, so the extension never overwrites them."""
    return base if n == N_PROMPTS else f"{base}_n{n}"


def generation_digest(bits: list[dict], seed: str, n: int) -> str:
    """What the generated bits are made of: both prompts, the reply shapes, timing, model and the prompts' contents. A
    change starts a new run folder. The judge has its own digest (`judge_digest`), as its verdicts can be redone alone.
    Exp 30 calls it with its own prompts."""
    parts = [json.dumps({a: [i, generate.PROMPTS[i]] for a, i in ARMS.items()}, sort_keys=True), generate.TEENS_RULE,
             json.dumps(generate.MODES, sort_keys=True), json.dumps(generate.REPLY_SCHEMA, sort_keys=True),
             json.dumps(generate.COUNTED_REPLY_SCHEMA, sort_keys=True),
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.RETRIES,
                  generate.WRITER_CONTEXT, generate.HANG_GRACE_S, llm.MIN_ATTEMPT_S)),
             writer_model(), json.dumps(llm.OLLAMA_NO_THINK, sort_keys=True), "json_schema=True", "reask_bad_shape_once",
             json.dumps(bits, sort_keys=True), seed, str(n)]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8]


def judge_digest(version: str = "v1") -> str:
    """Per judge: the criterion's text is in the hash, so v2's verdicts live in their own cache folder and cannot be
    read back as v1's."""
    shown = judge_user(["a \u2019b\u2019 \u2014 c", "d"])  # how the lines are rendered, plain punctuation included
    return hashlib.sha256("\x00".join([JUDGES[version], json.dumps(JUDGE_SCHEMA, sort_keys=True), shown,
                                       repr(sorted(PLAIN.items())), *map(str, judge_model())]).encode()).hexdigest()[:8]


def run_dir(n: int = N_PROMPTS, seed: str = SEED, root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or DATA) / f"run-{gen_digest(n, seed)}"


def params(n: int = N_PROMPTS, seed: str = SEED) -> dict:
    return {"n_prompts": n, "seed": seed, "digest_": gen_digest(n, seed), "judge_digest_": judge_digest()}


def ext_params(n: int = N_EXT, seed: str = SEED) -> dict:
    """The extension's writing step has no judge in it, so no judge digest."""
    return {"n_prompts": n, "seed": seed, "digest_": gen_digest(n, seed), "prompts_digest_": prompts_digest(n, seed)}


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
            "lost": lost, "failure": failure, "bad_replies": client.bad_replies - bad0, "prompt_hash": prompt_hash(prompt)}


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


def verdict_path(directory: pathlib.Path, h: str, version: str = "v1") -> pathlib.Path:
    return directory / "judge_cache" / judge_digest(version) / f"{h}.json"


def judge_thread(client, lines: list[str], directory: pathlib.Path, version: str = "v1") -> tuple[bool | None, bool]:
    """(verdict or None when the judge could not answer, whether a call was made). Cached by the text's hash; an
    outage is not cached, it is not the judge's answer."""
    path = verdict_path(directory, thread_hash(lines), version)
    cached = _read(path)
    if cached is not None:
        return cached["connected"], False
    try:
        verdict = parse_verdict(client.chat(JUDGES[version], judge_user(lines), json_reply=JUDGE_SCHEMA, validate=parse_verdict))
    except (llm.LLMError, ValueError, KeyError, TypeError, AttributeError):
        return None, True
    _write(path, {"connected": verdict})
    return verdict, True


def judge_all(client, records: list[dict], directory: pathlib.Path, version: str = "v1") -> dict[str, bool | None]:
    verdicts: dict[str, bool | None] = {}
    for rec in records:
        for lines in rec["threads"]:
            h = thread_hash(lines)
            if h not in verdicts:
                verdicts[h], _ = judge_thread(client, lines, directory, version)
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
            "judge_failures": sum(v is None for v in verdicts.values()), "verdict": decide(b, check),
            "superseded_by": "v2 (D-114)"}


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
    if any(r is None for r in recs):
        return None
    for (p, a), r in zip([(p, a) for p in bits for a in ARMS], recs):
        h = r.get("prompt_hash")
        # Exp 27's own 20 bits were written before the hash existed (their prompts are pinned by a test); any later one has it
        if (h is None and int(p["id"][1:]) >= N_PROMPTS) or (h is not None and h != prompt_hash(p)):
            raise SystemExit(f"{p['id']}-{a} in {directory} was not written for this prompt (hash {h}, now {prompt_hash(p)}): "
                             "the prompts changed since it was written")
    return recs


def run(n_prompts: int = N_PROMPTS, seed: str = SEED, digest_: str = "", judge_digest_: str = "") -> dict:
    assert digest_ == gen_digest(n_prompts, seed), "the prompts, timing or request changed since this call was built"
    assert judge_digest_ == judge_digest(), "the judge's prompt or model changed since this call was built"
    client = generate.writer_client()
    bits, directory = prompts(n_prompts, seed), run_dir(n_prompts, seed)
    if gather(directory, bits) is None and not getattr(client, "local", False):
        raise SystemExit("Exp 27 writes with the local writer: set JOKE_WRITER_PROVIDER=ollama")
    records = generate_bits(client, bits, directory)
    verdicts = judge_all(make_judge_client(), records, directory)
    return summarize(records, verdicts, read_answers(directory), on_gpu=writer_on_gpu(directory))


def timed_out_only(rec: dict) -> bool:
    """A bit with no thread whose every loss was a transport timeout (the slow-machine failure, nothing to do with the text)."""
    return not rec["threads"] and bool(rec["lost"]) and all("transport" in x and "Timeout" in x for x in rec["lost"])


def run_extension(n_prompts: int = N_EXT, seed: str = SEED, digest_: str = "", prompts_digest_: str | None = None) -> dict:
    """Writes the missing bits of prompts c20 onward into Exp 27's run folder and returns timing and loss numbers: no
    judging (the rescore does it with v2) and no review. Exp 27's own 20 bits must already be there: with another
    digest (a changed writer setting, another checkout) the folder would be empty and the first 20 would be written
    again, so this stops instead of quietly mixing two runs. **Timeouts (mandatory, automatic):** every fresh bit (c20
    on) with no thread whose losses are all transport timeouts is written once more, in both arms alike, whatever else
    is known; the first attempt is kept under `timed_out/`, and a rewritten bit is flagged so it is never rewritten
    twice. The record names them in `bits_rewritten`."""
    assert digest_ == gen_digest(n_prompts, seed), "the prompts, timing or request changed since this call was built"
    assert prompts_digest_ in (None, prompts_digest(n_prompts, seed)), "the extension's prompts changed since this call was built"
    assert n_prompts > N_PROMPTS, "the extension adds prompts to Exp 27's 20"
    client = generate.writer_client()
    bits, directory = prompts(n_prompts, seed), run_dir(n_prompts, seed)
    if gather(directory, bits[:N_PROMPTS]) is None:
        raise SystemExit(f"Exp 27's {N_PROMPTS} prompts' bits are not all in {directory}: run from the checkout that holds "
                         "data/thread_coherence, with the writer settings Exp 27 ran with")
    if gather(directory, bits) is None and not getattr(client, "local", False):
        raise SystemExit("Exp 27 writes with the local writer: set JOKE_WRITER_PROVIDER=ollama")
    records = generate_bits(client, bits, directory)
    fresh = {p["id"]: p for p in bits[N_PROMPTS:]}
    redo = [i for i, r in enumerate(records) if r["prompt"] in fresh and not r.get("rewritten") and timed_out_only(r)]
    if redo:
        generate.warm_up(generate.ThreadGenerator(client))
    for i in redo:
        old = records[i]
        _write(directory / "timed_out" / f"{old['prompt']}-{old['arm']}.json", old)
        new = {**write_bit(client, fresh[old["prompt"]], old["arm"]), "rewritten": True}
        _write(bit_path(directory, old["prompt"], old["arm"]), new)
        records[i] = new
    arms = {}
    for a in ARMS:
        s = arm_stats(records, a, {})
        arms[a] = {k: v for k, v in s.items() if k not in ("threads_judged", "threads_connected")}
    return {"n_prompts": n_prompts, "reused_prompts": N_PROMPTS, "new_prompts": n_prompts - N_PROMPTS, "arms": arms,
            "bits_rewritten": [f"{r['prompt']}-{r['arm']}" for r in records if r.get("rewritten")],
            "writer_on_gpu_after_run": writer_on_gpu(directory)}


def score(n_prompts: int = N_PROMPTS, seed: str = SEED) -> dict:
    """The numbers again from the run folder, with the user's answers, and no model call. Raises when a bit or a
    verdict is missing: finish `run` first."""
    if n_prompts > N_PROMPTS:
        raise SystemExit("score is Exp 27's v1 path with the spot-check, superseded by v2 (D-114): use `rescore --judge v2`")
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


# --- rescoring with a calibrated judge (D-114) -------------------------------------------------------------------

RESCORE_NAME = "thread_coherence_rescore"


def rescore_params(judge: str, n: int = N_PROMPTS, seed: str = SEED) -> dict:
    out = {"judge": judge, "n_prompts": n, "seed": seed, "digest_": gen_digest(n, seed), "judge_digest_": judge_digest(judge)}
    if n > N_PROMPTS:
        out["prompts_digest_"] = prompts_digest(n, seed)
    return out


def rescore_record(judge: str = "v2", n: int = N_PROMPTS) -> dict | None:
    return cache.load(metric_name(RESCORE_NAME, n), rescore_params(judge, n))


def rescore_allowed(judge: str) -> bool:
    """Exp 30's fixed rule: only the judge it kept, and only when its disagreement with the user on the fresh sample was
    under 40%, may re-score Exp 27. v1 is never re-scored: its verdicts are Exp 27's own record."""
    from sim import judge_calibration as jc  # lazy: judge_calibration imports this module
    res = jc.stored()
    return res is not None and judge != "v1" and res.get("rescore_allowed") is True and res.get("kept") == judge


def exp30_provenance() -> dict | None:
    """What a re-score depends on: the Exp 30 record that allowed it (its digest, verdict and agreement counts). A
    later Exp 30 rerun or relabel that changes any of them makes an earlier re-score stale."""
    from sim import judge_calibration as jc
    r = jc.stored()
    if r is None:
        return None
    j = r.get("judges", {})
    return {"calibration_digest": jc.digest(), "verdict": r.get("verdict"), "kept": r.get("kept"),
            "agree_v1": j.get("v1", {}).get("agree"), "agree_v2": j.get("v2", {}).get("agree"), "scored": r.get("scored")}


def rescore_current(res: dict, n: int = N_PROMPTS) -> bool:
    """Current for this n and the Exp 30 record as it stands now. A record without `n` is the 20-prompt one."""
    return res.get("n", N_PROMPTS) == n and res.get("exp30") is not None and res["exp30"] == exp30_provenance()


def rescore_status(judge: str = "v2", n: int = N_PROMPTS) -> tuple[str, dict | None]:
    """('none' | 'stale' | 'current', the stored re-score). Stale: made under an Exp 30 record that has changed, or
    for another n."""
    res = rescore_record(judge, n)
    if res is None:
        return "none", None
    return ("current" if rescore_current(res, n) else "stale"), res


def rescore(judge: str, n_prompts: int = N_PROMPTS, seed: str = SEED, digest_: str = "", judge_digest_: str = "",
            client=None, prompts_digest_: str | None = None) -> dict:
    """Exp 27's bar again, with the stored threads judged by `judge` (cached under that judge's digest). No new writing.
    The spot-check is not repeated: the fresh-sample agreement of Exp 30 is the judge check, because the 10 threads of
    Exp 27's spot-check are judge v2's training set. So this record holds the bar and nothing that adopts gen-v5."""
    assert digest_ == gen_digest(n_prompts, seed) and judge_digest_ == judge_digest(judge), "the prompts or the judge changed"
    assert prompts_digest_ in (None, prompts_digest(n_prompts, seed)), "the prompts changed since this call was built"
    if not rescore_allowed(judge):
        raise SystemExit(f"judge {judge} may not re-score Exp 27: Exp 30 has not kept and trusted it (D-114)")
    bits, directory = prompts(n_prompts, seed), run_dir(n_prompts, seed)
    records = gather(directory, bits)
    if records is None:
        raise SystemExit("Exp 27's bits are not all in the run folder" + (
            f": run `python -m sim.thread_coherence run --n {n_prompts}` first" if n_prompts > N_PROMPTS else ""))
    verdicts = judge_all(client or make_judge_client(), records, directory, judge)
    if n_prompts > N_PROMPTS and any(v is None for v in verdicts.values()):
        raise SystemExit("the judge could not answer some threads (an outage is not cached): run again to fill them in, "
                         "so the 100-prompt record is not made on a gap")
    rates = per_prompt(records, verdicts)
    ids, diffs = paired_diffs(rates)
    extra: dict = {"bar_over": "all"}
    if n_prompts > N_PROMPTS:
        # D-111 addendum: the first 20 triggered the extension, so only the fresh prompts decide; the pooled bar is reported
        new = {p["id"] for p in prompts(n_prompts, seed)[N_PROMPTS:]}
        keep = np.array([i in new for i in ids], dtype=bool)
        b = bar(diffs[keep])
        extra = {"bar_over": "fresh", "fresh_paired_prompts": [i for i in ids if i in new], "pooled_bar": bar(diffs)}
    else:
        b = bar(diffs)
    return {"judge": judge, "n": n_prompts, "exp30": exp30_provenance(), "n_prompts": len(rates), "paired_prompts": ids, "bar": b,
            **extra,
            "arms": {a: {"threads_judged": sum(v[a]["judged"] for v in rates.values() if a in v),
                         "threads_connected": sum(v[a]["connected"] for v in rates.values() if a in v)} for a in ARMS},
            "per_prompt": {p: {a: rates[p][a] for a in ARMS if a in rates[p]} for p in sorted(rates)},
            "judge_failures": sum(v is None for v in verdicts.values()), "bar_passes": b["passes"]}


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
    ap.add_argument("command", choices=("run", "review", "score", "rescore"))
    ap.add_argument("--n", type=int, default=N_PROMPTS, choices=(N_PROMPTS, N_EXT),
                    help=f"prompts: {N_PROMPTS} is Exp 27, {N_EXT} its extension (the fresh {N_EXT - N_PROMPTS} decide)")
    ap.add_argument("--judge", choices=tuple(JUDGES), default="v2", help="rescore: which judge re-scores")
    ap.add_argument("--rerun", action="store_true", help="run: recompute from the run folder even if a record exists")
    ap.add_argument("--env-file")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file, override=True)
    os.environ.setdefault("JOKE_WRITER_PROVIDER", "ollama")
    if a.command == "review":
        raise SystemExit("review is superseded by judge v2 (D-114): do not label these again; the extension is "
                         f"`run --n {N_EXT}` then `rescore --judge v2 --n {N_EXT}`")
    if a.n > N_PROMPTS and a.command == "score":
        raise SystemExit(f"score is Exp 27's v1 path, superseded by v2 (D-114): use `rescore --judge v2 --n {a.n}`")
    if a.command == "run" and a.n > N_PROMPTS:
        res = cache.run_or_load(metric_name(NAME, a.n), run_extension, rerun=a.rerun, **ext_params(a.n))
        print(json.dumps({k: res[k] for k in ("n_prompts", "new_prompts", "writer_on_gpu_after_run")}), flush=True)
        for arm, st in res["arms"].items():
            print(f"arm {arm} ({st['prompt']}): delivered {st['threads_delivered']}, lost {st['threads_lost']}, "
                  f"bit p50/p95 {cg.fmt(st['bit_p50_s'])}/{cg.fmt(st['bit_p95_s'])} s")
        print(f"now: python -m sim.thread_coherence rescore --judge v2 --n {a.n}")
        return
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
    elif a.command == "rescore":
        status, _ = rescore_status(a.judge, a.n)
        res = cache.run_or_load(metric_name(RESCORE_NAME, a.n), rescore, rerun=a.rerun or status == "stale",
                                **rescore_params(a.judge, a.n))
        b = res["bar"]
        print(f"judge {res['judge']}, n={a.n}: mean B - A {b['mean_diff']} CI {b['ci95']} over {b['n_paired']} prompts; bar passes: {res['bar_passes']}")
        return
    else:
        res = {**score(), "superseded_by": "v2 (D-114)"}
        cache.save(NAME, params(), res)
        print("Judge v1's verdict is superseded by v2 (D-114); the spot-check below is not asked for again.")
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
