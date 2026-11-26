"""Exp 30 (D-114): which thread judge agrees with the user more, the old one (v1) or one rewritten from what the user's
spot-check showed (v2)?

Usage (Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`, `ANTHROPIC_API_KEY` for the judges):
    python -m sim.judge_calibration run [--env-file backend/.env]   # write fresh bits, judge the drawn threads with both
    python -m sim.judge_calibration review                          # you label 20 threads: one bit or separate jokes
    python -m sim.judge_calibration score                           # agreement, paired CI, verdict (no model call)
    python -m sim.thread_coherence rescore --judge v2               # only if this says `rescore_allowed`

Why. Exp 27's judge (v1) agreed with the user on 4 of 10 threads, which is chance (D-111). v2's criterion was written
from those same 10 threads, so they are its training set and cannot test it. This run tests both judges on threads that
neither was written from.

Fresh sample. 10 new prompts (ids `f00` to `f09`, seed `exp30`, topics not in Exp 27's list), both arms (A: gen-v4,
B: gen-v5) written through the show's own path (`thread_coherence.generate_bits`: warm-up, GPU check, resumable) into
`data/thread_coherence/calib-<digest>/` (gitignored). The threads to label are drawn from what was written, and the draw
does not look at either judge's verdict, so neither judge picks its own test: 10 from arm A and 10 from arm B, a seeded
shuffle, at most 2 from one bit (one thread from each bit first), none identical to the 10 threads of Exp 27's spot-check.
You see the lines only: no verdict, no arm, shuffled; the question is neutral (Exp 27's wording was v1's own criterion).
Both judges judge the same drawn threads (Haiku 5.5, thinking off, one call each, cached per judge digest).

Noise, before any data. Agreement is counted over the 20 labelled threads; "v2 only right" is a thread where v2 matches
your label and v1 does not, and the difference is (v2 only right minus v1 only right) over 20. If the two judges were
right independently, with the stated probabilities, the rule below keeps v2 this often (exact, `p_keep_v2`) and the paired
bootstrap CI of the difference has its lower bound above 0 this often (simulated, `simulate_power`, 400 repetitions):

    v1 right / v2 right    keeps v2    CI lower bound above 0
    0.5 / 0.5 (tie)         44%         2%
    0.4 / 0.4 (tie)         44%         2%
    0.5 / 0.7               88%        26%
    0.4 / 0.7               97%        47%
    0.4 / 0.8               99%        73%

Reading. A point-estimate rule on 20 labels is a coin flip for equal judges (it picks v2 about 4 times in 10 and keeps v1
the rest, ties included) and the CI will usually include 0 unless v2 is much better. That is the user's chosen rule; the CI is
reported so nobody reads a 12-vs-9 as proof. Independence overstates the noise a little: two Haiku judges on the same
threads agree with each other, so fewer threads separate them.

Decision rule, fixed before any data (the user's, 2026-11-25). Keep v2 if its agreement with your labels is higher than
v1's on the 20 (a point estimate); if equal, keep v1. The kept judge is trusted when its disagreement with you on the 20
is under 40% (the void rule of Exp 27, `VOID_DISAGREE`: 8 or more of 20 disagreements voids it). Only a kept, trusted v2
may re-score Exp 27 (`thread_coherence rescore`, enforced in code); v1 is never re-scored, its verdicts are Exp 27's own.
Fewer than 20 threads to label is `inconclusive_sample`; missing labels or a judge failure is `pending`. Verdicts:
`keep_v1_trusted`, `keep_v1_untrusted`, `keep_v2_trusted`, `keep_v2_untrusted`, `inconclusive_sample`, `pending`.
Adopting gen-v5 is a separate step with its own conditions (D-114), not decided here.

Beside the bar, no bar: the share of threads you called one bit, for arm A and arm B (a judge-free signal, 10 labels
each, so wide), and each judge's share per arm. Generated text lives only under `data/thread_coherence/`; the record holds
counts and ids only.
"""
import argparse
import math
import os
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import generate  # noqa: E402
from sim import cache  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim import thread_coherence as tc  # noqa: E402
from sim.env import AGES, ENERGIES, VENUES  # noqa: E402

NAME = "judge_calibration"
N_PROMPTS = 10
SEED = "exp30"
PER_ARM = 10
MAX_PER_BIT = 2
DRAW_SEED = 30
BOOT_SEED = 30
N_BOOT = 10_000
VOID_DISAGREE = tc.VOID_DISAGREE
VERSIONS = ("v1", "v2")
ARMS = tc.ARMS

# Bland placeholder premises, none from Exp 27's list.
TOPICS = ("a leaky faucet", "a dentist waiting room", "a wedding toast", "a food delivery app", "a hotel breakfast buffet",
          "a standing desk", "a library fine", "a carpool", "a spelling bee", "a camping trip", "a garage sale",
          "a haircut", "a team-building retreat", "a lost phone charger", "a school pickup line", "a subscription renewal")


# --- prompts, folders ----------------------------------------------------------------------------------------------

def prompts(n: int = N_PROMPTS, seed: str = SEED) -> list[dict]:
    modes = tuple(generate.MODES)
    out = []
    for i in range(n):
        r = random.Random(f"{seed}:{i}")
        out.append({"id": f"f{i:02d}", "mode": modes[i % len(modes)],
                    "context": {"venue": r.choice(VENUES), "energy": r.choice(ENERGIES), "age": r.choice(AGES)},
                    "told": r.sample(TOPICS, i % (tc.MAX_TOLD + 1))})
    return out


def digest(n: int = N_PROMPTS, seed: str = SEED) -> str:
    return tc.generation_digest(prompts(n, seed), seed, n)


def run_dir(n: int = N_PROMPTS, seed: str = SEED, root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or tc.DATA) / f"calib-{digest(n, seed)}"


def params(n: int = N_PROMPTS, seed: str = SEED) -> dict:
    return {"n_prompts": n, "seed": seed, "digest_": digest(n, seed),
            "judge_digest_v1": tc.judge_digest("v1"), "judge_digest_v2": tc.judge_digest("v2")}


def labels_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "labels.json"


def read_answers(directory: pathlib.Path) -> dict[str, str]:
    return (tc._read(labels_path(directory)) or {}).get("answers", {})


def reviewed_before() -> set[str]:
    """Hashes of the 10 threads the user checked in Exp 27 (a local file; empty when absent)."""
    return set(tc.read_answers(tc.run_dir()))


# --- the draw ------------------------------------------------------------------------------------------------------

def draw(records: list[dict], exclude: set[str]) -> dict:
    """PER_ARM threads from each arm, seeded, independent of any verdict. Within an arm the bits are shuffled and each
    round takes one thread from every bit that has one left, for MAX_PER_BIT rounds, so the sample spreads over bits
    before it doubles up on one. A thread whose hash is in `exclude` (already reviewed) or already drawn is skipped.
    Picks come back shuffled across arms. `bit_of` and `arm_of` map a pick to its bit and arm."""
    seen = set(exclude)
    picks_by_arm: dict[str, list[str]] = {}
    bit_of: dict[str, str] = {}
    dropped = 0
    for arm in ARMS:
        rng = random.Random(f"{DRAW_SEED}:{arm}")
        bits = []
        for r in sorted((r for r in records if r["arm"] == arm), key=lambda r: r["prompt"]):
            hs = []
            for t in r["threads"]:
                h = tc.thread_hash(t)
                if h in exclude:
                    dropped += 1
                elif h not in hs:
                    hs.append(h)
                    bit_of[h] = f"{r['prompt']}-{arm}"
            rng.shuffle(hs)
            bits.append(hs)
        rng.shuffle(bits)
        got: list[str] = []
        for rnd in range(MAX_PER_BIT):
            for hs in bits:
                if len(got) < PER_ARM and rnd < len(hs) and hs[rnd] not in seen:
                    seen.add(hs[rnd])
                    got.append(hs[rnd])
        picks_by_arm[arm] = got
    picks = [h for arm in ARMS for h in picks_by_arm[arm]]
    random.Random(DRAW_SEED).shuffle(picks)
    arm_of = {h: arm for arm, hs in picks_by_arm.items() for h in hs}
    per_bit: dict[str, int] = {}
    for h in picks:
        per_bit[bit_of[h]] = per_bit.get(bit_of[h], 0) + 1
    return {"picks": picks, "asked": len(picks), "arm_A": len(picks_by_arm["A"]), "arm_B": len(picks_by_arm["B"]),
            "max_from_one_bit": max(per_bit.values(), default=0), "dropped_as_reviewed": dropped,
            "short": len(picks) < PER_ARM * len(ARMS), "bit_of": bit_of, "arm_of": arm_of}


def drawn_numbers(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("picks", "bit_of", "arm_of")}


# --- generating and judging ----------------------------------------------------------------------------------------

def gather(directory: pathlib.Path, bits: list[dict]) -> list[dict] | None:
    return tc.gather(directory, bits)


def judge_picks(client, records: list[dict], picks: list[str], directory: pathlib.Path, version: str) -> dict[str, bool | None]:
    text = tc.threads_by_hash(records)
    return {h: tc.judge_thread(client, text[h], directory, version)[0] for h in picks}


def stored_verdicts(records: list[dict], picks: list[str], directory: pathlib.Path) -> dict[str, dict[str, bool | None]]:
    out: dict[str, dict[str, bool | None]] = {v: {} for v in VERSIONS}
    for v in VERSIONS:
        for h in picks:
            c = tc._read(tc.verdict_path(directory, h, v))
            out[v][h] = None if c is None else c["connected"]
    return out


# --- the review ----------------------------------------------------------------------------------------------------

def review(directory: pathlib.Path, records: list[dict], read=input, out=print) -> int:
    """Asks y/n for each pick not answered yet, saving after every answer. The user sees the lines only (not a verdict,
    an arm or a judge), and a question that does not borrow either judge's wording. Returns how many are answered."""
    d = draw(records, reviewed_before())
    text = tc.threads_by_hash(records)
    answers = read_answers(directory)
    tc._write(labels_path(directory), {"answers": answers, "drawn": drawn_numbers(d)})
    for n, h in enumerate(d["picks"], 1):
        if h in answers:
            continue
        out(f"\nThread {n} of {len(d['picks'])}\n" + tc.judge_user(text[h]))
        while True:
            a = read("One bit, or separate jokes? [y = one bit / n = separate jokes / q = save and quit] ").strip().lower()
            if a in ("y", "n", "q"):
                break
        if a == "q":
            break
        answers[h] = a
        tc._write(labels_path(directory), {"answers": answers, "drawn": drawn_numbers(d)})
    return sum(h in answers for h in d["picks"])


# --- scoring -------------------------------------------------------------------------------------------------------

def keep(agree_v1: int, agree_v2: int) -> str:
    """The user's rule: v2 only when it agrees with the user on strictly more threads. A tie keeps v1."""
    return "v2" if agree_v2 > agree_v1 else "v1"


def trusted(disagree: int, n: int) -> bool:
    """Disagreement under VOID_DISAGREE of the threads scored (the void rule of Exp 27, as a strict bound)."""
    return n > 0 and disagree < VOID_DISAGREE * n - 1e-9


def verdict(short: bool, complete: bool, kept: str | None, is_trusted: bool) -> str:
    if short:
        return "inconclusive_sample"
    if not complete:
        return "pending"
    return f"keep_{kept}_{'trusted' if is_trusted else 'untrusted'}"


def boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float] | None:
    """95% CI of mean(b) - mean(a), the two groups resampled on their own (they are different threads)."""
    if len(a) == 0 or len(b) == 0:
        return None
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
    return [float(np.percentile(db - da, 2.5)), float(np.percentile(db - da, 97.5))]


def confusion(verdicts: dict, answers: dict, hs: list[str]) -> dict:
    return {"judge_yes_user_yes": sum(verdicts[h] and answers[h] == "y" for h in hs),
            "judge_yes_user_no": sum(verdicts[h] and answers[h] == "n" for h in hs),
            "judge_no_user_yes": sum((not verdicts[h]) and answers[h] == "y" for h in hs),
            "judge_no_user_no": sum((not verdicts[h]) and answers[h] == "n" for h in hs)}


def share(flags: list[bool]) -> dict:
    return {"n": len(flags), "one_bit": sum(flags), "share": sum(flags) / len(flags) if flags else None}


def summarize(records: list[dict], verdicts: dict[str, dict[str, bool | None]], answers: dict[str, str],
              exclude: set[str], n_boot: int = N_BOOT) -> dict:
    d = draw(records, exclude)
    picks = d["picks"]
    labelled = [h for h in picks if h in answers]
    failures = {v: sum(verdicts[v].get(h) is None for h in picks) for v in VERSIONS}
    scored = [h for h in labelled if all(verdicts[v].get(h) is not None for v in VERSIONS)]
    complete = bool(picks) and len(labelled) == len(picks) and not any(failures.values())
    ok = {v: np.array([(verdicts[v][h]) == (answers[h] == "y") for h in scored], dtype=float) for v in VERSIONS}
    judges = {}
    for v in VERSIONS:
        agree = int(ok[v].sum())
        judges[v] = {"agree": agree, "disagree": len(scored) - agree,
                     "agreement": agree / len(scored) if scored else None,
                     "disagreement": (len(scored) - agree) / len(scored) if scored else None,
                     "confusion": confusion(verdicts[v], answers, scored)}
    diffs = ok["v2"] - ok["v1"]
    ci = cg.boot_ci(diffs, cg.boot_idx(len(diffs), n_boot, BOOT_SEED)) if len(diffs) >= 2 else None
    arm_flags = {a: [answers[h] == "y" for h in labelled if d["arm_of"][h] == a] for a in ARMS}
    kept = keep(judges["v1"]["agree"], judges["v2"]["agree"]) if complete else None
    is_trusted = complete and trusted(judges[kept]["disagree"], len(scored))
    res = {"n_prompts": len(records) // len(ARMS), "drawn": drawn_numbers(d), "labelled": len(labelled), "scored": len(scored),
           "judge_failures": failures, "complete": complete, "judges": judges,
           "paired": {"v2_only_right": int((diffs > 0).sum()), "v1_only_right": int((diffs < 0).sum()),
                      "mean_diff": float(diffs.mean()) if len(diffs) else None, "ci95": ci},
           "user_one_bit_by_arm": {a: share(arm_flags[a]) for a in ARMS},
           "user_one_bit_diff_B_minus_A": {
               "diff": (share(arm_flags["B"])["share"] - share(arm_flags["A"])["share"]) if all(arm_flags.values()) else None,
               "ci95": boot_diff(np.array(arm_flags["A"], dtype=float), np.array(arm_flags["B"], dtype=float), n_boot)},
           "judge_one_bit_by_arm": {v: {a: share([bool(verdicts[v][h]) for h in picks if d["arm_of"][h] == a and verdicts[v].get(h) is not None])
                                        for a in ARMS} for v in VERSIONS},
           "void_at_share": VOID_DISAGREE, "kept": kept, "kept_trusted": is_trusted if complete else None,
           "rescore_allowed": bool(complete and not d["short"] and kept == "v2" and is_trusted),
           "verdict": verdict(d["short"], complete, kept, bool(is_trusted))}
    return res


# --- noise of the rule (stated before any data) -----------------------------------------------------------------

def p_keep_v2(p1: float, p2: float, n: int = PER_ARM * len(ARMS)) -> float:
    """Chance the rule keeps v2 if v1 and v2 are each right on a thread independently with probabilities p1, p2: v2
    strictly more often right than v1. Exact (trinomial over 'v2 only right', 'v1 only right', 'same')."""
    a, b = p2 * (1 - p1), p1 * (1 - p2)
    c = 1 - a - b
    total = 0.0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            if i > j:
                total += math.factorial(n) / (math.factorial(i) * math.factorial(j) * math.factorial(n - i - j)) \
                    * a ** i * b ** j * c ** (n - i - j)
    return total


def simulate_power(p1: float, p2: float, n: int = PER_ARM * len(ARMS), reps: int = 400, n_boot: int = 1000,
                   seed: int = BOOT_SEED) -> dict:
    """Same model, simulated: how often the rule keeps v2 and how often the paired CI's lower bound is above 0."""
    rng = np.random.default_rng(seed)
    idx = cg.boot_idx(n, n_boot, seed)
    kept = excl = 0
    for _ in range(reps):
        d = (rng.random(n) < p2).astype(float) - (rng.random(n) < p1).astype(float)
        kept += (d > 0).sum() > (d < 0).sum()
        excl += d.mean() > 0 and cg.boot_ci(d, idx)[0] > 0
    return {"keep_v2": kept / reps, "ci_lower_above_zero": excl / reps}


# --- running -------------------------------------------------------------------------------------------------------

def run(n_prompts: int = N_PROMPTS, seed: str = SEED, digest_: str = "", judge_digest_v1: str = "",
        judge_digest_v2: str = "") -> dict:
    assert digest_ == digest(n_prompts, seed), "the prompts, timing or request changed since this call was built"
    assert judge_digest_v1 == tc.judge_digest("v1") and judge_digest_v2 == tc.judge_digest("v2"), "a judge changed"
    client = generate.writer_client()
    bits, directory = prompts(n_prompts, seed), run_dir(n_prompts, seed)
    if gather(directory, bits) is None and not getattr(client, "local", False):
        raise SystemExit("Exp 30 writes with the local writer: set JOKE_WRITER_PROVIDER=ollama")
    records = tc.generate_bits(client, bits, directory)
    exclude = reviewed_before()
    picks = draw(records, exclude)["picks"]
    jclient = tc.make_judge_client()
    verdicts = {v: judge_picks(jclient, records, picks, directory, v) for v in VERSIONS}
    return summarize(records, verdicts, read_answers(directory), exclude)


def score(n_prompts: int = N_PROMPTS, seed: str = SEED) -> dict:
    """The numbers again from the run folder with the user's labels; no model call."""
    bits, directory = prompts(n_prompts, seed), run_dir(n_prompts, seed)
    records = gather(directory, bits)
    if records is None:
        raise SystemExit("not all bits are written yet: run `python -m sim.judge_calibration run` first")
    exclude = reviewed_before()
    return summarize(records, stored_verdicts(records, draw(records, exclude)["picks"], directory), read_answers(directory), exclude)


def stored() -> dict | None:
    return cache.load(NAME, params())


def table(res: dict) -> str:
    j, p, u = res["judges"], res["paired"], res["user_one_bit_by_arm"]
    ci = "n/a" if p["ci95"] is None else f"{p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f}"
    mean = "n/a" if p["mean_diff"] is None else f"{p['mean_diff']:+.2f}"
    f = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    rows = ["| Check | Result | Rule |", "|---|---|---|",
            f"| Threads labelled / scored | {res['labelled']} / {res['scored']} of {res['drawn']['asked']} "
            f"({res['drawn']['arm_A']} from A, {res['drawn']['arm_B']} from B) | 20, 10 per arm |",
            f"| v1 agrees with you | {j['v1']['agree']} of {res['scored']} ({f(j['v1']['agreement'])}) | |",
            f"| v2 agrees with you | {j['v2']['agree']} of {res['scored']} ({f(j['v2']['agreement'])}) | keep v2 only if higher than v1 |",
            f"| v2 minus v1 (threads where only v2 / only v1 is right: {p['v2_only_right']} / {p['v1_only_right']}) | {mean}, CI {ci} | beside the rule |",
            f"| Kept judge's disagreement with you | {res['kept']}: {f(j[res['kept']]['disagreement']) if res['kept'] else 'n/a'} | under {res['void_at_share']:.0%} to re-score Exp 27 |",
            f"| Verdict | {res['verdict']} (re-score allowed: {res['rescore_allowed']}) | |",
            f"| You called one bit: arm A / arm B | {f(u['A']['share'])} of {u['A']['n']} / {f(u['B']['share'])} of {u['B']['n']} | judge-free, beside the rule |"]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("run", "review", "score"))
    ap.add_argument("--rerun", action="store_true", help="run: recompute from the run folder even if a record exists")
    ap.add_argument("--env-file")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file, override=True)
    os.environ.setdefault("JOKE_WRITER_PROVIDER", "ollama")
    if a.command == "run":
        res = cache.run_or_load(NAME, run, rerun=a.rerun, **params())
    elif a.command == "review":
        bits, directory = prompts(), run_dir()
        records = gather(directory, bits)
        if records is None:
            raise SystemExit("run `python -m sim.judge_calibration run` first")
        done = review(directory, records)
        print(f"{done} of {draw(records, reviewed_before())['asked']} answered; now `score`.")
        return
    else:
        res = score()
        cache.save(NAME, params(), res)
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
