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
Exp 27's 10 reviewed hashes must load (`reviewed_before` fails loudly otherwise; `reviewed_known` is recorded).

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
Trust also needs the kept judge to be right at least once in each user class (a judge that always says yes can reach
13 of 20 against a user who says yes often). The gate is noise-sized: a judge whose true agreement is 50/60/65/70/80% passes
it 13/42/60/77/97% of the time (`p_trusted`), and the same 20 labels pick and certify the winner (winner's curse), so the
record gives a Wilson interval for the kept judge's agreement, each judge's yes-rate and Cohen's kappa. The paired CI
resamples whole bits; the by-arm one-bit difference resamples prompts. Gen-v5 adoption (a later PR, `adoptable()`) needs: the
rescored bar passes, v2 kept and trusted, the user's B-minus-A one-bit share not negative, the kept judge's B-minus-A inside
the user's B-minus-A interval (callbacks are in gen-v5's prompt and in v2's criterion, so the judge must not rate B higher
than the user's labels allow), and a re-score made under the current Exp 30 record.
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
    """Hashes of the 10 threads the user checked in Exp 27, from its local run folder. Fails loudly unless exactly
    tc.REVIEW_N load: a missing folder (another checkout, a changed writer setting that moved the digest) would
    otherwise exclude nothing without a word, and the fresh sample could repeat a thread v2 was written from."""
    got = set(tc.read_answers(tc.run_dir()))
    if len(got) != tc.REVIEW_N:
        raise SystemExit(f"found {len(got)} of Exp 27's {tc.REVIEW_N} reviewed threads in {tc.run_dir() / 'review.json'}: "
                         "run from the checkout that holds data/thread_coherence, with the writer settings Exp 27 ran with")
    return got


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
    return {"picks": picks, "asked": len(picks), "reviewed_known": len(exclude), "arm_A": len(picks_by_arm["A"]), "arm_B": len(picks_by_arm["B"]),
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


def both_classes(conf: dict) -> bool:
    """The kept judge must be right at least once where the user says one bit and once where the user says separate
    jokes. A judge that always says yes can reach 13 of 20 against a user who says yes often, and this stops it."""
    return conf["judge_yes_user_yes"] > 0 and conf["judge_no_user_no"] > 0


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]


def kappa(judge: list[bool], user: list[bool]) -> float | None:
    """Cohen's kappa: agreement beyond what the two yes-rates give by chance. None when chance is 1 (all one answer)."""
    n = len(judge)
    if n == 0:
        return None
    po = sum(j == u for j, u in zip(judge, user)) / n
    pj, pu = sum(judge) / n, sum(user) / n
    pe = pj * pu + (1 - pj) * (1 - pu)
    return None if pe >= 1 - 1e-12 else (po - pe) / (1 - pe)


def balanced_accuracy(judge: list[bool], user: list[bool]) -> float | None:
    rates = [sum(j for j, u in zip(judge, user) if u) / sum(user) if sum(user) else None,
             sum((not j) for j, u in zip(judge, user) if not u) / (len(user) - sum(user)) if len(user) - sum(user) else None]
    rates = [r for r in rates if r is not None]
    return sum(rates) / len(rates) if len(rates) == 2 else None


def cluster_boot_mean(x: np.ndarray, clusters: list[str], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float] | None:
    """95% CI of the mean of `x` resampling whole clusters (the bit, so 2 threads of one bit move together)."""
    ids = sorted(set(clusters))
    if len(ids) < 2:
        return None
    sums = np.array([sum(v for v, c in zip(x, clusters) if c == i) for i in ids], dtype=float)
    cnt = np.array([sum(c == i for c in clusters) for i in ids], dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(ids), (n_boot, len(ids)))
    means = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def boot_by_prompt(items: list[tuple[str, str, float]], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float] | None:
    """95% CI of mean(B) - mean(A) for (prompt, arm, value) items, resampling prompts: both arms of a prompt move
    together, so it is paired by prompt. A resample with no thread in an arm is dropped."""
    ps = sorted({p for p, _, _ in items})
    if len(ps) < 2 or not any(a == "A" for _, a, _ in items) or not any(a == "B" for _, a, _ in items):
        return None
    tab = {arm: (np.array([sum(v for p, a, v in items if p == q and a == arm) for q in ps], dtype=float),
                 np.array([sum(1 for p, a, _ in items if p == q and a == arm) for q in ps], dtype=float)) for arm in ARMS}
    idx = np.random.default_rng(seed).integers(0, len(ps), (n_boot, len(ps)))
    na, nb = tab["A"][1][idx].sum(axis=1), tab["B"][1][idx].sum(axis=1)
    ok = (na > 0) & (nb > 0)
    diff = (tab["B"][0][idx].sum(axis=1)[ok] / nb[ok]) - (tab["A"][0][idx].sum(axis=1)[ok] / na[ok])
    return [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))] if len(diff) else None


def confusion(verdicts: dict, answers: dict, hs: list[str]) -> dict:
    return {"judge_yes_user_yes": sum(verdicts[h] and answers[h] == "y" for h in hs),
            "judge_yes_user_no": sum(verdicts[h] and answers[h] == "n" for h in hs),
            "judge_no_user_yes": sum((not verdicts[h]) and answers[h] == "y" for h in hs),
            "judge_no_user_no": sum((not verdicts[h]) and answers[h] == "n" for h in hs)}


def false_connected(conf: dict) -> float | None:
    """Of the threads the user called separate jokes, the share the judge called one bit."""
    n = conf["judge_yes_user_no"] + conf["judge_no_user_no"]
    return conf["judge_yes_user_no"] / n if n else None


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
    user = [answers[h] == "y" for h in scored]
    ok = {v: np.array([(verdicts[v][h]) == (answers[h] == "y") for h in scored], dtype=float) for v in VERSIONS}
    judges = {}
    for v in VERSIONS:
        agree = int(ok[v].sum())
        flags = [bool(verdicts[v][h]) for h in scored]
        judges[v] = {"agree": agree, "disagree": len(scored) - agree,
                     "agreement": agree / len(scored) if scored else None,
                     "agreement_ci95": wilson(agree, len(scored)),
                     "disagreement": (len(scored) - agree) / len(scored) if scored else None,
                     "yes_rate": sum(flags) / len(flags) if flags else None,
                     "kappa": kappa(flags, user), "balanced_accuracy": balanced_accuracy(flags, user),
                     "confusion": confusion(verdicts[v], answers, scored),
                     "confusion_by_arm": {a: confusion(verdicts[v], answers, [h for h in scored if d["arm_of"][h] == a]) for a in ARMS}}
        judges[v]["false_connected_by_arm"] = {a: false_connected(judges[v]["confusion_by_arm"][a]) for a in ARMS}
    diffs = ok["v2"] - ok["v1"]
    bit_ids = [d["bit_of"][h] for h in scored]
    ci = cluster_boot_mean(diffs, bit_ids, n_boot) if len(diffs) else None
    ci_thread = cg.boot_ci(diffs, cg.boot_idx(len(diffs), n_boot, BOOT_SEED)) if len(diffs) >= 2 else None
    arm_flags = {a: [answers[h] == "y" for h in labelled if d["arm_of"][h] == a] for a in ARMS}
    prompt_of = lambda h: d["bit_of"][h].split("-")[0]  # noqa: E731
    user_ci = boot_by_prompt([(prompt_of(h), d["arm_of"][h], float(answers[h] == "y")) for h in labelled], n_boot)
    user_diff = (share(arm_flags["B"])["share"] - share(arm_flags["A"])["share"]) if all(arm_flags.values()) else None
    judge_share = {v: {a: share([bool(verdicts[v][h]) for h in picks if d["arm_of"][h] == a and verdicts[v].get(h) is not None])
                       for a in ARMS} for v in VERSIONS}
    judge_diff = {v: (judge_share[v]["B"]["share"] - judge_share[v]["A"]["share"]) if all(judge_share[v][a]["n"] for a in ARMS) else None
                  for v in VERSIONS}
    kept = keep(judges["v1"]["agree"], judges["v2"]["agree"]) if complete else None
    rate_ok = complete and trusted(judges[kept]["disagree"], len(scored))
    classes_ok = complete and both_classes(judges[kept]["confusion"])
    is_trusted = bool(rate_ok and classes_ok)
    guards = {"user_diff_not_negative": user_diff is not None and user_diff >= 0,
              "judge_diff_within_user_ci": bool(kept and user_ci and judge_diff[kept] is not None
                                                and user_ci[0] <= judge_diff[kept] <= user_ci[1])}
    return {"n_prompts": len(records) // len(ARMS), "drawn": drawn_numbers(d), "labelled": len(labelled), "scored": len(scored),
            "judge_failures": failures, "complete": complete, "judges": judges,
            "user_yes_rate": sum(user) / len(user) if user else None,
            "paired": {"v2_only_right": int((diffs > 0).sum()), "v1_only_right": int((diffs < 0).sum()),
                       "mean_diff": float(diffs.mean()) if len(diffs) else None, "ci95": ci, "ci95_thread_level": ci_thread},
            "user_one_bit_by_arm": {a: share(arm_flags[a]) for a in ARMS},
            "user_one_bit_diff_B_minus_A": {"diff": user_diff, "ci95": user_ci},
            "judge_one_bit_by_arm": judge_share,
            "judge_one_bit_diff_B_minus_A": judge_diff,
            "void_at_share": VOID_DISAGREE, "kept": kept,
            "kept_agreement_ci95": judges[kept]["agreement_ci95"] if kept else None,
            "kept_right_in_each_class": classes_ok if complete else None,
            "kept_trusted": is_trusted if complete else None,
            "adoption_guards": guards,
            "rescore_allowed": bool(complete and not d["short"] and kept == "v2" and is_trusted),
            "verdict": verdict(d["short"], complete, kept, is_trusted)}


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


def p_trusted(agreement: float, n: int = PER_ARM * len(ARMS)) -> float:
    """Chance a judge with this true per-thread agreement passes the trust gate (disagreement under 40%): at least
    n - ceil(0.4 n) + 1 agreements out of n. Exact binomial. Same 20 labels pick and certify the winner, so the kept
    judge is the luckier of two and its real chance is lower than this table says (winner's curse)."""
    need = n - math.ceil(VOID_DISAGREE * n - 1e-9) + 1
    return sum(math.comb(n, k) * agreement ** k * (1 - agreement) ** (n - k) for k in range(need, n + 1))


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


def adoptable() -> dict:
    """Every condition for the later gen-v5 adoption PR, read from the two stored records (D-114). `adoptable` is
    their conjunction; a re-score made under an earlier Exp 30 verdict, or for another n, is stale and does not count.
    The decisive re-score is the 100-prompt extension (tc.N_EXT): the 20-prompt one failed at exactly 0 (2026-11-26)."""
    cal = stored()
    rs = tc.rescore_record("v2", tc.N_EXT)
    g = (cal or {}).get("adoption_guards", {})
    conds = {"rescore_current": rs is not None and tc.rescore_current(rs, tc.N_EXT), "rescored_bar_passes": bool(rs and rs.get("bar_passes")),
             "v2_kept_and_trusted": bool(cal and cal.get("rescore_allowed")),
             "user_diff_not_negative": bool(g.get("user_diff_not_negative")),
             "judge_diff_within_user_ci": bool(g.get("judge_diff_within_user_ci"))}
    return {**conds, "adoptable": all(conds.values())}


def table(res: dict) -> str:
    j, p, u = res["judges"], res["paired"], res["user_one_bit_by_arm"]
    ci = "n/a" if p["ci95"] is None else f"{p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f}"
    mean = "n/a" if p["mean_diff"] is None else f"{p['mean_diff']:+.2f}"
    f = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    fm = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
    rows = ["| Check | Result | Rule |", "|---|---|---|",
            f"| Threads labelled / scored | {res['labelled']} / {res['scored']} of {res['drawn']['asked']} "
            f"({res['drawn']['arm_A']} from A, {res['drawn']['arm_B']} from B) | 20, 10 per arm |",
            f"| v1 agrees with you | {j['v1']['agree']} of {res['scored']} ({f(j['v1']['agreement'])}) | |",
            f"| v2 agrees with you | {j['v2']['agree']} of {res['scored']} ({f(j['v2']['agreement'])}) | keep v2 only if higher than v1 |",
            f"| v2 minus v1 (threads where only v2 / only v1 is right: {p['v2_only_right']} / {p['v1_only_right']}) | {mean}, CI {ci} | beside the rule |",
            f"| Agreement beyond chance (kappa), yes-rate: v1 / v2 | {fm(j['v1']['kappa'])}, {f(j['v1']['yes_rate'])} / {fm(j['v2']['kappa'])}, {f(j['v2']['yes_rate'])} (you: {f(res['user_yes_rate'])}) | beside the rule |",
            f"| Kept judge's disagreement with you | {res['kept']}: {f(j[res['kept']]['disagreement']) if res['kept'] else 'n/a'} | under {res['void_at_share']:.0%} to re-score Exp 27 |",
            f"| Verdict | {res['verdict']} (re-score allowed: {res['rescore_allowed']}; right in each user class: {res['kept_right_in_each_class']}) | |",
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
