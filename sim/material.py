"""Exp 33 (D-117): does giving each bit a topic and two character facts move the writer off the robot's own hardware
without making it less funny? Exp 34 (D-118) asks it again with the topic alone (arm C), against Exp 33's arm A.

Usage (Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`, `ANTHROPIC_API_KEY` set for the judge):
    python -m sim.material run [--env-file backend/.env]   # write both arms, label every bit -> reports/metrics/material.json
    python -m sim.material review                          # you compare 20 pairs, sides hidden and randomized
    python -m sim.material score [--record]                # recompute the numbers with your answers (no model calls)
Exp 34: the same three commands with `--exp 34` (its record is reports/metrics/material_topic.json).

Design: 5 consecutive simulated shows of 8 bits, 40 bits per arm. The mode and the room of every bit come from one fixed
seed and are the same for both arms (paired). Arm A is the live prompt with material off. Arm B is the same prompt with
material on; its topics skip the ones its earlier shows in this run used, starting from an empty history, so the run does
not depend on this laptop's logs. The thread played is thread 0 (the show plays one thread through). The writer is the
live one (`generate.writer_client()`); bits are written one at a time. Generated text lives only under `data/material/`
(gitignored); the metrics hold numbers.

Metric 1 (repetition, gates): the share of bits whose played thread (premise and lines) is about the robot's own
hardware, labeled by a Claude Haiku 5.5 judge with one fixed yes/no question, blind to the arm. A word-list share is
reported beside it as a cross-check. Bar, fixed before any run: B's share is at most 15% AND the paired bootstrap 95% CI of
(B - A), paired by bit slot, lies entirely below 0. Reported with no bar: the share separately for bits drawn from a
character topic and for everyday-topic bits.

Metric 2 (you, blind, gates funniness): 20 pairs, one per bit slot (seeded), A against B, sides randomized. For each pair
you answer (a) which is funnier and (b) which sounds more like a specific character. Bar for (a): B wins at least 10 of
20; a tie is not a win. (b) is reported with a Wilson 95% interval and is not gated. 20 pairs only detect large
differences: with no real difference B still reaches 10 wins about 59% of the time (ties ignored), so the bar reads as
"not clearly worse", not "funnier".

Adopt (flip the MATERIAL default to on, same PR) only when both bars pass. No `audience.yaml` value is involved: no sim
room reward is read.

Code changes elsewhere (the generator, the material files) change the digest: rerun with `--rerun` after one.

Exp 34 (D-118): arm C is the topic alone (no robot facts, its own paragraph, generator id +mat-topic-v1), same 40 slots and
same material seeds as Exp 33, so C's topics are B's topics. Arm A is NOT rewritten: Exp 33's arm A bits are read from
Exp 33's run folder, and the run refuses to start when that folder, its slots or its modes do not match. Same two metrics
and bars with C in B's place: C's hardware share at most 15% and the paired bootstrap 95% CI of (C - A) entirely below 0;
and C wins at least 10 of 20 new blind pairs (pair seed 34, not 32) on funnier. The pairs come only from the 20 slots Exp 33's
review did not show: the user has read those slots' arm A bits, so a familiar bit would mark its side as A. This is a second
attempt at the bars Exp 33 missed: under no real difference each attempt passes the funniness bar about 59% of the time, so a
pass is weak evidence.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import random
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import generate, llm  # noqa: E402
from app import material as app_material  # noqa: E402
from app.cache import normalise  # noqa: E402
from sim import cache  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim import thread_coherence as tc  # noqa: E402
from sim.blind import plain  # noqa: E402
from sim.env import AGES, ENERGIES, VENUES  # noqa: E402

NAME = "material"
N_SHOWS = 5
BITS_PER_SHOW = 8
SEED = "exp32"  # predates the renumbering to Exp 33; it feeds the run_or_load digest, so changing it orphans the record
ARMS = ("A", "B")  # A: material off, B: material on
N_BOOT = 10_000
BOOT_SEED = 32
DATA = ROOT / "data" / "material"
MAX_SHARE_B = 0.15
N_PAIRS = 20
PAIR_SEED = 32
FUNNY_WINS = 10
SHOWN = "shown_in_exp33_review"  # on a reused arm A bit: the user read it in Exp 33's review, so Exp 34 does not pair it


@dataclasses.dataclass(frozen=True)
class Exp:
    """What differs between Exp 33 and Exp 34. `arm` is the material arm; `seed` feeds the digest and the record;
    `slot_seed` orders the slots and the topics; `reuse_a` reads arm A from Exp 33's run folder instead of writing it."""
    name: str  # the record, reports/metrics/<name>.json
    seed: str
    slot_seed: str
    arm: str
    variant: str
    boot_seed: int
    pair_seed: int
    reuse_a: bool = False

    @property
    def arms(self) -> tuple[str, str]:
        return ("A", self.arm)


EXP33 = Exp(NAME, SEED, SEED, "B", app_material.FULL, BOOT_SEED, PAIR_SEED)
EXP34 = Exp("material_topic", "exp34", SEED, "C", app_material.TOPIC, 34, 34, reuse_a=True)
EXPS = {"33": EXP33, "34": EXP34}

HARDWARE_JUDGE = """You label stand-up material. You get the premise and five lines of one bit, written for a robot comedian.

Question: is the bit about the robot's own hardware or parts: its battery, power, charging, processor, memory, data storage, wires, cooling, shutting down or rebooting?

Answer true only when that hardware is what the bit is mostly about. A passing mention in one line is false. Judge the topic only, never whether it is funny.

Reply with JSON only: {"hardware": true} or {"hardware": false}."""
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["hardware"],
                "properties": {"hardware": {"type": "boolean"}}}
# The cross-check: the words that made up the 28% in the live logs (battery, power, data, charging, processor, memory).
HARDWARE_WORDS = re.compile(r"\b(batter(?:y|ies)|charg\w*|power(?:ed|s)?|processor\w*|cpu|gpu|memory|ram|data|wires?|"
                            r"circuit\w*|motherboard|overheat\w*|shut(?:ting)? ?down|reboot\w*|hard drive|storage)\b", re.I)  # no "fans": "sports fans" is a topic


# --- the slots, digests, folders ---------------------------------------------------------------------------------

def slots(n_shows: int = N_SHOWS, bits: int = BITS_PER_SHOW, seed: str = SEED) -> list[dict]:
    """show-major, bit-minor. A show has one room (venue, energy, age); every bit has its own mode. All from the seed, so
    both arms and every rerun see the same slots."""
    modes = tuple(generate.MODES)
    out = []
    for s in range(n_shows):
        r = random.Random(f"{seed}:room:{s}")
        context = {"venue": r.choice(VENUES), "energy": r.choice(ENERGIES), "age": r.choice(AGES)}
        for b in range(bits):
            out.append({"id": f"s{s}b{b}", "show": s, "bit": b, "mode": random.Random(f"{seed}:mode:{s}:{b}").choice(modes),
                        "context": context})
    return out


def material_seed(show: int, seed: str = SEED) -> int:
    return int(hashlib.sha256(f"{seed}:material:{show}".encode()).hexdigest()[:8], 16)


def file_hash(name: str) -> str:
    return hashlib.sha256((app_material.DATA / f"{name}.yaml").read_bytes()).hexdigest()[:12]


def digest(n_shows: int = N_SHOWS, bits: int = BITS_PER_SHOW, seed: str = SEED, exp: Exp = EXP33) -> str:
    """What the written bits are made of: the live prompt, the material paragraph, both material files, the draw rule, the
    slots, the writer and its settings. A change starts a new run folder. The judge has its own digest."""
    parts = [generate.PROMPT_ID, generate.PROMPTS[generate.PROMPT_ID], generate.MATERIAL_PARAGRAPH, generate.MATERIAL_SUFFIX,
             generate.TEENS_RULE, json.dumps(generate.MODES, sort_keys=True), json.dumps(generate.COUNTED_REPLY_SCHEMA, sort_keys=True),
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.RETRIES,
                  generate.WRITER_CONTEXT, generate.HANG_GRACE_S, llm.MIN_ATTEMPT_S)),
             tc.writer_model(), json.dumps(llm.OLLAMA_NO_THINK, sort_keys=True), file_hash(app_material.SHEET_ID),
             file_hash(app_material.LIST_ID), str((app_material.FACTS_PER_BIT, app_material.CHARACTER_EVERY)),
             json.dumps(slots(n_shows, bits, seed), sort_keys=True), seed, str((n_shows, bits))]
    if exp != EXP33:  # Exp 33's folder name stays what it was; Exp 34 adds what is its own
        parts += [exp.name, exp.seed, exp.arm, exp.variant, generate.TOPIC_PARAGRAPH, generate.TOPIC_SUFFIX]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8]


def judge_digest() -> str:
    shown = judge_user("a premise", ["a ’b’ — c"])
    return hashlib.sha256("\x00".join([HARDWARE_JUDGE, json.dumps(JUDGE_SCHEMA, sort_keys=True), shown,
                                       *map(str, tc.judge_model())]).encode()).hexdigest()[:8]


def run_dir(n_shows: int = N_SHOWS, bits: int = BITS_PER_SHOW, seed: str = SEED, root: pathlib.Path | None = None,
            exp: Exp = EXP33) -> pathlib.Path:
    return (root or DATA) / f"run-{digest(n_shows, bits, seed, exp)}"


def params(n_shows: int = N_SHOWS, bits: int = BITS_PER_SHOW, seed: str | None = None, exp: Exp = EXP33) -> dict:
    """`seed` is the slot seed (what orders slots and topics); the record's own seed string is `exp.seed`."""
    slot_seed = exp.slot_seed if seed is None else seed
    return {"n_shows": n_shows, "bits_per_show": bits, "seed": exp.seed, "digest_": digest(n_shows, bits, slot_seed, exp),
            "judge_digest_": judge_digest()}


# --- writing -----------------------------------------------------------------------------------------------------

def bit_path(directory: pathlib.Path, slot_id: str, arm: str) -> pathlib.Path:
    return directory / "bits" / f"{slot_id}-{arm}.json"


def write_bit(client, gen, slot: dict, arm: str) -> dict:
    """One bit through the show's own path; the played thread is thread 0. The bit is then told to the generator (which
    spends its material), exactly as the show does when a thread is played."""
    bad0 = client.bad_replies
    start = time.perf_counter()
    rec = {"slot": slot["id"], "arm": arm, "mode": slot["mode"], "premise": None, "lines": [], "topic": None, "from": None,
           "facts": [], "recycled": None, "lost": [], "failure": None}
    try:
        batches = [gen.generate(slot["mode"], slot["bit"], j, slot["context"]) for j in range(generate.JOKES)]
        rec["lines"] = [b.texts[0] for b in batches]
        rec["premise"] = gen._bits[slot["bit"]][2][0]
        rec["lost"] = list(batches[0].threads_lost)
        gen.played_thread(slot["bit"], 0)
        if (m := gen.material_row(slot["bit"])):
            rec.update(topic=m["topic"], facts=m["facts"], recycled=m["recycled"], **{"from": m["from"]})
    except Exception as e:  # noqa: BLE001 - a bit with no thread is a result, not a crash
        rec["failure"] = type(e).__name__
        rec["lost"] = list(getattr(e, "threads_lost", [])) or [f"{type(e).__name__}: {e}"[:100]]
    rec.update(wall_s=time.perf_counter() - start, bad_replies=client.bad_replies - bad0)
    return rec


def restore(gen, rec: dict, slot: dict) -> None:
    """Puts a recorded bit back into a fresh generator, so a resumed run draws what an uninterrupted one would."""
    if rec["premise"] is None:
        return
    gen._told.append(rec["premise"])
    if gen.material and rec["topic"] is not None:
        gen.material.played(slot["bit"], {"topic": rec["topic"], "facts": rec["facts"]})


def new_generator(client, arm: str, show: int, history: dict[str, float], seed: str, exp: Exp = EXP33):
    mat = app_material.ShowMaterial(material_seed(show, seed), history, variant=exp.variant) if arm == exp.arm else None
    return generate.ThreadGenerator(client, prompt_id=generate.PROMPT_ID, material=mat)


def reused_a(bits: list[dict], source: pathlib.Path) -> dict[str, dict]:
    """Exp 34's arm A: Exp 33's stored bits, slot id -> record. Refuses (SystemExit) when the folder is missing, a bit is
    missing, or a bit is not that slot's material-off bit, because a silent mismatch would pair C with the wrong A."""
    if not source.is_dir():
        raise SystemExit(f"Exp 34 reads Exp 33's arm A bits from {source}, which does not exist: Exp 33's prompt, material files or "
                         f"slots changed since it ran (its folder name is the digest), or it never ran here. Not rewriting arm A.")
    out = {}
    for s in bits:
        r = tc._read(bit_path(source, s["id"], "A"))
        if r is None:
            raise SystemExit(f"Exp 34: Exp 33's arm A bit for slot {s['id']} is missing from {source}")
        if r.get("slot") != s["id"] or r.get("arm") != "A" or r.get("mode") != s["mode"] or r.get("topic") is not None:
            raise SystemExit(f"Exp 34: the stored arm A bit for slot {s['id']} does not match the slot (slot, arm, mode or "
                             f"material-off): refusing to pair it")
        out[s["id"]] = r
    shown = exp33_shown(bits, source)
    return {sid: {**r, SHOWN: sid in shown} for sid, r in out.items()}


def exp33_shown(bits: list[dict], source: pathlib.Path) -> set[str]:
    """The slots Exp 33's review showed the user: its seeded pairs, plus any slot answered. Refuses when Exp 33's arm B
    bits are gone, since then nobody can tell which arm A bits the user has already read."""
    records = gather(source, bits, EXP33)
    if records is None:
        raise SystemExit(f"Exp 34: Exp 33's arm B bits are incomplete in {source}, so the slots its review showed cannot be "
                         f"recomputed: refusing to draw pairs that might show the user a bit already seen")
    return {p["slot"] for p in pairs(records, exp=EXP33)} | set(read_answers(source))


def generate_bits(client, bits: list[dict], directory: pathlib.Path, seed: str = SEED, http=None, exp: Exp = EXP33,
                  a_source: pathlib.Path | None = None) -> list[dict]:
    """Writes every missing (slot, arm) bit, resumable. The material arm's history is the topics its earlier shows in this
    run played (an empty history at show 0). The arm order alternates by slot so a slow stretch of the machine does not fall
    on one arm. Calls run one bit at a time. With `exp.reuse_a`, arm A is read from `a_source` and never written."""
    arms = exp.arms
    fixed_a = reused_a(bits, a_source) if exp.reuse_a else {}
    todo = [(s, a) for s in bits for a in arms if (s["id"] not in fixed_a or a != "A") and tc._read(bit_path(directory, s["id"], a)) is None]
    if todo:
        probe = generate.ThreadGenerator(client)
        generate.warm_up(probe, http) if http else generate.warm_up(probe)
    out: dict[tuple[str, str], dict] = {}
    history = {a: {} for a in arms}  # arm -> topic -> show it was last used in
    gens: dict[tuple[int, str], object] = {}
    for i, s in enumerate(bits):
        for arm in (arms if i % 2 == 0 else reversed(arms)):
            if arm == "A" and s["id"] in fixed_a:
                out[(s["id"], arm)] = fixed_a[s["id"]]
                continue
            key = (s["show"], arm)
            if key not in gens:
                gens[key] = new_generator(client, arm, s["show"], dict(history[arm]), seed, exp)
            gen = gens[key]
            path = bit_path(directory, s["id"], arm)
            rec = tc._read(path)
            if rec is None:
                rec = write_bit(client, gen, s, arm)
                tc._write(path, rec)
            else:
                restore(gen, rec, s)
            out[(s["id"], arm)] = rec
        last_of_show = i + 1 == len(bits) or bits[i + 1]["show"] != s["show"]
        if last_of_show:
            for arm in arms:
                for r in (out[(b["id"], arm)] for b in bits if b["show"] == s["show"]):
                    if r["topic"] is not None:
                        history[arm][normalise(r["topic"])] = float(s["show"])
    if todo:
        share, context, others = generate.placement(client, http) if http else generate.placement(client)
        tc._write(tc.placement_path(directory), {"gpu_share": share, "context_length": context, "others_loaded": others})
    return [out[(s["id"], a)] for s in bits for a in arms]


# --- labeling ----------------------------------------------------------------------------------------------------

def judge_user(premise: str, lines: list[str]) -> str:
    return f"Premise: {plain(premise)}\n" + "\n".join(f"{i}. {plain(x)}" for i, x in enumerate(lines, 1))


def bit_hash(rec: dict) -> str:
    return hashlib.sha256(judge_user(rec["premise"], rec["lines"]).encode()).hexdigest()[:16]


def regex_hardware(rec: dict) -> bool:
    return bool(HARDWARE_WORDS.search(" ".join([rec["premise"], *rec["lines"]])))


def parse_verdict(reply: str) -> bool:
    v = json.loads(llm.strip_fence(reply))["hardware"]
    if not isinstance(v, bool):
        raise ValueError("hardware is not a boolean")
    return v


def verdict_path(directory: pathlib.Path, h: str) -> pathlib.Path:
    return directory / "judge_cache" / judge_digest() / f"{h}.json"


def cached_verdict(dirs: tuple[pathlib.Path, ...], rec: dict) -> bool | None:
    for d in dirs:
        if (cached := tc._read(verdict_path(d, bit_hash(rec)))) is not None:
            return cached["hardware"]
    return None


def label(client, rec: dict, directory: pathlib.Path, also: tuple[pathlib.Path, ...] = ()) -> bool | None:
    """The judge's verdict for a delivered bit (None when it could not answer, an outage is not cached), cached by text.
    `also` are other run folders whose cached verdict for the same text counts (Exp 34 keeps Exp 33's labels for arm A)."""
    path = verdict_path(directory, bit_hash(rec))
    for d in (directory, *also):
        if (cached := tc._read(verdict_path(d, bit_hash(rec)))) is not None:
            return cached["hardware"]
    try:
        verdict = parse_verdict(client.chat(HARDWARE_JUDGE, judge_user(rec["premise"], rec["lines"]), json_reply=JUDGE_SCHEMA,
                                            validate=parse_verdict))
    except (llm.LLMError, ValueError, KeyError, TypeError, AttributeError):
        return None
    tc._write(path, {"hardware": verdict})
    return verdict


def label_all(client, records: list[dict], directory: pathlib.Path, also: tuple[pathlib.Path, ...] = ()) -> dict[str, bool | None]:
    out: dict[str, bool | None] = {}
    for r in records:
        if r["premise"] is not None and (h := bit_hash(r)) not in out:
            out[h] = label(client, r, directory, also)
    return out


# --- metric 1 ----------------------------------------------------------------------------------------------------

def by_slot(records: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in records:
        out.setdefault(r["slot"], {})[r["arm"]] = r
    return out


def hardware_of(rec: dict, verdicts: dict) -> bool | None:
    return None if rec["premise"] is None else verdicts.get(bit_hash(rec))


def share(flags: list[bool | None]) -> dict:
    """{n, hardware, share}: over the bits the judge answered."""
    got = [f for f in flags if f is not None]
    return {"n": len(got), "hardware": sum(got), "share": sum(got) / len(got) if got else None}


def paired(records: list[dict], verdicts: dict, exp: Exp = EXP33) -> tuple[list[str], np.ndarray]:
    """Material arm minus A hardware flag (-1, 0 or 1) for every slot where both arms delivered a bit the judge answered."""
    ids, diffs = [], []
    for sid, arms in sorted(by_slot(records).items()):
        a, b = (hardware_of(arms.get(x), verdicts) if x in arms else None for x in exp.arms)
        if a is not None and b is not None:
            ids.append(sid)
            diffs.append(int(b) - int(a))
    return ids, np.array(diffs, dtype=float)


def bar1(share_b: float | None, diffs: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """The material arm's share at most MAX_SHARE_B AND the paired bootstrap 95% CI of (it - A) entirely below 0
    (resampling slots)."""
    if len(diffs) < 2 or share_b is None:
        return {"n_paired": int(len(diffs)), "mean_diff": None, "ci95": None, "share_ok": False, "ci_below_zero": False,
                "passes": False}
    ci = cg.boot_ci(diffs, cg.boot_idx(len(diffs), n_boot, seed))
    ok, below = share_b <= MAX_SHARE_B + 1e-12, ci[1] < 0
    return {"n_paired": int(len(diffs)), "mean_diff": float(diffs.mean()), "ci95": ci, "share_ok": ok, "ci_below_zero": below,
            "passes": ok and below}


def metric1(records: list[dict], verdicts: dict, n_boot: int = N_BOOT, exp: Exp = EXP33) -> dict:
    arm = {a: [r for r in records if r["arm"] == a] for a in exp.arms}
    shares = {a: share([hardware_of(r, verdicts) for r in arm[a]]) for a in exp.arms}
    ids, diffs = paired(records, verdicts, exp)
    split = {}
    for a in exp.arms:  # no bar: the character-topic bits against the everyday ones, on the material arm's slots in both arms
        for kind in ("character", "everyday"):
            slots_of = {r["slot"] for r in arm[exp.arm] if r["from"] == kind}
            split[f"{a}_{kind}"] = share([hardware_of(r, verdicts) for r in arm[a] if r["slot"] in slots_of])
    regex = {a: share([regex_hardware(r) for r in arm[a] if r["premise"] is not None]) for a in exp.arms}
    return {"share": shares, "regex_share": regex, "by_topic_kind": split, "paired_slots": ids,
            "bar": bar1(shares[exp.arm]["share"], diffs, n_boot, exp.boot_seed)}


# --- metric 2: the blind pairs ----------------------------------------------------------------------------------

def review_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "review.json"


def pairs(records: list[dict], n: int = N_PAIRS, seed: int | None = None, exp: Exp = EXP33) -> list[dict]:
    """n slots where both arms delivered a bit, a fixed seeded draw; `left` says which arm is shown on the left. A slot
    whose arm A bit Exp 33's review already showed is left out (`SHOWN`), so no side is recognizable as the old one."""
    rng = random.Random(exp.pair_seed if seed is None else seed)
    ok = sorted(sid for sid, arms in by_slot(records).items()
                if all(a in arms and arms[a]["premise"] is not None for a in exp.arms) and not arms["A"].get(SHOWN))
    picked = rng.sample(ok, min(n, len(ok)))
    return [{"slot": sid, "left": rng.choice(exp.arms)} for sid in picked]


def read_answers(directory: pathlib.Path) -> dict[str, dict]:
    return (tc._read(review_path(directory)) or {}).get("answers", {})


def ask(read, prompt: str) -> str:
    while True:
        a = read(prompt).strip().lower()
        if a in ("l", "r", "t", "q"):
            return a


def review(directory: pathlib.Path, records: list[dict], read=input, out=print, exp: Exp = EXP33) -> int:
    """Asks about each pair not answered yet, saving after every answer. You see the lines and the premise only: not the arm
    or the topic. l = left, r = right, t = tie (a tie is not a win for B), q = stop for now."""
    drawn, slot = pairs(records, exp=exp), by_slot(records)
    answers = read_answers(directory)
    for n, p in enumerate(drawn, 1):
        if p["slot"] in answers:
            continue
        right = next(a for a in exp.arms if a != p["left"])
        out(f"\nPair {n} of {len(drawn)}")
        for side, arm in (("LEFT", p["left"]), ("RIGHT", right)):
            r = slot[p["slot"]][arm]
            out(f"--- {side}\n" + judge_user(r["premise"], r["lines"]))
        funnier = ask(read, "(a) Which is funnier? [l/r/t/q] ")
        if funnier == "q":
            break
        character = ask(read, "(b) Which sounds more like a specific character? [l/r/t/q] ")
        if character == "q":
            break
        answers[p["slot"]] = {"funnier": funnier, "character": character}
        tc._write(review_path(directory), {"answers": answers})
    return sum(p["slot"] in answers for p in drawn)


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def metric2(records: list[dict], answers: dict[str, dict], exp: Exp = EXP33) -> dict:
    drawn = pairs(records, exp=exp)
    m = exp.arm

    def b_wins(field: str) -> tuple[int, int, int]:
        won = lost = tied = 0
        for p in drawn:
            if (a := answers.get(p["slot"])) is None:
                continue
            side = a[field]
            if side == "t":
                tied += 1
            elif (side == "l") == (p["left"] == m):
                won += 1
            else:
                lost += 1
        return won, lost, tied
    fw, fl, ft = b_wins("funnier")
    cw, cl, ct = b_wins("character")
    asked, answered = len(drawn), fw + fl + ft
    return {"asked": asked, "answered": answered, "complete": asked > 0 and answered == asked,
            "funnier": {f"{m}_wins": fw, "A_wins": fl, "ties": ft, "bar": FUNNY_WINS, "passes": fw >= FUNNY_WINS},
            "character": {f"{m}_wins": cw, "A_wins": cl, "ties": ct, f"{m}_share": cw / (cw + cl + ct) if cw + cl + ct else None,
                          "wilson95": wilson(cw, cw + cl + ct)},
            "short": asked < N_PAIRS}


def decide(m1: dict, m2: dict) -> str:
    """`keep_off`: the repetition bar failed (the user's answers cannot rescue it) or, with it passed, B won fewer than 10 of
    20 on funniness. `pending_review`: the repetition bar passed and the pairs are not all answered. `adopt`: both passed."""
    if not m1["bar"]["passes"]:
        return "keep_off"
    if m2["funnier"]["passes"]:
        return "adopt"  # 10 wins cannot be undone by the pairs still unanswered
    return "pending_review" if not m2["complete"] else "keep_off"


def summarize(records: list[dict], verdicts: dict, answers: dict, n_boot: int = N_BOOT, on_gpu: bool | None = None,
              exp: Exp = EXP33) -> dict:
    m1, m2 = metric1(records, verdicts, n_boot, exp), metric2(records, answers, exp)
    arms = {}
    for a in exp.arms:
        mine = [r for r in records if r["arm"] == a]
        walls = [r["wall_s"] for r in mine]
        arms[a] = {"bits": len(mine), "bits_delivered": sum(r["premise"] is not None for r in mine),
                   "bits_with_no_thread": sum(r["premise"] is None for r in mine), "threads_lost": sum(len(r["lost"]) for r in mine),
                   "bad_replies": sum(r["bad_replies"] for r in mine), "bit_p50_s": cg.percentile(walls, 50),
                   "bit_p95_s": cg.percentile(walls, 95)}
    mats = [r for r in records if r["arm"] == exp.arm and r["topic"] is not None]
    arms[exp.arm].update(recycled_topics=sum(bool(r["recycled"]) for r in mats), character_topic_bits=sum(r["from"] == "character" for r in mats),
                     distinct_topics=len({r["topic"] for r in mats}))
    return {"n_slots": len(by_slot(records)), "arms": arms, "metric1": m1, "metric2": m2, "writer_on_gpu_after_run": on_gpu,
            "judge_failures": sum(v is None for v in verdicts.values()), "verdict": decide(m1, m2)}


# --- running -----------------------------------------------------------------------------------------------------

def gather(directory: pathlib.Path, bits: list[dict], exp: Exp = EXP33, a_source: pathlib.Path | None = None) -> list[dict] | None:
    """Every (slot, arm) bit in the run folder, None when one is missing. With `exp.reuse_a`, arm A comes from `a_source`
    (checked: a mismatch refuses)."""
    fixed_a = reused_a(bits, a_source) if exp.reuse_a else {}
    recs = [fixed_a[s["id"]] if a == "A" and exp.reuse_a else tc._read(bit_path(directory, s["id"], a)) for s in bits for a in exp.arms]
    return None if any(r is None for r in recs) else recs


def a_dir(n_shows: int, bits_per_show: int, exp: Exp) -> pathlib.Path | None:
    """Where this experiment's arm A bits are stored when they are Exp 33's: its run folder (named by its digest)."""
    return run_dir(n_shows, bits_per_show, exp.slot_seed, exp=EXP33) if exp.reuse_a else None


def run(n_shows: int = N_SHOWS, bits_per_show: int = BITS_PER_SHOW, seed: str | None = None, digest_: str = "",
        judge_digest_: str = "", exp: Exp = EXP33) -> dict:
    slot_seed = exp.slot_seed
    assert digest_ == digest(n_shows, bits_per_show, slot_seed, exp), "the prompt, material files, timing or slots changed since this call was built"
    assert judge_digest_ == judge_digest(), "the judge's prompt or model changed since this call was built"
    client = generate.writer_client()
    bits, directory = slots(n_shows, bits_per_show, slot_seed), run_dir(n_shows, bits_per_show, slot_seed, exp=exp)
    source = a_dir(n_shows, bits_per_show, exp)
    if gather(directory, bits, exp, source) is None and not getattr(client, "local", False):
        raise SystemExit(f"Exp {EXP_NUMBER[exp.name]} writes with the local writer: set JOKE_WRITER_PROVIDER=ollama")
    records = generate_bits(client, bits, directory, slot_seed, exp=exp, a_source=source)
    verdicts = label_all(tc.make_judge_client(), records, directory, (source,) if source else ())
    return summarize(records, verdicts, read_answers(directory), on_gpu=tc.writer_on_gpu(directory), exp=exp)


def score(n_shows: int = N_SHOWS, bits_per_show: int = BITS_PER_SHOW, exp: Exp = EXP33) -> dict:
    """The numbers again from the run folder with your answers, no model call. Raises when a bit is missing."""
    bits, directory = slots(n_shows, bits_per_show, exp.slot_seed), run_dir(n_shows, bits_per_show, exp.slot_seed, exp=exp)
    source = a_dir(n_shows, bits_per_show, exp)
    records = gather(directory, bits, exp, source)
    if records is None:
        raise SystemExit(f"not all bits are written yet: run `python -m sim.material run{CLI_FLAG[exp.name]}` first")
    dirs = (directory, *((source,) if source else ()))
    verdicts = {bit_hash(r): cached_verdict(dirs, r) for r in records if r["premise"] is not None}
    return summarize(records, verdicts, read_answers(directory), on_gpu=tc.writer_on_gpu(directory), exp=exp)


def stored(exp: Exp = EXP33) -> dict | None:
    return cache.load(exp.name, params(exp=exp))


EXP_NUMBER = {EXP33.name: 33, EXP34.name: 34}
CLI_FLAG = {EXP33.name: "", EXP34.name: " --exp 34"}


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def table(res: dict, exp: Exp = EXP33) -> str:
    m1, m2 = res["metric1"], res["metric2"]
    b = m1["bar"]
    ci = "n/a" if b["ci95"] is None else f"{b['ci95'][0]:+.3f} to {b['ci95'][1]:+.3f}"
    f, c = m2["funnier"], m2["character"]
    w = c["wilson95"]
    x = exp.arm
    on = "material on" if exp.variant == app_material.FULL else "topic only"
    rows = ["| Check | Result | Bar |", "|---|---|---|",
            f"| Hardware share, {x} ({on}) | {pct(m1['share'][x]['share'])} of {m1['share'][x]['n']} bits | at most {MAX_SHARE_B:.0%} |",
            f"| Hardware share, A (material off) | {pct(m1['share']['A']['share'])} of {m1['share']['A']['n']} bits | beside the bar |",
            f"| Paired {x} - A over {b['n_paired']} slots | {cg.fmt(b['mean_diff'], '{:+.3f}')}, CI {ci} | CI entirely below 0 |",
            f"| Word-list share, A / {x} (cross-check) | {pct(m1['regex_share']['A']['share'])} / {pct(m1['regex_share'][x]['share'])} | beside the bar |",
            f"| Hardware share on character-topic / everyday bits ({x}) | {pct(m1['by_topic_kind'][x + '_character']['share'])} / "
            f"{pct(m1['by_topic_kind'][x + '_everyday']['share'])} | beside the bar |",
            f"| (a) Funnier: {x} wins | {f[x + '_wins']} of {m2['asked']} (A {f['A_wins']}, ties {f['ties']}; {m2['answered']} answered) | at least {FUNNY_WINS} |",
            f"| (b) More like a specific character: {x} wins | {c[x + '_wins']} of {m2['asked']} (A {c['A_wins']}, ties {c['ties']}), Wilson 95% "
            f"{'n/a' if w is None else f'{w[0]:.0%} to {w[1]:.0%}'} | beside the bar |",
            f"| Verdict | {res['verdict']} | both bars |"]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("run", "review", "score"))
    ap.add_argument("--exp", choices=tuple(EXPS), default="33", help="33 (topic and two facts, the default) or 34 (topic only)")
    ap.add_argument("--rerun", action="store_true", help="run: recompute from the run folder even if a record exists")
    ap.add_argument("--record", action="store_true", help="score: save the result as the experiment's record")
    ap.add_argument("--env-file")
    a = ap.parse_args()
    exp = EXPS[a.exp]
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file, override=True)
    os.environ.setdefault("JOKE_WRITER_PROVIDER", "ollama")
    if a.command == "run":
        res = cache.run_or_load(exp.name, lambda **p: run(exp=exp, **p), rerun=a.rerun, **params(exp=exp))
    elif a.command == "review":
        directory = run_dir(exp=exp)
        records = gather(directory, slots(seed=exp.slot_seed), exp, a_dir(N_SHOWS, BITS_PER_SHOW, exp))
        if records is None:
            raise SystemExit(f"run `python -m sim.material run{CLI_FLAG[exp.name]}` first")
        done = review(directory, records, exp=exp)
        print(f"{done} of {len(pairs(records, exp=exp))} pairs answered; now `score{CLI_FLAG[exp.name]} --record`.")
        return
    else:
        res = score(exp=exp)
        if a.record:
            cache.save(exp.name, params(exp=exp), res)
    print(table(res, exp), flush=True)


if __name__ == "__main__":
    main()
