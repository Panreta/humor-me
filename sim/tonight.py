"""Exp 35 (D-119): does tonight's context (a headline as the topic, one optional fact, the market line) make a bit funnier and
more "of tonight", and does the gist filter hand the operator material they would approve?

Usage (`ANTHROPIC_API_KEY` set for the gist step; Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192` for `run`):
    python -m sim.tonight fetch [--env-file backend/.env]   # today's headlines + market, gist and filter, then you approve each gist
    python -m sim.tonight run [--env-file backend/.env]     # 10 pairs, both arms with the live writer -> data/tonight/exp35/
    python -m sim.tonight review                            # you compare the 10 pairs, sides hidden and randomized
    python -m sim.tonight score [--record]                  # the numbers with your answers (no model calls) -> reports/metrics/tonight.json

`fetch` runs the same pipeline the show's `GET /tonight` runs (`app.tonight.build`): FRED's S&P 500 daily CSV and Google News
top stories; one Claude Haiku call writes one-sentence gists (countries, states, big cities, companies, products, holidays, sports teams or events and public figures in the headline may keep their names) and drops death, disaster, crime and war; a rule-level
name check; the safety check. Then for each gist that survived you answer (1) approve for the show? and (2) does it contain
a name that should not be there? and you may type 0 or 1 line about tonight (`MAX_TONIGHT_FACTS`, safety-checked). One day of headlines: a small
sample, and it is today's, so the fetch cannot be repeated later. The answers are saved in `fetch.json`.

Bar 1 (the filter, gates, set by Claude, confirmed by the user 2026-11-26): of the gists the filter kept, you reject at most 20% (a
rejection is "not approved", a flagged name counts), AND 0 kept gists contain a name that should not be there (you flag them: a person you did not approve, a private person, a name not in the headline; since D-120 an approved public figure is not flagged). Reported with a
Wilson 95% interval for the reject share. With a handful of gists the interval is wide: a pass says "not clearly bad".

`run`: 10 pairs. For each approved gist (cycled when there are fewer than 10) one bit is written twice with the live writer
(`generate.writer_client()`, bonsai on Ollama, one call at a time) in a seeded mode and room. Arm A is the live prompt with no
tonight context. Arm B is the live prompt plus tonight: the gist as the bit's topic, your fact (if any) and the market line (if
fetched), as the show sends them (`ThreadGenerator.set_tonight`, bit 2). The thread played is thread 0.

`review`: left/right blind, seeded. You answer (a) which is funnier and (b) which feels more like it is happening tonight.
Bar 2 (gates, set by Claude, confirmed by the user 2026-11-26): B wins at least 5 of 10 on (a); a tie is not a win. (b) is reported with a
Wilson 95% interval, no bar. 10 pairs only detect large differences: with a true 70% preference for B, 5 wins is reached
about 95% of the time, and under a coin flip it is still reached about 62% of the time, so the gate is "not clearly worse".

Adopt (flip the `TONIGHT` default on, same PR) only when both bars pass. No `audience.yaml` value is involved: no sim room
reward is read. Generated text, headlines and your answers live only under `data/tonight/exp35/` (gitignored); the metrics
hold numbers.

Code changes elsewhere (the generator, the paragraph) change the digest: rerun with `--rerun` after one.
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import generate, llm, safety  # noqa: E402
from app import tonight as app_tonight  # noqa: E402
from sim import cache  # noqa: E402
from sim import thread_coherence as tc  # noqa: E402
from sim.blind import plain  # noqa: E402
from sim.env import AGES, ENERGIES, VENUES  # noqa: E402

NAME = "tonight"
ARMS = ("A", "B")  # A: no tonight context, B: tonight context
N_PAIRS = 10
SEED = "exp35"
PAIR_SEED = 33
BIT = 2  # a headline bit: ThreadGenerator draws the gist for bits where bit % 4 == 2
MAX_REJECT_SHARE = 0.20
FUNNY_WINS = 5
DATA = app_tonight.DATA / "exp35"


def fetch_path(directory: pathlib.Path | None = None) -> pathlib.Path:
    return (directory or DATA) / "fetch.json"


def read_fetch(directory: pathlib.Path | None = None) -> dict | None:
    return tc._read(fetch_path(directory))


# --- fetch: today's material and your answers --------------------------------------------------------------------

def ask(read, prompt: str) -> bool:
    while True:
        a = read(prompt).strip().lower()
        if a in ("y", "n"):
            return a == "y"


def approve(built: dict, read=input, out=print, safety_client=None) -> dict:
    """Walks you through the gists the filter kept (approve? contains a name?) and one optional typed line. Returns the
    `fetch.json` record: the filter's counts, today's market, every answer, and the approved set."""
    answers = []
    kept = built["gists"]
    for n, gist in enumerate(kept, 1):
        out(f"\nGist {n} of {len(kept)}: {gist}")
        if names := built.get("people", {}).get(gist):  # the question below calls these "shown above"
            out(f"Public figures it may name: {', '.join(names)}")
        ok = ask(read, "Approve it for the show? [y/n] ")
        named = ask(read, "Does it contain a name that should not be there? (a person you did not approve, a private person, a name not in the headline; countries, states, cities, companies, products and the public figures shown above are fine) [y/n] ")
        answers.append({"gist": gist, "approved": ok, "name": named})
    fact = None
    if app_tonight.MAX_TONIGHT_FACTS:
        typed = app_tonight.clean_fact(read(f"\nOne line about tonight, or press Enter for none (at most {app_tonight.MAX_FACT_CHARS} characters): "))
        if typed:
            verdict = app_tonight.vet([typed], safety_client)[typed]
            out("Kept." if verdict.ok else f"The safety check refused it ({verdict.reason}); it is left out.")
            fact = typed if verdict.ok else None
    return {"date": built["date"], "fetched_at": built["fetched_at"], "sources": built["sources"], "errors": built["errors"],
            "market": built["market"], "report": built["report"], "keep": built.get("keep", {}), "people": built.get("people", {}), "answers": answers,
            "facts": [fact] if fact else [], "approved": [a["gist"] for a in answers if a["approved"]]}


def fetch(read=input, out=print, directory: pathlib.Path | None = None, build_fn=None, safety_client=None) -> dict:
    """Fetch today's headlines and market, filter them, ask you, save. `build_fn` is a test's stand-in for the network."""
    if build_fn is None:
        def one():
            return llm.from_env(generate.FILTER_TIMING[0], role="safety_check", retries=generate.FILTER_RETRIES,
                                deadline_s=generate.FILTER_TIMING[1])
        client = one()
        built = app_tonight.build(app_tonight.http_get, client, client)
        safety_client = client
    else:
        built = build_fn()
    for e in built["errors"]:
        out(f"Fetch problem: {e}")
    if not built["gists"]:
        raise SystemExit("no gists survived the filter today, so there is nothing to approve: try again later or fix the fetch above")
    rec = approve(built, read, out, safety_client)
    tc._write(fetch_path(directory), rec)
    return rec


# --- bar 1 -------------------------------------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def bar1(rec: dict) -> dict:
    answers = rec["answers"]
    kept = len(answers)
    rejected = sum(not a["approved"] for a in answers)
    named = sum(a["name"] for a in answers)
    share = rejected / kept if kept else None
    rep = rec["report"]
    return {"fetched": rep["fetched"], "dropped_by_category": rep["dropped_by_category"], "dropped_no_gist": rep["dropped_no_gist"],
            "rejected_by_name_check": rep["rejected_by_name_check"], "rejected_by_safety": rep["rejected_by_safety"],
            "kept": kept, "approved": kept - rejected, "rejected_by_user": rejected, "reject_share": share,
            "reject_wilson95": wilson(rejected, kept), "gists_with_a_name": named, "market_fetched": rec["market"] is not None,
            "facts_typed": len(rec["facts"]), "date": rec["date"], "bar": MAX_REJECT_SHARE,
            "passes": kept > 0 and share <= MAX_REJECT_SHARE + 1e-12 and named == 0}


# --- the pairs ---------------------------------------------------------------------------------------------------

def slots(rec: dict, n: int = N_PAIRS, seed: str = SEED) -> list[dict]:
    """n pairs: the approved gists cycled, each with a seeded mode and room, the same for both arms."""
    approved = rec["approved"]
    if not approved:
        raise SystemExit("you approved no gists in `fetch`, so there is nothing to write: rerun `fetch`")
    modes = tuple(generate.MODES)
    out = []
    for i in range(n):
        r = random.Random(f"{seed}:room:{i}")
        out.append({"id": f"p{i:02d}", "gist": approved[i % len(approved)], "mode": random.Random(f"{seed}:mode:{i}").choice(modes),
                    "context": {"venue": r.choice(VENUES), "energy": r.choice(ENERGIES), "age": r.choice(AGES)}})
    return out


def digest(rec: dict, n: int = N_PAIRS, seed: str = SEED) -> str:
    """What the written bits are made of: the live prompt, the tonight paragraph, today's approved material, the slots, the
    writer and its settings. A change starts a new run folder."""
    parts = [generate.PROMPT_ID, generate.PROMPTS[generate.PROMPT_ID], app_tonight.PARAGRAPH, app_tonight.SUFFIX, generate.TEENS_RULE,
             json.dumps(generate.MODES, sort_keys=True), json.dumps(generate.COUNTED_REPLY_SCHEMA, sort_keys=True),
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.RETRIES,
                  generate.WRITER_CONTEXT, generate.HANG_GRACE_S, llm.MIN_ATTEMPT_S)),
             tc.writer_model(), json.dumps(llm.OLLAMA_NO_THINK, sort_keys=True), json.dumps(slots(rec, n, seed), sort_keys=True),
             json.dumps(rec["facts"]), json.dumps((rec["market"] or {}).get("line")), seed, str(n)]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8]


def run_dir(rec: dict, n: int = N_PAIRS, seed: str = SEED, root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or DATA) / f"run-{digest(rec, n, seed)}"


def params(rec: dict, n: int = N_PAIRS, seed: str = SEED) -> dict:
    return {"n_pairs": n, "seed": seed, "digest_": digest(rec, n, seed)}


def bit_path(directory: pathlib.Path, pair: str, arm: str) -> pathlib.Path:
    return directory / "bits" / f"{pair}-{arm}.json"


def new_generator(client, arm: str, rec: dict, slot: dict):
    gen = generate.ThreadGenerator(client, prompt_id=generate.PROMPT_ID)
    if arm == "B":
        market = (rec["market"] or {}).get("line")
        gen.set_tonight(app_tonight.TonightContext(rec["facts"], market, [slot["gist"]]))
    return gen


def write_bit(client, rec: dict, slot: dict, arm: str) -> dict:
    """One bit through the show's own path (bit 2, so arm B's headline is drawn); thread 0 is the one played."""
    gen = new_generator(client, arm, rec, slot)
    bad0 = client.bad_replies
    start = time.perf_counter()
    out = {"pair": slot["id"], "arm": arm, "mode": slot["mode"], "gist": slot["gist"] if arm == "B" else None, "premise": None,
           "lines": [], "lost": [], "failure": None}
    try:
        batches = [gen.generate(slot["mode"], BIT, j, slot["context"]) for j in range(generate.JOKES)]
        out["lines"] = [b.texts[0] for b in batches]
        out["premise"] = gen._bits[BIT][2][0]
        out["lost"] = list(batches[0].threads_lost)
    except Exception as e:  # noqa: BLE001 - a bit with no thread is a result, not a crash
        out["failure"] = type(e).__name__
        out["lost"] = list(getattr(e, "threads_lost", [])) or [f"{type(e).__name__}: {e}"[:100]]
    out.update(wall_s=time.perf_counter() - start, bad_replies=client.bad_replies - bad0)
    return out


def generate_bits(client, rec: dict, pairs_: list[dict], directory: pathlib.Path, http=None) -> list[dict]:
    """Writes every missing (pair, arm) bit, resumable, one at a time. The arm order alternates by pair so a slow stretch of
    the machine does not fall on one arm."""
    todo = [(s, a) for s in pairs_ for a in ARMS if tc._read(bit_path(directory, s["id"], a)) is None]
    if todo:
        probe = generate.ThreadGenerator(client)
        generate.warm_up(probe, http) if http else generate.warm_up(probe)
    out = {}
    for i, s in enumerate(pairs_):
        for arm in (ARMS if i % 2 == 0 else reversed(ARMS)):
            path = bit_path(directory, s["id"], arm)
            got = tc._read(path)
            if got is None:
                got = write_bit(client, rec, s, arm)
                tc._write(path, got)
            out[(s["id"], arm)] = got
    if todo:
        share, context, others = generate.placement(client, http) if http else generate.placement(client)
        tc._write(tc.placement_path(directory), {"gpu_share": share, "context_length": context, "others_loaded": others})
    return [out[(s["id"], a)] for s in pairs_ for a in ARMS]


def gather(directory: pathlib.Path, pairs_: list[dict]) -> list[dict] | None:
    recs = [tc._read(bit_path(directory, s["id"], a)) for s in pairs_ for a in ARMS]
    return None if any(r is None for r in recs) else recs


# --- the blind review and bar 2 ----------------------------------------------------------------------------------

def review_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "review.json"


def by_pair(records: list[dict]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for r in records:
        out.setdefault(r["pair"], {})[r["arm"]] = r
    return out


def pairs_to_review(records: list[dict], seed: int = PAIR_SEED) -> list[dict]:
    """Every pair where both arms delivered a bit, in a seeded order; `left` says which arm is shown on the left."""
    rng = random.Random(seed)
    ok = sorted(p for p, arms in by_pair(records).items() if all(a in arms and arms[a]["premise"] is not None for a in ARMS))
    rng.shuffle(ok)
    return [{"pair": p, "left": rng.choice(ARMS)} for p in ok]


def read_answers(directory: pathlib.Path) -> dict[str, dict]:
    return (tc._read(review_path(directory)) or {}).get("answers", {})


def ask_side(read, prompt: str) -> str:
    while True:
        a = read(prompt).strip().lower()
        if a in ("l", "r", "t", "q"):
            return a


def shown(r: dict) -> str:
    return f"Premise: {plain(r['premise'])}\n" + "\n".join(f"{i}. {plain(x)}" for i, x in enumerate(r["lines"], 1))


def review(directory: pathlib.Path, records: list[dict], read=input, out=print) -> int:
    """Asks about each pair not answered yet, saving after every answer. You see the premise and lines only: not the arm, the
    headline or the facts. l = left, r = right, t = tie (a tie is not a win for B), q = stop for now."""
    drawn, slot = pairs_to_review(records), by_pair(records)
    answers = read_answers(directory)
    for n, p in enumerate(drawn, 1):
        if p["pair"] in answers:
            continue
        right = next(a for a in ARMS if a != p["left"])
        out(f"\nPair {n} of {len(drawn)}")
        for side, arm in (("LEFT", p["left"]), ("RIGHT", right)):
            out(f"--- {side}\n" + shown(slot[p["pair"]][arm]))
        funnier = ask_side(read, "(a) Which is funnier? [l/r/t/q] ")
        if funnier == "q":
            break
        tonight = ask_side(read, "(b) Which feels more like it is happening tonight? [l/r/t/q] ")
        if tonight == "q":
            break
        answers[p["pair"]] = {"funnier": funnier, "tonight": tonight}
        tc._write(review_path(directory), {"answers": answers})
    return sum(p["pair"] in answers for p in drawn)


def metric2(records: list[dict], answers: dict[str, dict], n_pairs: int = N_PAIRS) -> dict:
    drawn = pairs_to_review(records)

    def tally(field: str) -> tuple[int, int, int]:
        won = lost = tied = 0
        for p in drawn:
            if (a := answers.get(p["pair"])) is None:
                continue
            side = a[field]
            if side == "t":
                tied += 1
            elif (side == "l") == (p["left"] == "B"):
                won += 1
            else:
                lost += 1
        return won, lost, tied
    fw, fl, ft = tally("funnier")
    tw, tl, tt = tally("tonight")
    asked, answered = len(drawn), fw + fl + ft
    return {"asked": asked, "answered": answered, "complete": asked > 0 and answered == asked, "short": asked < n_pairs,
            "funnier": {"B_wins": fw, "A_wins": fl, "ties": ft, "bar": FUNNY_WINS, "passes": fw >= FUNNY_WINS},
            "tonight": {"B_wins": tw, "A_wins": tl, "ties": tt, "B_share": tw / (tw + tl + tt) if tw + tl + tt else None,
                        "wilson95": wilson(tw, tw + tl + tt)}}


def decide(b1: dict, m2: dict) -> str:
    """`keep_off`: the filter bar failed (the pairs cannot rescue it), or B won fewer than 5 of 10 with every pair answered.
    `pending_review`: the filter passed and pairs are unanswered. `adopt`: both passed (5 wins cannot be undone by the rest)."""
    if not b1["passes"]:
        return "keep_off"
    if m2["funnier"]["passes"]:
        return "adopt"
    return "pending_review" if not m2["complete"] else "keep_off"


def summarize(rec: dict, records: list[dict], answers: dict, on_gpu: bool | None = None) -> dict:
    b1, m2 = bar1(rec), metric2(records, answers)
    arms = {}
    for a in ARMS:
        mine = [r for r in records if r["arm"] == a]
        arms[a] = {"bits": len(mine), "bits_delivered": sum(r["premise"] is not None for r in mine),
                   "threads_lost": sum(len(r["lost"]) for r in mine), "bad_replies": sum(r["bad_replies"] for r in mine)}
    return {"bar1_filter": b1, "bar2_pairs": m2, "arms": arms, "distinct_gists": len({r["gist"] for r in records if r["gist"]}),
            "writer_on_gpu_after_run": on_gpu, "verdict": decide(b1, m2)}


# --- running -----------------------------------------------------------------------------------------------------

def need_fetch(directory: pathlib.Path | None = None) -> dict:
    rec = read_fetch(directory)
    if rec is None:
        raise SystemExit("run `python -m sim.tonight fetch` first (it saves today's approved gists)")
    return rec


def run(n_pairs: int = N_PAIRS, seed: str = SEED, digest_: str = "") -> dict:
    rec = need_fetch()
    assert digest_ == digest(rec, n_pairs, seed), "the prompt, paragraph, today's approved material or the slots changed since this call was built"
    client = generate.writer_client()
    pairs_, directory = slots(rec, n_pairs, seed), run_dir(rec, n_pairs, seed)
    if gather(directory, pairs_) is None and not getattr(client, "local", False):
        raise SystemExit("Exp 35 writes with the local writer: set JOKE_WRITER_PROVIDER=ollama")
    records = generate_bits(client, rec, pairs_, directory)
    return summarize(rec, records, read_answers(directory), on_gpu=tc.writer_on_gpu(directory))


def score(directory_root: pathlib.Path | None = None) -> dict:
    """The numbers again from the run folder with your answers, no model call. Raises when a bit is missing."""
    rec = need_fetch(directory_root)
    pairs_, directory = slots(rec), run_dir(rec, root=directory_root)
    records = gather(directory, pairs_)
    if records is None:
        raise SystemExit("not all bits are written yet: run `python -m sim.tonight run` first")
    return summarize(rec, records, read_answers(directory), on_gpu=tc.writer_on_gpu(directory))


def stored(directory: pathlib.Path | None = None) -> dict | None:
    """The record when it matches the local fetch. The fetch is gitignored and `fetch` on a later day replaces it (more days
    for bar 1 is the next step), so without the record's own day here the record is returned as recorded: a clone, or the
    next day, still shows Exp 35. Same day but a new digest means the code changed since: None, rerun."""
    rec = read_fetch(directory)
    if rec is not None and (hit := cache.load(NAME, params(rec))) is not None:
        return hit
    path = cache.METRICS / f"{NAME}.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())["results"]
    return res if rec is None or rec.get("date") != res["bar1_filter"]["date"] else None


def pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def interval(w) -> str:
    return "n/a" if w is None else f"{w[0]:.0%} to {w[1]:.0%}"


def table(res: dict) -> str:
    b, m = res["bar1_filter"], res["bar2_pairs"]
    f, t = m["funnier"], m["tonight"]
    drops = b["dropped_by_category"]
    rows = ["| Check | Result | Bar |", "|---|---|---|",
            f"| Headlines fetched / dropped (death, disaster, crime, war) | {b['fetched']} / {drops['death']}, {drops['disaster']}, {drops['crime']}, {drops['war']} | beside the bar |",
            f"| Rejected by the name check / by the safety check | {b['rejected_by_name_check']} / {b['rejected_by_safety']} | beside the bar |",
            f"| Gists kept / approved by you | {b['kept']} / {b['approved']} | beside the bar |",
            f"| You rejected, Wilson 95% (one day, a small sample) | {pct(b['reject_share'])}, {interval(b['reject_wilson95'])} | at most {b['bar']:.0%} |",
            f"| Kept gists with a real name (you flagged) | {b['gists_with_a_name']} | 0 |",
            f"| (a) Funnier: B wins | {f['B_wins']} of {m['asked']} (A {f['A_wins']}, ties {f['ties']}; {m['answered']} answered) | at least {FUNNY_WINS} |",
            f"| (b) More like tonight: B wins | {t['B_wins']} of {m['asked']} (A {t['A_wins']}, ties {t['ties']}), Wilson 95% {interval(t['wilson95'])} | beside the bar |",
            f"| Verdict | {res['verdict']} | both bars |"]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("fetch", "run", "review", "score"))
    ap.add_argument("--rerun", action="store_true", help="run: recompute from the run folder even if a record exists")
    ap.add_argument("--record", action="store_true", help="score: save the result as the Exp 35 record")
    ap.add_argument("--env-file")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        if not pathlib.Path(a.env_file).is_file():  # load_dotenv ignores a missing file; in a worktree backend/.env is not there
            raise SystemExit(f"--env-file {a.env_file} not found; from a worktree use the main checkout's, e.g. ../../../backend/.env")
        load_dotenv(a.env_file, override=True)
    os.environ.setdefault("JOKE_WRITER_PROVIDER", "ollama")
    if a.command == "fetch":
        if read_fetch() is not None:
            print(f"Replacing the saved fetch from {read_fetch()['date']}: the old run folder stays but its digest no longer matches.")
        rec = fetch()
        print(f"\nSaved: {len(rec['approved'])} approved of {len(rec['answers'])} kept; {len(rec['facts'])} typed line(s).")
        return
    rec = need_fetch()
    if a.command == "run":
        res = cache.run_or_load(NAME, run, rerun=a.rerun, **params(rec))
    elif a.command == "review":
        records = gather(run_dir(rec), slots(rec))
        if records is None:
            raise SystemExit("run `python -m sim.tonight run` first")
        done = review(run_dir(rec), records)
        print(f"{done} of {len(pairs_to_review(records))} pairs answered; now `score --record`.")
        return
    else:
        res = score()
        if a.record:
            cache.save(NAME, params(rec), res)
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
