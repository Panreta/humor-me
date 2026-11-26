"""Exp 29 (D-113): which stage of a bit's wait is the biggest, so the right one gets fixed first?

Usage, after the live shows (seconds from the local logs, no model call, never any line of text):
    python -m sim.stage_timing <session id> [<session id> ...] [--dir data/sessions] [--record]

Name the ids of the shows to count (file names without `.jsonl`). With none it reads every log in the folder; shows
whose generator is the hand-written bank (`bank-v1`) are listed but never counted, because their stages are about
zero. Plain, it prints and saves nothing; `--record` writes `reports/metrics/stage_timing.json` (ids and seconds).
A log from before D-113 has no `prep` rows: listed with its waits, stage columns empty, not counted.

A bit's wait runs from the previous `bit_end` (`show_start` for bit 0) to its first `joke_played`'s audio start. The
`prep` row splits the part before the bit was ready into generate_s (the writer), safety_s (the filter), bank_s (the
hand-checked bank), other_s (what no timer covers: picking a thread, building plans), render_queue_s (waiting for the
serial voice behind a bridge or cached render) and render_s (the synthesis). When the voice cannot split the render
the whole wait is counted as render_s and the queue is empty. `bridge_overhang_s` on the first joke is what the room
waited after the bit was ready because a bridge line was still playing. Median and p95 are nearest-rank.

Revised after review and before any data was recorded (the first draft compared medians, which hide a stage that is
large on some bits and zero on most, and had no interval on its gate). The rule, fixed before any data:
  * Counted: bits 1 and up of counted shows. Bit 0 (cold model, the opener's renders, no bridge) is reported apart and
    does not decide. A bit is dropped when a pause overlaps its wait (the pause time would be read as a stage) or its
    first render failed (`first_render_failed`); the counts dropped are reported.
  * A stage's share is its total seconds over the total wait, over the counted bits: the mean seconds a fix could
    save per bit, so the shares add to about 1 (the rest is time before prep starts).
  * The stage with the largest share is fixed first when (1) its share is at least MIN_SHARE (25%) and (2) its share
    beats TIE_RATIO (1.1x) the runner-up's with the 95% bootstrap interval of `share - 1.1 x runner-up` above 0. The
    bootstrap resamples whole shows (bits in a show share a room and a mood), 2000 times, seeded. An interval that
    straddles 0 is "inconclusive", below 0 is "no one stage". At least MIN_SHOWS (2) counted shows and MIN_BITS (16)
    counted bits are needed, else "not enough data". With two or three shows the interval is wide and "inconclusive"
    is the likely outcome until more shows are recorded; that is the honest answer.
The question was raised by show 6c09c721b0bb (waits 9.8 to 17.2 s, writer about 3.5 s, Exp 26); it motivates the
question and is not the bar. The fixes it chooses between:
  bridge_overhang_s -> (a) end the bridge early or start the joke after the current line instead of letting the bridge
                       loop decide
  render_queue_s    -> (b) render the first joke ahead of bridge and cached renders
  safety_s          -> (c) safety-check each thread as it arrives, overlapping the filter with generation
  generate_s, render_s, bank_s, other_s -> none of (a) to (c): the writer (Exp 26) or the voice speed is the limit,
                       reported only.
A bit prepared on a right guess of its mode (D-115, `prep.speculative`) is marked and left out of the counted bits and
the shares: its stage seconds overlap the previous bit's jokes, so they say nothing about its wait. Its wait and bridge
count still show in the per-show columns (the `basis`), which is the before/after measure for D-115: bits 1 and up,
wait p50/p95 and bridges per bit, for the shows 6c09c721b0bb and 0e102d77d795 (before) against shows after the change.
The 25%, the 1.1x, the 16-bit and 2-show minimums, and the share definition are judgment calls, not measured.
"""
import argparse
import hashlib
import json
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.log import SESSIONS, read_session  # noqa: E402
from app.pauses import intervals, overlaps  # noqa: E402
from sim import cache  # noqa: E402
from sim.compare_generators import fmt, percentile  # noqa: E402

NAME = "stage_timing"
MIN_SHARE = 0.25
TIE_RATIO = 1.1
MIN_BITS = 16
MIN_SHOWS = 2
BOOTSTRAP = 2000
SEED = 29
STAGES = ("generate_s", "safety_s", "bank_s", "other_s", "render_queue_s", "render_s", "bridge_overhang_s")
FIX = {"bridge_overhang_s": "(a) end the bridge early / start the joke after the current line",
       "render_queue_s": "(b) render the first joke ahead of bridge and cached renders",
       "safety_s": "(c) safety-check each thread as it arrives, overlapping generation"}


def load(directory: pathlib.Path, sessions: list[str] | None) -> tuple[list[tuple[str, list[dict]]], list[str]]:
    """(shows as (session id, rows), oldest first; the ids that did not read)."""
    paths = [pathlib.Path(directory) / f"{s}.jsonl" for s in sessions] if sessions else sorted(pathlib.Path(directory).glob("*.jsonl"))
    shows, bad = [], []
    for path in paths:
        try:
            rows = read_session(path)
            if not rows or rows[0]["event"] != "show_start":
                raise ValueError("no show_start")
            shows.append((path.stem, rows))
        except (OSError, ValueError):
            bad.append(path.stem)
    shows.sort(key=lambda s: s[1][0]["at"])
    return shows, bad


def bits_of(rows: list[dict]) -> list[dict]:
    """One dict per bit that played a line: its wait, bridge count and, when logged, the prep stages and overhang.
    `paused` is true when a pause overlaps the wait; `failed` when the bit's first render failed."""
    start = rows[0]["at"]
    prep = {r["bit"]: r for r in rows if r["event"] == "prep"}
    first: dict[int, dict] = {}
    for r in rows:
        if r["event"] == "joke_played":
            first.setdefault(r["bit"], r)
    ends = {r["bit"]: r["at"] for r in rows if r["event"] == "bit_end"}
    pauses = intervals(rows)
    out = []
    for bit in sorted(first):
        began = start if bit == 0 else ends.get(bit - 1)
        if began is None:
            continue
        p, played = prep.get(bit), first[bit]
        queue = p.get("render_queue_s") if p else None
        render = None if p is None else (p.get("render_s") if queue is not None else p["render_wait_s"])  # no split: all render
        out.append({"bit": bit, "wait_s": max(0.0, played["audio_start_at"] - began),
                    "bridges": sum(r["event"] == "bridge" and r["bit"] == bit for r in rows),
                    "prepped": p is not None, "failed": bool(p and p.get("first_render_failed")),
                    "speculative": bool(p and p.get("speculative")),
                    "paused": overlaps(began, played["audio_start_at"], pauses),
                    "generate_s": p and p["generate_s"], "safety_s": p and p["safety_s"], "bank_s": p and p.get("bank_s"),
                    "other_s": p and p.get("other_s"), "render_queue_s": queue, "render_s": render,
                    "bridge_overhang_s": played.get("bridge_overhang_s")})
    return out


def stats(values: list) -> dict:
    xs = [v for v in values if v is not None]
    return {"n": len(xs), "p50": statistics.median(xs) if xs else None, "p95": percentile(xs, 95)}


def counted(bits: list[dict]) -> list[dict]:
    """The bits that decide: prepped, not bit 0, no pause in the wait, first render not failed, not prepared on a
    guess (D-115: its stage seconds ran during the previous bit's jokes, so they are not part of its wait)."""
    return [b for b in bits if b["prepped"] and b["bit"] >= 1 and not b["paused"] and not b["failed"]
            and not b["speculative"]]


def totals(bits: list[dict]) -> dict:
    """Seconds summed over bits; an absent stage (the voice could not split the queue) adds 0."""
    return {"n": len(bits), "wait": sum(b["wait_s"] for b in bits),
            **{k: sum(b[k] or 0.0 for b in bits) for k in STAGES}}


def guess_stats(rows: list[dict]) -> dict:
    """What the next-bit guesses of a show did (D-115), from its `prediction` rows: how many, how many right, how many used,
    the same for guesses whose copy was taught the partial reward (`learned`; the sim's bar is for these), and the safety
    requests wasted on wrong ones. The live hit rate can be lower than Exp 31's: a bit with fewer than 3 labels at the
    guess falls back to the model as it is, which the sim found right far less often."""
    g = [r for r in rows if r["event"] == "prediction"]
    learned = [r for r in g if r.get("learned")]
    return {"n": len(g), "hits": sum(r["hit"] for r in g), "adopted": sum(bool(r.get("adopted", r["hit"])) for r in g),
            "learned_n": len(learned), "learned_hits": sum(r["hit"] for r in learned),
            "wasted_prefetch": sum(bool(r.get("spec_prefetched")) and not r.get("adopted", r["hit"]) for r in g)}


def show_report(session: str, rows: list[dict]) -> dict:
    bits = bits_of(rows)
    generator = rows[0]["generator_id"]
    use = counted(bits)
    included = bool(use) and not generator.startswith("bank")
    basis = use if any(b["prepped"] for b in bits) else [b for b in bits if b["bit"] >= 1]
    # the wait and bridge columns keep the bits prepared on a guess: that wait is what D-115 shortens
    waiting = list(basis) + [b for b in bits if b["speculative"] and b["bit"] >= 1 and not b["paused"] and not b["failed"]]
    return {"session": session, "room": rows[0]["room"], "generator_id": generator, "included": included,
            "bits": len(bits), "bits_with_prep": sum(b["prepped"] for b in bits), "bits_counted": len(use),
            "dropped_paused": sum(b["prepped"] and b["bit"] >= 1 and b["paused"] for b in bits),
            "dropped_render_failed": sum(b["prepped"] and b["bit"] >= 1 and b["failed"] for b in bits),
            "speculative_bits": sum(b["speculative"] for b in bits), "guesses": guess_stats(rows),
            "wait_s": stats([b["wait_s"] for b in waiting]), "bridges": stats([b["bridges"] for b in waiting]),
            **{k: stats([b[k] for b in basis]) for k in STAGES},
            "sums": totals(use) if included else None}


def shares(sums: list[dict]) -> dict:
    wait = sum(s["wait"] for s in sums)
    return {k: (sum(s[k] for s in sums) / wait if wait else 0.0) for k in STAGES}


def _gap(sums: list[dict], stage: str, other: str) -> float | None:
    sh = shares(sums)
    return None if not sum(s["wait"] for s in sums) else sh[stage] - TIE_RATIO * sh[other]


def decide(sums: list[dict]) -> dict:
    """The fixed rule on the counted shows' totals: {"outcome", "stage", "fix", "share", "runner_up", "ci"}."""
    nothing = {"stage": None, "fix": None, "share": None, "runner_up": None, "ci": None}
    n_bits, wait = sum(s["n"] for s in sums), sum(s["wait"] for s in sums)
    if len(sums) < MIN_SHOWS or n_bits < MIN_BITS or wait <= 0:
        return {"outcome": "not_enough_data", **nothing}
    ranked = sorted(shares(sums).items(), key=lambda kv: kv[1], reverse=True)
    (stage, top), (other, second) = ranked[0], ranked[1]
    rng, draws = random.Random(SEED), []
    for _ in range(BOOTSTRAP):
        gap = _gap([sums[rng.randrange(len(sums))] for _ in sums], stage, other)
        if gap is not None:
            draws.append(gap)
    draws.sort()
    ci = [draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]] if draws else None
    base = {"stage": stage, "share": top, "runner_up": second, "ci": ci, "fix": None}
    if top < MIN_SHARE or ci is None or ci[1] < 0:
        return {"outcome": "no_one_stage", **base}
    if ci[0] <= 0:
        return {"outcome": "inconclusive", **base}
    return {"outcome": "fix" if stage in FIX else "no_candidate_fix", **base, "fix": FIX.get(stage)}


def run(sessions: list[str] | None = None, directory: str | pathlib.Path = SESSIONS, key: str = "") -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    shows, bad = load(pathlib.Path(directory), sessions)
    reports = [show_report(sid, rows) for sid, rows in shows]
    use = [(sid, rows) for (sid, rows), rep in zip(shows, reports) if rep["included"]]
    bits = [b for _, rows in use for b in counted(bits_of(rows))]
    zero = [b for _, rows in use for b in bits_of(rows) if b["prepped"] and b["bit"] == 0]
    sums = [r["sums"] for r in reports if r["included"]]
    keys = ("n", "hits", "adopted", "learned_n", "learned_hits", "wasted_prefetch")
    guesses = {k: sum(r["guesses"][k] for r in reports if r["included"]) for k in keys}
    pooled = {"guesses": guesses, "shows_counted": len(use), "bits_counted": len(bits), "wait_s": stats([b["wait_s"] for b in bits]),
              "shares": shares(sums) if sums else {k: None for k in STAGES},
              **{k: stats([b[k] for b in bits]) for k in STAGES}}
    bit0 = {"bits": len(zero), "wait_s": stats([b["wait_s"] for b in zero]), **{k: stats([b[k] for b in zero]) for k in STAGES}}
    return {"shows": reports, "unreadable_logs": bad, "pooled": pooled, "bit0": bit0, "min_share": MIN_SHARE,
            "tie_ratio": TIE_RATIO, "min_bits": MIN_BITS, "min_shows": MIN_SHOWS, "decision": decide(sums)}


def digest(sessions: list[str] | None, directory: str | pathlib.Path = SESSIONS) -> str:
    """Stand-in for the files read, so a new or edited log recomputes."""
    h = hashlib.sha256()
    names = sorted(sessions) if sessions else sorted(p.stem for p in pathlib.Path(directory).glob("*.jsonl"))
    for name in names:
        path = pathlib.Path(directory) / f"{name}.jsonl"
        h.update(name.encode() + (path.read_bytes() if path.exists() else b""))
    return h.hexdigest()[:16]


def record(sessions: list[str] | None = None, directory: str | pathlib.Path = SESSIONS) -> dict:
    return cache.run_or_load(NAME, lambda key: run(sessions, directory, key), rerun=True, key=digest(sessions, directory))


def stored() -> dict | None:
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def cell(s: dict) -> str:
    return f"{fmt(s['p50'], '{:.1f}')} / {fmt(s['p95'], '{:.1f}')}"


def table(res: dict) -> str:
    head = ["Show", "Generator", "Bits counted of 1+ (paused, render failed dropped)", "Bridges per bit p50", "Wait", "Generate",
            "Safety", "Bank", "Other", "Render queue", "Render", "Bridge overhang"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in res["shows"]:
        mark = "" if s["included"] else " (not counted)"
        rows.append(f"| {s['session']} {s['room']}{mark} | {s['generator_id']} | {s['bits_counted']} ({s['dropped_paused']}, "
                    f"{s['dropped_render_failed']}) | {fmt(s['bridges']['p50'], '{:.0f}')} | {cell(s['wait_s'])} | "
                    + " | ".join(cell(s[k]) for k in STAGES) + " |")
    p, z = res["pooled"], res["bit0"]
    rows.append(f"| all counted, bits 1+ (decides) | | {p['bits_counted']} | | {cell(p['wait_s'])} | "
                + " | ".join(cell(p[k]) for k in STAGES) + " |")
    rows.append(f"| bit 0 only (reported, does not decide) | | {z['bits']} | | {cell(z['wait_s'])} | "
                + " | ".join(cell(z[k]) for k in STAGES) + " |")
    share = " | ".join("-" if p["shares"][k] is None else f"{p['shares'][k]:.0%}" for k in STAGES)
    rows.append(f"| share of the total wait (the rule's number) | | | | | {share} |")
    g = res["pooled"].get("guesses") or {"n": 0}
    guess = ""
    if g["n"]:
        guess = (f"\n\nNext-bit guesses (D-115), counted shows: {g['hits']}/{g['n']} right, {g['adopted']} used; of the guesses "
                 f"made on 3 labels (the sim's case) {g['learned_hits']}/{g['learned_n']} right; {g['wasted_prefetch']} safety "
                 "requests wasted on wrong ones. Exp 31's sim hit rate was about 0.7; fewer labels give fewer right guesses.")
    return "Seconds, p50 / p95; the last row is each stage's total over the total wait.\n\n" + "\n".join(rows) + guess


def verdict(res: dict) -> str:
    d = res["decision"]
    if d["outcome"] == "not_enough_data":
        return f"Biggest stage: NOT DECIDED, needs {res['min_shows']} counted shows and {res['min_bits']} counted bits (bits 1+)"
    share = f"{d['share']:.0%} of the wait, runner-up {d['runner_up']:.0%}, 95% interval of share - {res['tie_ratio']}x runner-up {d['ci'][0]:+.2f} to {d['ci'][1]:+.2f}" if d["ci"] else f"{d['share']:.0%} of the wait"
    if d["outcome"] == "no_one_stage":
        return f"Biggest stage: no one stage ({d['stage']} leads at {share}; needs {res['min_share']:.0%} and an interval above 0)"
    if d["outcome"] == "inconclusive":
        return f"Biggest stage: INCONCLUSIVE ({d['stage']} leads at {share}); record more shows"
    if d["outcome"] == "no_candidate_fix":
        return f"Biggest stage: {d['stage']} ({share}); none of fixes (a) to (c) touches it"
    return f"Biggest stage: {d['stage']} ({share}); fix {d['fix']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="*", help="session ids (file names without .jsonl); default: every log in --dir")
    ap.add_argument("--dir", default=str(SESSIONS))
    ap.add_argument("--record", action="store_true", help="save as the Exp 29 evidence (replaces the last)")
    a = ap.parse_args()
    res = record(a.sessions or None, a.dir) if a.record else run(a.sessions or None, a.dir)
    print(table(res), "", verdict(res), sep="\n")
    print(f"recorded to reports/metrics/{NAME}.json" if a.record else "not recorded (add --record)")


if __name__ == "__main__":
    main()
