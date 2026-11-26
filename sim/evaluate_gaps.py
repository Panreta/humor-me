"""Exp 31 live check (D-115): did guessing the next bit's mode shorten the wait before a bit's first joke?

Usage, after the live shows (seconds from the local logs, no model call, never any line of text):
    python -m sim.evaluate_gaps --before <id> ... --after <id> ... [--dir data/sessions] [--record]

`--before` are shows without the guess, `--after` shows with it. Plain, it prints and saves nothing; `--record` writes
`reports/metrics/live_gaps.json` (ids and numbers only).

Per show: for bits 1 and up, the wait from the previous `bit_end` to the first joke's audio start (the same measure as
`sim.stage_timing`; a bit whose wait overlaps a pause, or whose first render failed, is dropped), the total of those
waits, how many were zero (at most ZERO_S), the bridge lines played (all `bridge` rows: cached lines in shows before
D-112, filler after) and the show's rating. For shows after the change also the guess hit rate with its Wilson 95%
interval, the guesses made on 3 labels and the safety requests wasted on wrong guesses. A show that did not finish is
listed under `excluded` and counts nowhere. A show from before D-113 has no `prep` rows; its waits come from `bit_end`
and joke timestamps, which is all this reads. The wait starts at `bit_end`, which is 0.3 to 2.5 s after the last joke's
audio in both sets.

No bar: the before and after shows are different shows (not paired seeds), n is small, and the audience is the made-up
sim one. Wait times depend on `audience.yaml` only through when the reactions arrive. Read it as a live sanity check
of Exp 31's sim claim, not as a test.
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.log import SESSIONS  # noqa: E402
from sim import cache  # noqa: E402
from sim.compare_generators import fmt  # noqa: E402
from sim.stage_timing import bits_of, guess_stats, load, stats  # noqa: E402

Z = 1.96

NAME = "live_gaps"
ZERO_S = 0.1


def wilson(k: int, n: int) -> list[float] | None:
    """95% Wilson interval of a rate; guesses within a show are not independent, so it is optimistic."""
    if not n:
        return None
    p, d = k / n, 1 + Z * Z / n
    mid, half = (p + Z * Z / (2 * n)) / d, Z * ((p * (1 - p) / n + Z * Z / (4 * n * n)) ** 0.5) / d
    return [mid - half, mid + half]


def show_report(session: str, rows: list[dict]) -> dict:
    gaps = [b for b in bits_of(rows) if b["bit"] >= 1 and not b["paused"] and not b["failed"]]
    ends = [r["status"] for r in rows if r["event"] == "show_end"]
    ratings = [r["value"] for r in rows if r["event"] == "rating"]
    return {"session": session, "room": rows[0]["room"], "generator_id": rows[0]["generator_id"],
            "status": ends[-1] if ends else "none", "rating": ratings[-1] if ratings else None,
            "gaps": [round(b["wait_s"], 3) for b in gaps], "total_wait_s": sum(b["wait_s"] for b in gaps), "wait_s": stats([b["wait_s"] for b in gaps]),
            "zero_waits": sum(b["wait_s"] <= ZERO_S for b in gaps),
            "bridge_lines": sum(r["event"] == "bridge" for r in rows), "guesses": guess_stats(rows)}


def pool(shows: list[dict]) -> dict:
    waits = [w for s in shows for w in s["gaps"]]
    keys = ("n", "hits", "adopted", "learned_n", "learned_hits", "wasted_prefetch")
    out = {"shows": len(shows), "gaps": len(waits), "wait_s": stats(waits),
            "total_wait_per_show_s": sum(s["total_wait_s"] for s in shows) / len(shows) if shows else None,
            "zero_waits": sum(s["zero_waits"] for s in shows), "bridge_lines": sum(s["bridge_lines"] for s in shows),
            "guesses": {k: sum(s["guesses"][k] for s in shows) for k in keys}}
    out["hit_ci95"] = wilson(out["guesses"]["hits"], out["guesses"]["n"])
    return out


def run(before: list[str], after: list[str], directory: str | pathlib.Path = SESSIONS, key: str = "") -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    shows, bad = load(pathlib.Path(directory), before + after)
    by = {sid: show_report(sid, rows) for sid, rows in shows}
    excluded = [{"session": s["session"], "status": s["status"]} for s in by.values() if s["status"] != "finished"]
    b = [by[s] for s in before if s in by and by[s]["status"] == "finished"]
    a = [by[s] for s in after if s in by and by[s]["status"] == "finished"]
    return {"before": b, "after": a, "excluded": excluded, "before_pooled": pool(b), "after_pooled": pool(a), "unreadable_logs": bad,
            "zero_s": ZERO_S}


def digest(ids: list[str], directory: str | pathlib.Path = SESSIONS) -> str:
    h = hashlib.sha256()
    for sid in sorted(ids):
        path = pathlib.Path(directory) / f"{sid}.jsonl"
        h.update(sid.encode() + (path.read_bytes() if path.exists() else b"missing"))
    return h.hexdigest()[:16]


def record(before: list[str], after: list[str], directory: str | pathlib.Path = SESSIONS) -> dict:
    return cache.run_or_load(NAME, lambda before, after, key: run(before, after, directory, key), rerun=True,
                             before=before, after=after, key=digest(before + after, directory))


def stored() -> dict | None:
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def table(res: dict) -> str:
    head = ["Show", "Guess", "Gaps (bits 1+)", "Wait p50 / p95 (s)", "Total wait per show (s)", "Zero-wait gaps", "Bridge lines", "Guesses right / made",
            "Right on 3 labels", "Safety wasted", "Rating"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def line(name: str, guess: str, s: dict, rating: str) -> str:
        g, w = s["guesses"], s["wait_s"]
        return (f"| {name} | {guess} | {len(s['gaps']) if isinstance(s['gaps'], list) else s['gaps']} | "
                f"{fmt(w['p50'], '{:.1f}')} / {fmt(w['p95'], '{:.1f}')} | {fmt(s.get('total_wait_s', s.get('total_wait_per_show_s')), '{:.1f}')} | {s['zero_waits']} | {s['bridge_lines']} | "
                f"{g['hits']} / {g['n']} | {g['learned_hits']} / {g['learned_n']} | {g['wasted_prefetch']} | {rating} |")

    for side, guess in (("before", "no"), ("after", "yes")):
        for s in res[side]:
            rows.append(line(s["session"], guess, s, "-" if s["rating"] is None else str(s["rating"])))
        rows.append(line(f"all {side}", guess, res[f"{side}_pooled"], ""))
    g, ci = res["after_pooled"]["guesses"], res["after_pooled"]["hit_ci95"]
    tail = ""
    if ci:
        tail = (f"\n\nGuess hit rate {g['hits']}/{g['n']} = {g['hits'] / g['n']:.2f}, Wilson 95% interval {ci[0]:.2f} to {ci[1]:.2f} "
                "(optimistic: guesses within a show are not independent); Exp 31's bar was a lower end of at least 0.5.")
    if res["excluded"]:
        tail += "\n\nNot finished, counted nowhere: " + ", ".join(f"{e['session']} ({e['status']})" for e in res["excluded"])
    return "\n".join(rows) + tail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before", nargs="+", required=True)
    ap.add_argument("--after", nargs="+", required=True)
    ap.add_argument("--dir", default=str(SESSIONS))
    ap.add_argument("--record", action="store_true", help="save as the Exp 31 live evidence (replaces the last)")
    a = ap.parse_args()
    res = record(a.before, a.after, a.dir) if a.record else run(a.before, a.after, a.dir)
    print(table(res))
    print(f"recorded to reports/metrics/{NAME}.json" if a.record else "not recorded (add --record)")


if __name__ == "__main__":
    main()
