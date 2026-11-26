"""Exp 31 live check (D-115): did guessing the next bit's mode shorten the wait before a bit's first joke?

Usage, after the live shows (seconds from the local logs, no model call, never any line of text):
    python -m sim.evaluate_gaps --before <id> ... --after <id> ... [--dir data/sessions] [--record]

`--before` are shows without the guess, `--after` shows with it. Plain, it prints and saves nothing; `--record` writes
`reports/metrics/live_gaps.json` (ids and numbers only).

Per show: for bits 1 and up, the wait from the previous `bit_end` to the first joke's audio start (the same measure as
`sim.stage_timing`; a bit whose wait overlaps a pause is dropped), how many waits were zero (at most ZERO_S), the filler
lines played (all `bridge` rows) and the show's rating. For shows after the change also the guess hit rate, the guesses
made on 3 labels and the safety requests wasted on wrong guesses. A show from before D-113 has no `prep` rows; its
waits come from the bridge and joke timestamps, which is all this reads.

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

NAME = "live_gaps"
ZERO_S = 0.1


def show_report(session: str, rows: list[dict]) -> dict:
    gaps = [b for b in bits_of(rows) if b["bit"] >= 1 and not b["paused"]]
    ends = [r["status"] for r in rows if r["event"] == "show_end"]
    ratings = [r["value"] for r in rows if r["event"] == "rating"]
    return {"session": session, "room": rows[0]["room"], "generator_id": rows[0]["generator_id"],
            "status": ends[-1] if ends else "none", "rating": ratings[-1] if ratings else None,
            "gaps": [round(b["wait_s"], 3) for b in gaps], "wait_s": stats([b["wait_s"] for b in gaps]),
            "zero_waits": sum(b["wait_s"] <= ZERO_S for b in gaps),
            "filler_lines": sum(r["event"] == "bridge" for r in rows), "guesses": guess_stats(rows)}


def pool(shows: list[dict]) -> dict:
    waits = [w for s in shows for w in s["gaps"]]
    keys = ("n", "hits", "adopted", "learned_n", "learned_hits", "wasted_prefetch")
    return {"shows": len(shows), "gaps": len(waits), "wait_s": stats(waits),
            "zero_waits": sum(s["zero_waits"] for s in shows), "filler_lines": sum(s["filler_lines"] for s in shows),
            "guesses": {k: sum(s["guesses"][k] for s in shows) for k in keys}}


def run(before: list[str], after: list[str], directory: str | pathlib.Path = SESSIONS, key: str = "") -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    shows, bad = load(pathlib.Path(directory), before + after)
    by = {sid: show_report(sid, rows) for sid, rows in shows}
    b = [by[s] for s in before if s in by]
    a = [by[s] for s in after if s in by]
    return {"before": b, "after": a, "before_pooled": pool(b), "after_pooled": pool(a), "unreadable_logs": bad,
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
    head = ["Show", "Guess", "Gaps (bits 1+)", "Wait p50 / p95 (s)", "Zero-wait gaps", "Filler lines", "Guesses right / made",
            "Right on 3 labels", "Safety wasted", "Rating"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def line(name: str, guess: str, s: dict, rating: str) -> str:
        g, w = s["guesses"], s["wait_s"]
        return (f"| {name} | {guess} | {len(s['gaps']) if isinstance(s['gaps'], list) else s['gaps']} | "
                f"{fmt(w['p50'], '{:.1f}')} / {fmt(w['p95'], '{:.1f}')} | {s['zero_waits']} | {s['filler_lines']} | "
                f"{g['hits']} / {g['n']} | {g['learned_hits']} / {g['learned_n']} | {g['wasted_prefetch']} | {rating} |")

    for side, guess in (("before", "no"), ("after", "yes")):
        for s in res[side]:
            rows.append(line(s["session"], guess, s, "-" if s["rating"] is None else str(s["rating"])))
        rows.append(line(f"all {side}", guess, res[f"{side}_pooled"], ""))
    return "\n".join(rows)


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
