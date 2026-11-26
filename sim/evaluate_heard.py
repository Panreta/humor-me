"""Exp 28 (D-112): how many of a show's bridge lines had the listener heard before?

Usage:
    python -m sim.evaluate_heard [--dir data/sessions] [--record]

Reads every session log in the folder (local files only, no model call). Plain, it prints and saves nothing;
`--record` writes `reports/metrics/heard_bridges.json`: counts and session ids only, never a line of text.

A line is heard when its normalised text (lowercase, spaces collapsed) was a `joke_played` or `bridge` row of a show
that started earlier, in any room (`app.cache.heard_texts`). Per show it counts the bridge lines by kind (cached from an
earlier live show, filler from `generate.BRIDGES`) and how many of each were heard before, plus the cached lines the
rule would have kept in the pool (`popular_bits` over the earlier shows, with and without the rule) and the filler
repeated inside the show. Across shows it counts how many of the 10 filler lines the listener has heard.

Bar, fixed before the run on these logs (the problem was first seen in show e204ec80e62e, so the run confirms it and
cannot discover it): the rule is needed if, in some show, at least half of the cached bridge lines had been heard before.
The rule itself is checked by its guarantee, not by a number: over every recorded show, the pool the rule builds from
the earlier shows holds 0 heard lines.
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import cache as live_cache  # noqa: E402
from app.generate import BRIDGES  # noqa: E402
from app.log import SESSIONS, read_session  # noqa: E402
from sim import cache  # noqa: E402

NAME = "heard_bridges"
NEEDED_SHARE = 0.5
FILLER = {live_cache.normalise(t) for t in BRIDGES}


def load(directory: pathlib.Path) -> tuple[list[tuple[str, list[dict]]], list[str]]:
    """(valid shows as (session id, rows), oldest first; the ids of logs that did not read)."""
    shows, bad = [], []
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        try:
            rows = read_session(path)
            if not rows or rows[0]["event"] != "show_start":
                raise ValueError("no show_start")
            shows.append((path.stem, rows))
        except (OSError, ValueError):
            bad.append(path.stem)
    shows.sort(key=lambda s: s[1][0]["at"])
    return shows, bad


def pool_lines(earlier: list[list[dict]], skip_heard: bool) -> int:
    heard = live_cache.heard_texts(earlier) if skip_heard else frozenset()
    pool = live_cache.popular_bits(earlier, heard=heard)
    return sum(len(b.lines) for bits in pool.values() for b in bits)


def show_report(session: str, rows: list[dict], earlier: list[list[dict]]) -> dict:
    """Counts for one show against the shows before it. No text leaves this function."""
    heard = live_cache.heard_texts(earlier)
    norm = live_cache.normalise
    bridges = [r for r in rows if r["event"] == "bridge"]
    cached = [r for r in bridges if r.get("kind") == "cached"]
    filler = [r for r in bridges if r.get("kind", "filler") == "filler"]  # a bridge row before D-91 has no kind: all were filler
    jokes = [r for r in rows if r["event"] == "joke_played"]
    said = [norm(r["text"]) for r in filler]
    end = next((r["status"] for r in rows if r["event"] == "show_end"), "none")
    row = next((r for r in rows if r["event"] == "cache"), None)
    return {"session": session, "at": rows[0]["at"], "room": rows[0]["room"], "status": end,
            "jokes": len(jokes), "jokes_heard_before": sum(norm(r["text"]) in heard for r in jokes),
            "bridges": len(bridges), "cached": len(cached), "cached_heard_before": sum(norm(r["text"]) in heard for r in cached),
            "filler": len(filler), "filler_heard_before": sum(t in heard for t in said),
            "filler_repeated_in_show": len(said) - len(set(said)),
            "pool_lines_without_rule": pool_lines(earlier, False), "pool_lines_with_rule": pool_lines(earlier, True),
            "heard_dropped_logged": None if row is None else row.get("heard_dropped")}


def earlier_of(shows: list[tuple[str, list[dict]]], i: int) -> list[list[dict]]:
    return [r for _, r in shows[:i]]


def heard_in_pool(shows: list[tuple[str, list[dict]]], i: int) -> int:
    """Heard lines in the pool the rule builds for show i: 0 by construction, counted rather than assumed."""
    earlier = earlier_of(shows, i)
    heard = live_cache.heard_texts(earlier)
    pool = live_cache.popular_bits(earlier, heard=heard)
    return sum(live_cache.normalise(ln.text) in heard for bits in pool.values() for b in bits for ln in b.lines)


def run(directory: str | pathlib.Path = SESSIONS, key: str = "") -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    shows, bad = load(pathlib.Path(directory))
    reports = [show_report(sid, rows, [r for _, r in shows[:i]]) for i, (sid, rows) in enumerate(shows)]
    played = {live_cache.normalise(r["text"]) for _, rows in shows for r in rows
              if r["event"] == "bridge" and r.get("kind", "filler") == "filler"} & FILLER
    with_cached = [s for s in reports if s["cached"]]
    worst = max((s["cached_heard_before"] / s["cached"] for s in with_cached), default=None)
    return {"n_shows": len(reports), "unreadable_logs": bad, "shows": reports, "n_filler_lines": len(FILLER),
            "filler_lines_heard": len(played), "needed_share": NEEDED_SHARE, "worst_cached_heard_share": worst,
            "rule_needed": worst is not None and worst >= NEEDED_SHARE,
            "rule_pool_heard_lines": sum(heard_in_pool(shows, i) for i in range(len(shows)))}


def digest(directory: str | pathlib.Path = SESSIONS) -> str:
    """Stand-in for the files read, so a new or edited log recomputes."""
    h = hashlib.sha256()
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()[:16]


def record(directory: str | pathlib.Path = SESSIONS) -> dict:
    return cache.run_or_load(NAME, lambda key: run(directory, key), rerun=True, key=digest(directory))


def stored() -> dict | None:
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def table(res: dict) -> str:
    lines = ["| Show | Room | Jokes heard before | Cached bridges heard before | Filler heard before | Filler repeated in show | Pool lines (no rule / rule) |",
             "|---|---|---|---|---|---|---|"]
    for s in res["shows"]:
        lines.append(f"| {s['session']} | {s['room']} | {s['jokes_heard_before']} of {s['jokes']} | "
                     f"{s['cached_heard_before']} of {s['cached']} | {s['filler_heard_before']} of {s['filler']} | "
                     f"{s['filler_repeated_in_show']} | {s['pool_lines_without_rule']} / {s['pool_lines_with_rule']} |")
    return "\n".join(lines)


def verdict(res: dict) -> str:
    worst = res["worst_cached_heard_share"]
    if worst is None:
        return "Rule needed: NOT DECIDED, no recorded show played a cached bridge line"
    return (f"Rule needed (at least {res['needed_share']:.0%} of one show's cached bridges heard before): "
            f"{'YES' if res['rule_needed'] else 'NO'} (worst show {worst:.0%}); "
            f"{res['filler_lines_heard']} of {res['n_filler_lines']} filler lines heard so far")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(SESSIONS))
    ap.add_argument("--record", action="store_true", help="save as the Exp 28 evidence (replaces the last)")
    a = ap.parse_args()
    res = record(a.dir) if a.record else run(a.dir)
    print(table(res), "", verdict(res), sep="\n")
    print(f"recorded to reports/metrics/{NAME}.json" if a.record else "not recorded (add --record)")


if __name__ == "__main__":
    main()
