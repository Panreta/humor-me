"""Exp 29 (D-113): which stage of a bit's wait is the biggest, so the right one gets fixed first?

Usage, after a show (counts and seconds only, no model call, never any line of text):
    python -m sim.stage_timing [session ids ...] [--dir data/sessions] [--record]

With no ids it reads every log in the folder. Plain, it prints and saves nothing; `--record` writes
`reports/metrics/stage_timing.json` (session ids and seconds). A log from before D-113 has no `prep` rows: it is listed
with its bit waits, with the stage columns empty, and is left out of the decision.

A bit's wait runs from the previous `bit_end` (`show_start` for bit 0) to its first `joke_played`'s audio start. The
`prep` row splits the part before the bit was ready into generate_s (the writer), safety_s (the filter), bank_s (the
hand-checked bank), render_queue_s (waiting for the serial voice behind a bridge or cached render) and render_s (the
synthesis); `bridge_overhang_s` on the first joke is what the room waited after the bit was ready because a bridge line
was still playing. Median and p95 are nearest-rank.

Decision rule, fixed before any data (the question was raised by show 6c09c721b0bb, whose waits were 9.8 to 17.2 s with
the writer at about 3.5 s, Exp 26; that show only motivates the question and is not the bar). Over the bits of the
recorded shows that have a `prep` row, the share of a stage is its median seconds over the median wait. The stage with
the largest share is fixed first, and only when it is at least MIN_SHARE (25%) of the median wait and at least
TIE_RATIO (1.1x) the runner-up; below that the answer is "no one stage", and nothing is built on a guess. At least
MIN_BITS (16, two shows) bits are needed, else "not enough data". The fixes it chooses between:
  bridge_overhang_s -> (a) end the bridge early or start the joke after the current line instead of letting the bridge
                       loop decide
  render_queue_s    -> (b) render the first joke ahead of bridge and cached renders
  safety_s          -> (c) safety-check each thread as it arrives, overlapping the filter with generation
  generate_s, render_s, bank_s -> none of (a) to (c): the writer (Exp 26) or the voice speed is the limit, reported only.
"""
import argparse
import hashlib
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.log import SESSIONS, read_session  # noqa: E402
from sim import cache  # noqa: E402
from sim.compare_generators import fmt, percentile  # noqa: E402

NAME = "stage_timing"
MIN_SHARE = 0.25
TIE_RATIO = 1.1
MIN_BITS = 16
STAGES = ("generate_s", "safety_s", "bank_s", "render_queue_s", "render_s", "bridge_overhang_s")
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
    """One dict per bit that played a line: its wait, bridge count and, when logged, the prep stages and overhang."""
    start = rows[0]["at"]
    prep = {r["bit"]: r for r in rows if r["event"] == "prep"}
    first: dict[int, dict] = {}
    for r in rows:
        if r["event"] == "joke_played":
            first.setdefault(r["bit"], r)
    out = []
    ends = {r["bit"]: r["at"] for r in rows if r["event"] == "bit_end"}
    for bit in sorted(first):
        began = start if bit == 0 else ends.get(bit - 1)
        if began is None:
            continue
        p, played = prep.get(bit), first[bit]
        out.append({"bit": bit, "wait_s": max(0.0, played["audio_start_at"] - began),
                    "bridges": sum(r["event"] == "bridge" and r["bit"] == bit for r in rows),
                    "prepped": p is not None,
                    **{k: (p.get(k) if p else None) for k in STAGES if k != "bridge_overhang_s"},
                    "bridge_overhang_s": played.get("bridge_overhang_s")})
    return out


def stats(values: list[float]) -> dict:
    xs = [v for v in values if v is not None]
    return {"n": len(xs), "p50": statistics.median(xs) if xs else None, "p95": percentile(xs, 95)}


def show_report(session: str, rows: list[dict]) -> dict:
    bits = bits_of(rows)
    return {"session": session, "room": rows[0]["room"], "bits": len(bits), "bits_with_prep": sum(b["prepped"] for b in bits),
            "wait_s": stats([b["wait_s"] for b in bits]), "bridges": stats([b["bridges"] for b in bits]),
            **{k: stats([b[k] for b in bits]) for k in STAGES}}


def decide(pooled: dict) -> dict:
    """The fixed rule on the pooled stats: {"outcome", "stage", "fix", "share"}."""
    n, wait = pooled["bits_with_prep"], pooled["wait_s"]["p50"]
    if n < MIN_BITS or not wait:
        return {"outcome": "not_enough_data", "stage": None, "fix": None, "share": None}
    shares = sorted(((pooled[k]["p50"] / wait, k) for k in STAGES if pooled[k]["p50"] is not None), reverse=True)
    if not shares:
        return {"outcome": "not_enough_data", "stage": None, "fix": None, "share": None}
    (top, stage), second = shares[0], shares[1][0] if len(shares) > 1 else 0.0
    if top < MIN_SHARE or top < TIE_RATIO * second:
        return {"outcome": "no_one_stage", "stage": stage, "fix": None, "share": top}
    return {"outcome": "fix" if stage in FIX else "no_candidate_fix", "stage": stage, "fix": FIX.get(stage), "share": top}


def run(sessions: list[str] | None = None, directory: str | pathlib.Path = SESSIONS, key: str = "") -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    shows, bad = load(pathlib.Path(directory), sessions)
    reports = [show_report(sid, rows) for sid, rows in shows]
    counted = [(sid, rows) for sid, rows in shows if any(r["event"] == "prep" for r in rows)]
    pooled_bits = [b for _, rows in counted for b in bits_of(rows) if b["prepped"]]
    pooled = {"bits_with_prep": len(pooled_bits), "shows_with_prep": len(counted),
              "wait_s": stats([b["wait_s"] for b in pooled_bits]), **{k: stats([b[k] for b in pooled_bits]) for k in STAGES}}
    return {"shows": reports, "unreadable_logs": bad, "pooled": pooled, "min_share": MIN_SHARE, "tie_ratio": TIE_RATIO,
            "min_bits": MIN_BITS, "decision": decide(pooled)}


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
    head = ["Show", "Bits (with prep)", "Bridges per bit p50", "Wait", "Generate", "Safety", "Bank", "Render queue", "Render", "Bridge overhang"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in [*res["shows"], {"session": "all with prep", "room": "", "bits": res["pooled"]["bits_with_prep"],
                              "bits_with_prep": res["pooled"]["bits_with_prep"], "bridges": {"p50": None},
                              "wait_s": res["pooled"]["wait_s"], **{k: res["pooled"][k] for k in STAGES}}]:
        rows.append(f"| {s['session']} {s['room']} | {s['bits']} ({s['bits_with_prep']}) | {fmt(s['bridges']['p50'], '{:.0f}')} | "
                    f"{cell(s['wait_s'])} | " + " | ".join(cell(s[k]) for k in STAGES) + " |")
    return "Seconds, p50 / p95.\n\n" + "\n".join(rows)


def verdict(res: dict) -> str:
    d = res["decision"]
    if d["outcome"] == "not_enough_data":
        return f"Biggest stage: NOT DECIDED, fewer than {res['min_bits']} bits with a prep row"
    share = f"{d['share']:.0%} of the median wait"
    if d["outcome"] == "no_one_stage":
        return f"Biggest stage: no one stage ({d['stage']} leads at {share}; the rule needs {res['min_share']:.0%} and {res['tie_ratio']}x the runner-up)"
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
