"""Exp 32 (D-116): the show ledger, one row per logged show, so the report can say how joke quality moved between setups.

Usage, after each show (seconds and counts from the local logs, no model call; the record never holds a line of text):
    python -m sim.show_ledger [--dir data/sessions] [--record]     # print the ledger; --record saves it
    python -m sim.show_ledger export [--dir data/sessions]         # write the played bits to reports/show_bits (committed)
then `python scripts/run_notebook.py notebooks/decision-plot.ipynb` for the chart.

Plain, it prints and saves nothing; `--record` writes `reports/metrics/show_ledger.json`. The cache key is the session
files' names, sizes and mtimes plus the head of `main`, so a new show (or a merge that names an old commit's PR)
re-records.

The reported outcome is the user's rating (invariant 5). It is the user's own for every show, sim room or live. What the
audience did is different: in a `sim` room the reactions come from the made-up audience in `sim/audience.yaml`, so
reaction levels and bit rewards count as quality evidence for LIVE shows only. They are kept per show and per setup in
two separate blocks (`live`, `simulated audience`) and never pooled. Delivery signals (bank share, lost threads, waits,
bridge lines, guess hit rate) come from the writer and the voice, which are real in both rooms; the waits depend on the
audience only through when reactions arrive.

A setup is the writer (generator id: prompt and model) x next-bit guess on/off (`prediction` rows present) x the bridge
rule (from the `cache` row: none, repeats allowed, or skips lines already heard, D-112). Setups are ordered by the first
show that used them. Means and ratings count finished shows only; an aborted show is a row with its bits played and
counts nowhere else. A log that does not read is listed under `unreadable` with the error class.

Rating noise: ratings are few, so a setup's mean comes with its range, and adjacent setups (in start order, rated ones
only) get an exact two-sided permutation test on the difference of means. `within_noise` is true when p is at least 0.05
or either side has fewer than 2 ratings. These are different shows, not paired seeds, so the test says only whether the
ratings could be a reshuffle, not that the setup caused the difference.

Judgment calls, not measured: the setup axes, the 0.05 line, the label for each setup change (`CHANGES`), and mapping a
show's `git_sha` to the pull request that first merged it into `main`.
"""
import argparse
import datetime
import hashlib
import itertools
import math
import json
import pathlib
import re
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.log import SESSIONS, read_session  # noqa: E402
from sim import cache  # noqa: E402
from sim.acceptance import bank_jokes  # noqa: E402
from sim.compare_generators import fmt  # noqa: E402
from sim.evaluate_gaps import ZERO_S, show_report as gap_report  # noqa: E402
from sim.stage_timing import bits_of  # noqa: E402

NAME = "show_ledger"
VERSION = 3  # bump when the record changes shape, so an unchanged set of logs still re-records
BITS_DIR = ROOT / "reports" / "show_bits"
ALPHA = 0.05
LIVE, SIM = "live", "simulated audience"
BANK_PREFIX = "bank"

# What a first show of a setup changed, for the chart's markers. Judgment calls; each names its decision.
CHANGES = {
    "writer": {"bank-v1": "hand-written bank only", "gen-v3": "T9 LLM writer",
               "gen-v4": "D-110 one call per thread", "gen-v5": "D-111 addendum: connected-bit prompt live"},
    "model": {"bonsai-2:4b": "D-109 shows write with bonsai"},
    "reply_fixes": "D-111 reply-format fixes",
    "prediction": "D-115 next bit prepared early",
    "bridge": {"repeat-ok": "D-91 cached bridge lines", "skip-heard": "D-112 skips heard lines"},
}
# Plain words first, the code name in brackets after them, wherever a person reads a setup.
PROMPT_WORDS = {"gen-v3": "3 threads in one reply", "gen-v4": "one call per thread, 3 in parallel",
                "gen-v5": "connected-bit prompt, each line builds on the last"}
MODEL_WORDS = {"gemini-flash-lite-latest": "Gemini", "bonsai-2:4b": "local (bonsai)"}
BRIDGE_WORDS = {"no-cache": "no cached lines", "repeat-ok": "may replay heard lines", "skip-heard": "skips heard lines"}
# The first merged pull request whose code has the change; a show on older code lacks it (D-111's reply fixes, D-115's guess).
REPLY_FIXES_PR = 79
GUESS_PR = 83


def writer_parts(generator_id: str) -> tuple[str, str]:
    prompt, _, model = generator_id.partition(":")
    return prompt, model


def bridge_rule(rows: list[dict]) -> str:
    """From the `cache` row: absent (before D-91), repeats allowed (no flag, or False), or skips heard lines (D-112)."""
    c = next((r for r in rows if r["event"] == "cache"), None)
    if c is None:
        return "no-cache"
    return "skip-heard" if c.get("skip_heard") else "repeat-ok"


def code_id(rec: dict) -> str:
    """The prompt's code name; gen-v4 shows on code before D-111's reply-format fixes are a different setup."""
    return rec["prompt"] + (" before D-111 fixes" if rec.get("reply_fixes") is False else "")


def setup_of(rec: dict) -> str:
    return f"{rec['generator_id']}{' before D-111 fixes' if rec['reply_fixes'] is False else ''} / guess {'on' if rec['prediction'] else 'off'} / bridges {rec['bridge']}"


def lost_class(reason: str) -> str:
    """The error class of a lost thread's reason (reasons hold the class and counts, never joke text)."""
    if "ReadTimeout" in reason or "Timeout" in reason:
        return "timeout"
    if reason.startswith("re-ask failed"):
        return "re-ask failed"
    if "expected a premise" in reason or "malformed reply" in reason:
        return "bad reply shape"
    return "other"


def threads_lost(rows: list[dict]) -> dict:
    """Lost threads per bit are logged on each of its jokes' rows; the bit's own list is the longest of them."""
    per_bit: dict[int, list[str]] = {}
    for r in rows:
        if r["event"] == "candidates":
            lost = r.get("threads_lost", [])
            if len(lost) > len(per_bit.get(r["bit"], [])):
                per_bit[r["bit"]] = lost
    by_class: dict[str, int] = {}
    for lost in per_bit.values():
        for reason in lost:
            by_class[lost_class(reason)] = by_class.get(lost_class(reason), 0) + 1
    return {"threads": sum(len(v) for v in per_bit.values()), "bits": len(per_bit), "by_class": by_class}


def bit_lines(rows: list[dict]) -> dict[int, list[dict]]:
    """The lines actually played, per bit in order: text (held in memory only), from the bank or not, and the reaction."""
    cands = {(r["bit"], r["joke"]): r for r in rows if r["event"] == "candidates"}
    reacted = {(r["bit"], r["joke"]): r for r in rows if r["event"] == "reaction"}
    out: dict[int, list[dict]] = {}
    for r in rows:
        if r["event"] != "joke_played":
            continue
        c = cands[(r["bit"], r["joke"])]
        reason = c["items"][r["chosen"]]["reason"] or ""
        banked = c["generator_id"].startswith(BANK_PREFIX) or reason.startswith("hand-checked bank")
        out.setdefault(r["bit"], []).append({"joke": r["joke"], "text": r["text"], "bank": banked,
                                             "level": (reacted.get((r["bit"], r["joke"])) or {}).get("level")})
    return out


def bit_hash(lines: list[str]) -> str:
    """Names a bit in the record without holding it: the first 16 hex of the sha256 of its played lines."""
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pr_of(sha: str, ref: str = "main", cwd: pathlib.Path = ROOT) -> dict:
    """The pull request a show's code belongs to ("#80", branch), else the short sha. A sha that is itself a merge on
    `ref`'s first-parent line is that merge; any other commit gets the first merge that brought it in. `git` failing is
    not an error."""
    short = sha[:7]
    try:
        def git(*a):
            return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        full = git("rev-parse", "--verify", f"{sha}^{{commit}}")
        if full in git("rev-list", "--first-parent", ref).splitlines():
            merge = full
        else:
            first = git("rev-list", "--first-parent", "--ancestry-path", "--reverse", f"{full}..{ref}").splitlines()
            merge = first[0] if first else full
        subject = git("log", "-1", "--format=%s", merge)
    except (OSError, subprocess.SubprocessError):
        return {"sha": short, "pr": None, "branch": None}
    m = re.match(r"Merge pull request #(\d+) from \S+?/(\S+)", subject)
    return {"sha": short, "pr": int(m.group(1)) if m else None, "branch": m.group(2) if m else None}


def main_head(ref: str = "main", cwd: pathlib.Path = ROOT) -> str:
    try:
        return subprocess.run(["git", "rev-parse", ref], cwd=cwd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def local_time(at: float) -> datetime.datetime:
    """The show's start on the machine's clock (so an evening show lands on its own day); UTC when the OS cannot say."""
    try:
        return datetime.datetime.fromtimestamp(at, datetime.timezone.utc).astimezone()
    except (OSError, OverflowError, ValueError):
        return datetime.datetime.fromtimestamp(at, datetime.timezone.utc)


def show_row(session: str, rows: list[dict], pr: dict | None = None) -> dict:
    start = rows[0]
    pr = pr or {"sha": start["git_sha"][:7], "pr": None, "branch": None}
    ends = [r["status"] for r in rows if r["event"] == "show_end"]
    ratings = [r["value"] for r in rows if r["event"] == "rating"]
    gaps = gap_report(session, rows)
    lines = bit_lines(rows)
    modes = {r["bit"]: r["action"] for r in rows if r["event"] == "bit_decision"}
    live = start["room"] == "live"
    levels = [r["level"] for r in rows if r["event"] == "reaction" and r["level"] is not None]
    rewards = [r["reward"] for r in rows if r["event"] == "bit_end" and r["reward"] is not None]
    played = sum(len(v) for v in lines.values())
    lost = threads_lost(rows)
    waits = gaps["gaps"]
    guesses = gaps["guesses"]
    prompt, model = writer_parts(start["generator_id"])
    n_bits = sum(r["event"] == "bit_end" for r in rows)
    guessed = any(r["event"] == "prediction" for r in rows)
    # A show cut short before its first guess has no prediction rows; its code decides (PREDICT_NEXT_BIT defaults on).
    inferred = not guessed and n_bits < 2 and pr["pr"] is not None and pr["pr"] >= GUESS_PR
    when = local_time(start["at"])
    rec = {
        "session": session, "at": start["at"],
        "started": when.strftime("%Y-%m-%d %H:%M"), "utc_offset": when.strftime("%z")[:3] + ":" + when.strftime("%z")[3:],
        "room": start["room"], "status": ends[-1] if ends else "none",
        "bits_played": n_bits, "jokes_played": played,
        "git_sha": start["git_sha"][:7], "pr": pr,
        "reply_fixes": (pr["pr"] is None or pr["pr"] >= REPLY_FIXES_PR) if prompt == "gen-v4" else None, "generator_id": start["generator_id"], "prompt": prompt, "model": model,
        "prediction": guessed or inferred, "prediction_inferred": inferred, "bridge": bridge_rule(rows),
        "rating": ratings[-1] if ratings else None,
        "audience": {"kind": LIVE if live else SIM, "reaction_n": len(levels), "reaction_mean": mean(levels),
                     "bit_rewards_n": len(rewards), "reward_mean": mean(rewards)},
        "delivery": {"bank_share": bank_jokes(rows) / played if played else None, "threads_lost": lost["threads"],
                     "bits_with_lost_threads": lost["bits"], "lost_by_class": lost["by_class"],
                     "wait_median_s": statistics.median(waits) if waits else None, "wait_worst_s": max(waits) if waits else None,
                     "total_wait_s": gaps["total_wait_s"], "gaps": len(waits), "bridge_lines": gaps["bridge_lines"],
                     "guesses": guesses["n"], "guess_hits": guesses["hits"],
                     "pauses": sum(r["event"] == "pause" for r in rows)},
        "bits": [{"bit": b, "mode": modes.get(b), "lines": len(v), "hash": bit_hash([x["text"] for x in v]),
                  "bank_lines": sum(x["bank"] for x in v)} for b, v in sorted(lines.items())],
    }
    rec["setup"] = setup_of(rec)
    rec["setup_label"] = setup_words(rec)
    rec["code_label"] = code_label(rec)
    return rec


def permutation_p(a: list[int], b: list[int]) -> float:
    """Exact two-sided p of the difference of means: every way to split the pooled ratings into groups of the same sizes."""
    pooled, n = a + b, len(a)
    observed = abs(statistics.mean(a) - statistics.mean(b))
    total = hits = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        chosen = set(idx)
        x = [pooled[i] for i in idx]
        y = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        total += 1
        hits += abs(statistics.mean(x) - statistics.mean(y)) >= observed - 1e-12
    return hits / total


def smallest_p(n: int, k: int) -> float:
    """The smallest p a test of groups of n and k can give (two extreme splits): what these sizes can ever show."""
    return 2 / math.comb(n + k, k)


def change_label(prev: dict | None, cur: dict) -> str:
    if prev is None:
        return "first show"
    parts = []
    if cur["prompt"] != prev["prompt"]:
        parts.append(CHANGES["writer"].get(cur["prompt"], cur["prompt"]))
    elif cur["model"] != prev["model"]:
        parts.append(CHANGES["model"].get(cur["model"], cur["model"]))
    elif prev["reply_fixes"] is False and cur["reply_fixes"] is not False:
        parts.append(CHANGES["reply_fixes"])
    if cur["prediction"] and not prev["prediction"]:
        parts.append(CHANGES["prediction"])
    if cur["bridge"] != prev["bridge"]:
        parts.append(CHANGES["bridge"].get(cur["bridge"], cur["bridge"]))
    return " + ".join(p for p in parts if p) or "changed"


def audience_block(shows: list[dict]) -> dict:
    """Reaction and reward numbers of ONE audience kind; the caller passes live shows or sim shows, never both."""
    return {"shows": len(shows), "reaction_n": sum(s["audience"]["reaction_n"] for s in shows),
            "reaction_mean": mean([s["audience"]["reaction_mean"] for s in shows]),
            "reward_mean": mean([s["audience"]["reward_mean"] for s in shows])}


def setup_row(key: str, shows: list[dict], first: dict, prev: dict | None) -> dict:
    done = [s for s in shows if s["status"] == "finished"]
    ratings = sorted(s["rating"] for s in done if s["rating"] is not None)
    live_ratings = [s["rating"] for s in done if s["rating"] is not None and s["room"] == "live"]
    d = [s["delivery"] for s in done]
    hits, made = sum(x["guess_hits"] for x in d), sum(x["guesses"] for x in d)
    lost_by_class: dict[str, int] = {}
    for x in d:
        for k, v in x["lost_by_class"].items():
            lost_by_class[k] = lost_by_class.get(k, 0) + v
    return {
        "setup": key, "label": setup_words(first), "code_label": code_label(first), "generator_id": first["generator_id"], "prediction": first["prediction"], "bridge": first["bridge"],
        "first_session": first["session"], "first_started": first["started"], "first_at": first["at"], "change": change_label(prev, first),
        "pr": pr_label(first), "reply_fixes": first["reply_fixes"], "shows": len(shows), "finished": len(done), "aborted": len(shows) - len(done),
        "rating": {"n": len(ratings), "live_n": len(live_ratings), "values": ratings, "mean": mean(ratings),
                   "min": ratings[0] if ratings else None, "max": ratings[-1] if ratings else None},
        "delivery": {"bank_share": mean([x["bank_share"] for x in d]),
                     "threads_lost_per_show": mean([x["threads_lost"] for x in d]), "lost_by_class": lost_by_class,
                     "wait_median_s": mean([x["wait_median_s"] for x in d]), "wait_worst_s": max([x["wait_worst_s"] for x in d if x["wait_worst_s"] is not None], default=None),
                     "total_wait_s": mean([x["total_wait_s"] for x in d]), "bridge_lines": mean([x["bridge_lines"] for x in d]),
                     "guess_hits": hits, "guesses": made, "guess_hit_rate": hits / made if made else None},
        "audience": {LIVE: audience_block([s for s in done if s["room"] == "live"]),
                     SIM: audience_block([s for s in done if s["room"] == "sim"])},
    }


def pr_label(show: dict) -> str:
    return f"#{show['pr']['pr']} {show['pr']['branch']}" if show["pr"]["pr"] else show["git_sha"]


def adjacent_tests(setups: list[dict]) -> list[dict]:
    """`within_noise` is null when the test cannot say: too few ratings, or the sizes cannot reach p below ALPHA at all
    (`p_min`). Only then is true ("the gap could be a reshuffle") or false ("unlikely to be a reshuffle") evidence."""
    rated = [s for s in setups if s["rating"]["n"]]
    out = []
    for a, b in zip(rated, rated[1:]):
        x, y = a["rating"]["values"], b["rating"]["values"]
        ok = len(x) >= 2 and len(y) >= 2
        p = permutation_p(x, y) if ok else None
        p_min = smallest_p(len(x), len(y)) if ok else None
        can_reject = ok and p_min < ALPHA
        out.append({"from": a["setup"], "to": b["setup"], "n": [len(x), len(y)], "diff_of_means": b["rating"]["mean"] - a["rating"]["mean"],
                    "p": p, "p_min": p_min, "within_noise": (p >= ALPHA) if can_reject else None})
    return out


def run(directory: str | pathlib.Path = SESSIONS, key: str = "", ref: str = "main") -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    shows, unreadable, prs = [], [], {}
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        try:
            rows = read_session(path)
            if not rows or rows[0]["event"] != "show_start":
                raise ValueError("no show_start")
            sha = rows[0]["git_sha"]
            shows.append(show_row(path.stem, rows, prs.setdefault(sha, pr_of(sha, ref))))
        except (OSError, ValueError, KeyError) as e:
            unreadable.append({"session": path.stem, "reason": type(e).__name__})
    shows.sort(key=lambda s: s["at"])
    groups: dict[str, list[dict]] = {}
    for s in shows:
        groups.setdefault(s["setup"], []).append(s)
    setups, prev = [], None
    for k, g in groups.items():
        setups.append(setup_row(k, g, g[0], prev))
        prev = g[0]
    return {"shows": shows, "setups": setups, "adjacent": adjacent_tests(setups), "unreadable": unreadable,
            "n_files": len(list(pathlib.Path(directory).glob("*.jsonl"))), "alpha": ALPHA}


def digest(directory: str | pathlib.Path = SESSIONS, ref: str = "main") -> str:
    h = hashlib.sha256((main_head(ref) + json.dumps([VERSION, CHANGES, CODE_NAMES], sort_keys=True)).encode())  # a label change re-records too
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        st = path.stat()
        h.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def record(directory: str | pathlib.Path = SESSIONS, ref: str = "main", rerun: bool = False) -> dict:
    """`rerun=False` loads the stored record when no log changed; a new or edited log changes the key and re-records."""
    return cache.run_or_load(NAME, lambda key: run(directory, key, ref), rerun=rerun, key=digest(directory, ref))


def stored() -> dict | None:
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def table(res: dict) -> str:
    head = ["Setup (in start order)", "What changed (and the PR its first show ran on)", "Shows", "Ratings (n)", "Mean", "Range", "Bank share", "Threads lost per show",
            "Typical wait (average of per-show medians, s)", "Total wait per show (s)", "Bridge lines", "Guess right / made"]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in res["setups"]:
        r, d = s["rating"], s["delivery"]
        rng = "-" if not r["n"] else (f"{r['min']}" if r["min"] == r["max"] else f"{r['min']} to {r['max']}")
        rows.append(f"| {s['label']} | {s['change']} ({s['pr']}) | {s['finished']} done, {s['aborted']} aborted | {r['n']} | {fmt(r['mean'], '{:.2f}')} | {rng} | "
                    f"{fmt(d['bank_share'], '{:.0%}')} | {fmt(d['threads_lost_per_show'], '{:.1f}')} | {fmt(d['wait_median_s'], '{:.1f}')} | "
                    f"{fmt(d['total_wait_s'], '{:.0f}')} | {fmt(d['bridge_lines'], '{:.1f}')} | {d['guess_hits']} / {d['guesses']} |")
    out = "\n".join(rows)
    notes, labels = [], {s["setup"]: s["label"] for s in res["setups"]}
    for t in res["adjacent"]:
        if t["p"] is None:
            verdict = "too few ratings to tell (a setup has fewer than 2)"
        else:
            verdict = (f"a random reshuffle of the same ratings gives a gap this big {t['p']:.0%} of the time; "
                       f"at these sizes the best possible is {t['p_min']:.0%}, so ")
            verdict += ("too few ratings to tell" if t["within_noise"] is None else
                        "the gap could be a reshuffle" if t["within_noise"] else "the gap is unlikely to be a reshuffle")
        notes.append(f"- {labels[t['from']]} to {labels[t['to']]} (next rated setup), n {t['n'][0]} and {t['n'][1]}: mean {t['diff_of_means']:+.2f}, {verdict}")
    live = [s for s in res["shows"] if s["room"] == "live"]
    out += "\n\nAdjacent rated setups, exact permutation test:\n" + "\n".join(notes)
    out += (f"\n\nLive shows: {', '.join(s['session'] + ' (' + s['status'] + ')' for s in live) or 'none'}. Reaction levels and bit rewards count for "
            "live shows only; in a sim room they come from the made-up audience and are kept apart (simulated audience).")
    if res["unreadable"]:
        out += "\n\nUnreadable logs, listed and not counted: " + ", ".join(f"{u['session']} ({u['reason']})" for u in res["unreadable"])
    return out


def slug(setup: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", setup.lower()).strip("-")


REACTION = {3: "big laugh", 2: "laugh", 1: "small laugh", 0: "no laugh", None: "no reaction recorded"}
BRIDGE_NOTE = {"cached": "replayed line from show {s}", "filler": "filler while the next bit is written"}


# The reference table for the code names (decisions from docs/decisions.md, pull requests from `git log --merges`).
CODE_NAMES = [
    ("bank-v1", "hand-written bank only", "D-66", "#40"),
    ("gen-v3 Gemini", "Gemini writer, 3 threads in one reply", "D-82", "#50"),
    ("gen-v3 bonsai", "local (bonsai) writer, 3 threads in one reply", "D-107, D-109", "#77"),
    ("gen-v4 before D-111 fixes", "one call per thread, 3 in parallel, before the reply-format fixes", "D-110", "#78"),
    ("gen-v4", "one call per thread, 3 in parallel, with the reply-format fixes", "D-110, D-111", "#78, #79"),
    ("gen-v5", "connected-bit prompt, each line builds on the last", "D-111 addendum", "#84"),
    ("+ guess", "next bit prepared early", "D-115", "#83"),
    ("+ skip-heard", "skips heard lines", "D-112", "#80"),
]


def code_table() -> str:
    rows = ["| Code name | What it is | Decision | PR |", "|---|---|---|---|"]
    return "\n".join(rows + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in CODE_NAMES])


def code_label(rec: dict) -> str:
    """The short chart label: the code name only (plots use this; text uses `setup_words`)."""
    base = rec["prompt"] + (" " + {"gemini-flash-lite-latest": "Gemini", "bonsai-2:4b": "bonsai"}.get(rec["model"], rec["model"]) if rec["prompt"] == "gen-v3" else "")
    base = base + (" before D-111 fixes" if rec.get("reply_fixes") is False else "")
    return base + (" + skip-heard" if rec["bridge"] == "skip-heard" else "") + (" + guess" if rec["prediction"] else "")


def setup_words(rec: dict) -> str:
    """The setup in plain words (what was different), then its code name in brackets."""
    if rec["prompt"].startswith(BANK_PREFIX):
        writer = f"hand-written bank only ({rec['prompt']})"
    else:
        early = ", before the reply-format fixes" if rec.get("reply_fixes") is False else ""
        writer = f"{MODEL_WORDS.get(rec['model'], rec['model'])} writer, {PROMPT_WORDS.get(rec['prompt'], rec['prompt'])}{early} ({code_id(rec)})"
    guess = "early" if rec["prediction"] else "after"
    return f"{writer}; next bit prepared {guess}; {BRIDGE_WORDS[rec['bridge']]}"


def bridge_note(row: dict) -> str:
    """A bridge row's label. A row with no `kind` is an older filler line; a cached one names the show it was replayed from."""
    if row.get("kind") == "cached":
        return BRIDGE_NOTE["cached"].format(s=(row.get("source") or {}).get("session", "an earlier show"))
    return BRIDGE_NOTE["filler"]


def gap_note(wait: float | None, kinds: list[str]) -> str:
    """The wait and the bridge lines played in it, by kind: "filler" while the next bit was written, "cached" replayed."""
    filler, replayed = sum(k != "cached" for k in kinds), sum(k == "cached" for k in kinds)
    n = len(kinds)
    if not n:
        lines = "no bridge lines"
    elif not replayed:
        lines = f"{n} filler line{'s' * (n != 1)}"
    elif not filler:
        lines = f"{n} bridge line{'s' * (n != 1)}, {'all ' if n > 1 else ''}replayed"
    else:
        lines = f"{n} bridge lines ({filler} filler, {replayed} replayed)"
    if wait is None:
        return "wait not measured (a pause or a failed first line)" + (f", {lines}" if n else "")
    if wait <= ZERO_S and not n:
        return "no wait"
    return f"waited {wait:.0f} s, {lines}"


def ended_words(rec: dict) -> str:
    n = f"{rec['bits_played']} bit{'s' * (rec['bits_played'] != 1)}"
    return {"finished": f"finished, {n}", "aborted": f"aborted after {n}"}.get(rec["status"], f"no end row, {n}")


def show_markdown(rec: dict, rows: list[dict]) -> str:
    """One show top to bottom as it was heard: every played line and bridge line, in audio order."""
    lines, modes = bit_lines(rows), {r["bit"]: r["action"] for r in rows if r["event"] == "bit_decision"}
    played = {(r["bit"], r["joke"]): r for r in rows if r["event"] == "joke_played"}
    waits = {b["bit"]: (None if b["paused"] or b["failed"] else b["wait_s"]) for b in bits_of(rows)}
    bridges = sorted((r for r in rows if r["event"] == "bridge"), key=lambda r: r["audio_start_at"])
    first = {b: min(played[(b, ln["joke"])]["audio_start_at"] for ln in v) for b, v in lines.items()}
    last = {b: max(played[(b, ln["joke"])]["audio_start_at"] for ln in v) for b, v in lines.items()}
    audience = rec["audience"]["kind"]
    out = [f"# Show {rec['session']}", "", f"- Date (local, UTC{rec['utc_offset']}): {rec['started']}", f"- Setup: {rec['setup_label']}",
           f"- Room: {'live' if rec['room'] == 'live' else 'simulated'}",
           f"- Your rating: {rec['rating'] if rec['rating'] is not None else 'none'}",
           f"- How it ended: {ended_words(rec)}", ""]

    def bridge_lines(group: list[dict]) -> list[str]:
        return [f"*({bridge_note(r)})* {r['text']}" for r in group] + [""] * bool(group)

    order = sorted(lines)
    placed: set[int] = set()
    for i, bit in enumerate(order):
        lo = last[order[i - 1]] if i else float("-inf")
        gap = [r for r in bridges if lo < r["audio_start_at"] < first[bit]]
        placed.update(id(r) for r in gap)
        if i:
            out += [f"*Gap before this bit: {gap_note(waits.get(bit), [r.get("kind", "filler") for r in gap])}.*", ""]
        elif gap:
            out += ["*Before the first bit:*", ""]
        out += bridge_lines(gap)
        out += [f"## Bit {bit + 1} — {modes.get(bit, 'unknown').capitalize()}", ""]
        for n, ln in enumerate(lines[bit], 1):
            tags = ["from the bank"] * ln["bank"]
            reaction = REACTION[ln["level"]]
            out.append(f"{n}. {ln['text']}" + (f" *({tags[0]})*" if tags else "")
                       + f" — {reaction if audience == LIVE else f'(simulated audience: {reaction})'}")
        out.append("")
    tail = [r for r in bridges if id(r) not in placed]
    if tail:
        out += ["*After the last bit:*", ""] + bridge_lines(tail)
    return "\n".join(out)


def clear_exported(out: pathlib.Path) -> None:
    """Remove the files an earlier export wrote (and folders left empty), so a renamed or dropped show leaves nothing stale."""
    if not out.exists():
        return
    for f in out.rglob("*.md"):
        f.unlink()
    for d in sorted((d for d in out.rglob("*") if d.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()


def export(directory: str | pathlib.Path = SESSIONS, out: pathlib.Path = BITS_DIR, ref: str = "main") -> list[pathlib.Path]:
    """Write each show as a plain file that reads top to bottom like the show, filler included (D-116: committed on purpose).
    Only text the writer or the bank generated and the show played is read: `joke_played.text` and `bridge.text`, with
    the reaction level of each. Nothing said by people in the room is ever read; the log schema has no such field (it
    holds keys and levels, the room's venue/energy/age tags and a rating; speech lives in the separate, uncommitted WAV).
    Returns the files written (the index last)."""
    clear_exported(out)
    entries, written, prs = [], [], {}
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        try:
            rows = read_session(path)
            if not rows or rows[0]["event"] != "show_start":
                continue
            sha = rows[0]["git_sha"]
            rec = show_row(path.stem, rows, prs.setdefault(sha, pr_of(sha, ref)))
        except (OSError, ValueError, KeyError):
            continue
        folder = out / slug(rec["code_label"])
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{rec['session']}.md"
        target.write_text(show_markdown(rec, rows), encoding="utf-8")
        written.append(target)
        entries.append(rec)
    entries.sort(key=lambda r: r["at"], reverse=True)
    index = ["# Shows by setup", "", "## Code names", "", code_table(), "", "Each show reads top to bottom as it was heard, filler lines included. The lines were generated by the writer "
             "or the bank; nothing said by people in the room. Newest first.", ""]
    for setup in dict.fromkeys(r["setup"] for r in entries):
        group = [r for r in entries if r["setup"] == setup]
        folder = slug(group[0]["code_label"])
        index += [f"## {group[0]['setup_label']}", "", f"| Date (local, UTC{group[0]['utc_offset']}) | Room | How it ended | Your rating | Show |", "|---|---|---|---|---|"]
        index += [f"| {r['started']} | {'live' if r['room'] == 'live' else 'simulated'} | {ended_words(r)} | "
                  f"{r['rating'] if r['rating'] is not None else 'none'} | [{r['session']}]({folder}/{r['session']}.md) |" for r in group]
        index.append("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.md").write_text("\n".join(index), encoding="utf-8")
    written.append(out / "index.md")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", choices=["export"], help="export: write the played bits to reports/show_bits")
    ap.add_argument("--dir", default=str(SESSIONS))
    ap.add_argument("--record", action="store_true", help="save as the Exp 32 record (re-records when any log changed)")
    a = ap.parse_args()
    if a.command == "export":
        files = export(a.dir)
        print(f"wrote {len(files)} files under {BITS_DIR} (committed: generated lines only)")
        return
    res = record(a.dir, rerun=False) if a.record else run(a.dir)
    print(table(res))
    print(f"recorded to reports/metrics/{NAME}.json" if a.record else "not recorded (add --record)")


if __name__ == "__main__":
    main()
