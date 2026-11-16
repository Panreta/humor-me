"""Exp 22: the Stage 0 acceptance run (docs/plans/live-loop.md, T11). Scores the shows the user ran.

Usage:
    python -m sim.acceptance <session id> [<session id> ...] [--dir data/sessions] [--rerun]

Reads the logs (and the recordings beside them) and reports, per show, whether it is a clean one, then the plan's
pass checks. Nothing from the logs is stored but numbers: no text, no audio, no paths (`reports/metrics/acceptance.json`).

Checks, from the plan:
  (a) 3 simulated shows in a row, each valid and finished (no abort);
  (b) at least 1 live show, valid and finished;
  (c) label spread: a live show where one key takes more than 60% of the presses is flagged. A judgment call in
      the plan, so it is reported and never blocks;
  (d) real per-bit reward noise against the sim's, and "rerun Exp 18" when it differs by more than half.

What "the sim's 0.15" is: `sim.policies.BIT_REWARD_SD`, also LinUCB's `alpha`. It is a rounded stand-in for the sd of
a bit's reward (mean of 5 levels / 3) that the sim produces from its latent noise (sd 1.0 in `audience.yaml`): measured
at the best mode, 0.128 to 0.140 over the 27 rooms, mean 0.136 (`evaluate_env.env_reward_sd`). It is the spread of
the 5 jokes' mean around a fixed expectation, so the live analogue is the spread of a bit's jokes around that bit's
own mean, divided down to the mean of 5: sqrt(pooled within-bit variance of level/3 x mean(1/n labels)).
Joke-to-joke differences in funniness inside a bit count as noise here, as the sim's per-joke draws do; differences
between bits do not (they are what the director is trying to learn).
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.director import joke_levels  # noqa: E402
from app.log import KEY_LEVELS, SESSIONS, read_session  # noqa: E402
from app.reward import MIN_LABELS  # noqa: E402
from sim import cache  # noqa: E402
from sim.policies import BIT_REWARD_SD  # noqa: E402

SPREAD_LIMIT = 0.60  # plan: one key over 60% of presses
SIM_SD_MEASURED = 0.136  # evaluate_env.env_reward_sd, mean over the 27 rooms (0.128 to 0.140)
RERUN_FACTOR = 0.5  # "differs by more than half"
SIMS_IN_A_ROW = 3
N_JOKES, N_BITS = 40, 8
KEY_OF = {level: key for key, level in KEY_LEVELS.items()}


def label_spread(levels: list[int | None]) -> dict:
    """Count and share per key of each joke's final label (the last press, which is what the reward uses), `none` for
    a window nobody answered. `presses` excludes `none`, and the flag looks at presses only."""
    counts = {key: 0 for key in (*KEY_LEVELS, "none")}
    for level in levels:
        counts["none" if level is None else KEY_OF[level]] += 1
    presses = sum(counts[k] for k in KEY_LEVELS)
    share = {k: counts[k] / len(levels) if levels else 0.0 for k in counts}
    top = max(KEY_LEVELS, key=lambda k: counts[k])
    return {"counts": counts, "share": share, "presses": presses, "top_key": top if presses else None,
            "top_share_of_presses": counts[top] / presses if presses else 0.0,
            "flagged": presses > 0 and counts[top] / presses > SPREAD_LIMIT}


def bit_levels(rows: list[dict]) -> list[list[int | None]]:
    """Per finished bit (one with a `bit_end` row), the final level of each joke."""
    return [joke_levels(rows, r["bit"]) for r in rows if r["event"] == "bit_end"]


def noise(bits: list[list[int | None]]) -> dict:
    """The live analogue of the sim's per-bit reward sd (module docstring), over bits with enough labels to count
    as a reward. `ci95` is a normal approximation on the pooled sd with df = sum(n - 1); it assumes independent,
    roughly normal residuals, which 3-4 level scores on few bits only roughly are, so read it as a guide."""
    var_sum = df = inv_n = 0.0
    n_bits = 0
    for levels in bits:
        given = [lv / 3 for lv in levels if lv is not None]
        if len(given) < MIN_LABELS:
            continue
        mean = sum(given) / len(given)
        var_sum += sum((x - mean) ** 2 for x in given)
        df += len(given) - 1
        inv_n += 1 / len(given)
        n_bits += 1
    if not df:
        return {"n_bits": 0, "df": 0, "joke_sd": None, "bit_sd": None, "scale": None, "ci95": None}
    joke_sd = math.sqrt(var_sum / df)
    scale = math.sqrt(inv_n / n_bits)
    se = joke_sd / math.sqrt(2 * df)
    return {"n_bits": n_bits, "df": int(df), "joke_sd": joke_sd, "bit_sd": joke_sd * scale, "scale": scale,
            "ci95": [max(0.0, joke_sd - 1.96 * se) * scale, (joke_sd + 1.96 * se) * scale]}


def noise_verdict(est: dict) -> dict:
    """Against the plan's 0.15: rerun Exp 18 when the estimate is more than half away from it. `decided` says whether
    the whole interval is on one side of the band's edge, so a rerun call on a few bits is marked as a guess."""
    if est["bit_sd"] is None:
        return {"rerun_exp18": None, "decided": False, "band": None}
    low, high = BIT_REWARD_SD * (1 - RERUN_FACTOR), BIT_REWARD_SD * (1 + RERUN_FACTOR)
    point = not low <= est["bit_sd"] <= high
    lo, hi = est["ci95"]
    inside, outside = low <= lo and hi <= high, hi < low or lo > high
    return {"rerun_exp18": point, "decided": inside or outside, "band": [low, high],
            "relative_to_sim": est["bit_sd"] / BIT_REWARD_SD - 1}


def wav_seconds(path: pathlib.Path) -> float | None:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except (OSError, EOFError, wave.Error):
        return None


def bank_jokes(rows: list[dict]) -> int:
    """Jokes played from the hand-checked bank: the played item's reason says so (backend/app/show.py)."""
    items = {(r["bit"], r["joke"]): r["items"] for r in rows if r["event"] == "candidates"}
    return sum((items[(r["bit"], r["joke"])][r["chosen"]]["reason"] or "").startswith("hand-checked bank")
               for r in rows if r["event"] == "joke_played")  # a passing generated item has reason null


def audio_coverage(rows: list[dict], directory: pathlib.Path) -> dict | None:
    audio = rows[0]["audio"]
    if audio.get("path") is None:
        return None
    end = next((r["at"] for r in rows if r["event"] == "show_end"), rows[-1]["at"])
    span, seconds = end - audio["start_at"], wav_seconds(directory / audio["path"])
    return {"wav_s": seconds, "log_span_s": span, "coverage": None if seconds is None or span <= 0 else seconds / span}


def show_report(session: str, directory: pathlib.Path) -> dict:
    """One show's row. A log that does not read is reported as invalid, with the reason, and nothing else about it."""
    path = directory / f"{session}.jsonl"
    try:
        rows = read_session(path)
        if not rows or rows[0]["event"] != "show_start":
            raise ValueError("log does not begin with show_start")
    except (OSError, ValueError) as e:
        return {"session": session, "valid": False, "error": f"{type(e).__name__}: {e}"}
    start, ends = rows[0], [r for r in rows if r["event"] == "show_end"]
    played = [r for r in rows if r["event"] == "joke_played"]
    levels = [lv for bit in bit_levels(rows) for lv in bit]
    ratings = [r["value"] for r in rows if r["event"] == "rating"]
    return {"session": session, "valid": True, "error": None, "at": start["at"], "room": start["room"],
            "status": ends[-1]["status"] if ends else "none", "generator_id": start["generator_id"],
            "n_jokes": len(played), "n_bits": sum(r["event"] == "bit_end" for r in rows),
            "bank_jokes": bank_jokes(rows), "bank_share": bank_jokes(rows) / len(played) if played else None,
            "n_bridges": sum(r["event"] == "bridge" for r in rows), "labels": label_spread(levels),
            "rating": ratings[-1] if ratings else None, "audio": audio_coverage(rows, directory),
            "noise": noise(bit_levels(rows))}


def clean(show: dict) -> bool:
    return show["valid"] and show["status"] == "finished"


def sims_in_a_row(shows: list[dict]) -> bool:
    """Three simulated shows, consecutive in start order among the sim shows given, all clean."""
    sims = [s for s in sorted((s for s in shows if s["valid"]), key=lambda s: s["at"]) if s["room"] == "sim"]
    return any(all(clean(s) for s in sims[i:i + SIMS_IN_A_ROW]) for i in range(len(sims) - SIMS_IN_A_ROW + 1))


def pooled_noise(shows: list[dict]) -> dict:
    """Over every live show given: the pooled estimate and its verdict."""
    live = [s for s in shows if s["valid"] and s["room"] == "live" and s["status"] == "finished"]
    est = _combine([s["noise"] for s in live])
    return {"n_live_shows": len(live), **est, "sim_sd_plan": BIT_REWARD_SD, "sim_sd_measured": SIM_SD_MEASURED,
            **noise_verdict(est)}


def _combine(parts: list[dict]) -> dict:
    """Pool per-show estimates exactly: joke_sd^2 is a df-weighted mean, and 1/n enters through the bit count."""
    parts = [p for p in parts if p["df"]]
    if not parts:
        return {"n_bits": 0, "df": 0, "joke_sd": None, "bit_sd": None, "scale": None, "ci95": None}
    df = sum(p["df"] for p in parts)
    joke_sd = math.sqrt(sum(p["df"] * p["joke_sd"] ** 2 for p in parts) / df)
    n_bits = sum(p["n_bits"] for p in parts)
    scale = math.sqrt(sum(p["n_bits"] * p["scale"] ** 2 for p in parts) / n_bits)
    se = joke_sd / math.sqrt(2 * df)
    return {"n_bits": n_bits, "df": df, "joke_sd": joke_sd, "bit_sd": joke_sd * scale,
            "ci95": [max(0.0, joke_sd - 1.96 * se) * scale, (joke_sd + 1.96 * se) * scale]}


def run(sessions: list[str], key: str = "", directory: str | pathlib.Path = SESSIONS) -> dict:
    """`key` only distinguishes cache entries (see `digest`); the logs are read from `directory`."""
    directory = pathlib.Path(directory)
    shows = [show_report(s, directory) for s in sessions]
    live = [s for s in shows if s["valid"] and s["room"] == "live"]
    return {"shows": shows,
            "pass_a_sims_in_a_row": sims_in_a_row(shows),
            "pass_b_live_show": any(clean(s) for s in live),
            "flag_c_label_spread": [s["session"] for s in live if s["labels"]["flagged"]],
            "noise": pooled_noise(shows)}


def digest(sessions: list[str], directory: str | pathlib.Path = SESSIONS) -> str:
    """The cache key's stand-in for the log files, so an edited or newly rated log recomputes."""
    h = hashlib.sha256()
    for s in sessions:
        path = pathlib.Path(directory) / f"{s}.jsonl"
        h.update(path.read_bytes() if path.exists() else b"missing")
    return h.hexdigest()[:16]


def cached(sessions: list[str], directory: str | pathlib.Path = SESSIONS, rerun: bool = False) -> dict:
    return cache.run_or_load("acceptance", lambda sessions, key: run(sessions, key, directory), rerun=rerun,
                       sessions=sessions, key=digest(sessions, directory))


def stored() -> dict | None:
    """The last result `cached` saved, without needing the logs (they stay on the user's machine); None before a run."""
    path = cache.METRICS / "acceptance.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def table(res: dict) -> str:
    lines = ["| Show | Room | Valid | Status | Generator | Jokes | Bank | Bridges | W / K / Space / D / none | Rating | Audio |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in res["shows"]:
        if not s["valid"]:
            lines.append(f"| {s['session']} | - | NO: {s['error']} | - | - | - | - | - | - | - | - |")
            continue
        c = s["labels"]["counts"]
        cov = (s["audio"] or {}).get("coverage")
        lines.append(f"| {s['session']} | {s['room']} | yes | {s['status']} | {s['generator_id']} | {s['n_jokes']} | "
                     f"{pct(s['bank_share'])} | {s['n_bridges']} | "
                     f"{c['W']} / {c['K']} / {c['Space']} / {c['D']} / {c['none']} | {s['rating'] or '-'} | "
                     f"{'not recorded' if s['audio'] is None else pct(cov)} |")
    return "\n".join(lines)


def verdict(res: dict) -> str:
    n = res["noise"]
    out = [f"(a) {SIMS_IN_A_ROW} simulated shows in a row, all clean: {'PASS' if res['pass_a_sims_in_a_row'] else 'NOT MET'}",
           f"(b) a live show, valid and finished: {'PASS' if res['pass_b_live_show'] else 'NOT MET'}",
           "(c) label spread, one key over 60% of presses (reported, never blocks): "
           + (", ".join(res["flag_c_label_spread"]) + " FLAGGED" if res["flag_c_label_spread"] else "none flagged")]
    if n.get("bit_sd") is None:
        out.append("(d) per-bit reward noise: no live bit with 3 or more labels yet")
    else:
        lo, hi = n["ci95"]
        out.append(f"(d) per-bit reward noise {n['bit_sd']:.3f} (95% {lo:.3f} to {hi:.3f}) from {n['n_bits']} bits of "
                   f"{n['n_live_shows']} live show(s), against the sim's {n['sim_sd_plan']} (measured {n['sim_sd_measured']}); "
                   f"band {n['band'][0]:.3f} to {n['band'][1]:.3f}: "
                   + ("RERUN Exp 18" if n["rerun_exp18"] else "no rerun needed")
                   + ("" if n["decided"] else " (the interval crosses the band's edge: a guess, not a finding)"))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+", help="session ids: the file names in data/sessions without .jsonl")
    ap.add_argument("--dir", default=str(SESSIONS))
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    res = cached(a.sessions, a.dir, rerun=a.rerun)
    print(table(res), "", verdict(res), sep="\n")


if __name__ == "__main__":
    main()
