"""Exp 21: do logged times line up with the recorded audio?

Usage (needs a real mic; the user runs it, see docs/design.md "Audio and consent"):
    # 1. a SIMULATED room, GENERATOR=bank, consent box ticked, speakers audible to the mic
    # 2. beside a 15-minute show, on the same machine:
    python -m sim.evaluate_clock play --n 90 --interval 10 --out clicks.json
    # 3. afterwards:
    python -m sim.evaluate_clock check data/sessions/<id>.jsonl clicks.json

`play` beeps (2 kHz, 40 ms) through the speakers and writes the wall-clock time before each beep; the show's recording
hears them. `check` finds each beep in the WAV with a narrow band around 2 kHz (so the voice and laughter do not win),
converts it to wall clock with the log's `show_start.audio.start_at`, and reports the offset from the time `play`
logged. Bars, fixed in docs/plans/live-loop.md (T10): p95 offset <= 100 ms and no drift. "No drift" was not quantified
there; the user set it at a fitted drift over the show of at most 50 ms, half the p95 bar so drift cannot use up the whole budget. Clicks
that match nothing are counted, and fewer than 90% matched also fails, so a missed beep cannot hide a bad offset.
`play` stamps the time before the beep starts, so the sound card's latency is in the offset; read a constant part as that.
Results are cached in reports/metrics/audio_clock.json. Importable: run(), table(), detect_clicks(), play_clicks().
"""
import argparse
import hashlib
import json
import pathlib
import sys
import time
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.log import read_session  # noqa: E402
from sim.cache import run_or_load  # noqa: E402

BAR_P95_S, BAR_DRIFT_S, BAR_MATCHED = 0.1, 0.05, 0.9
MATCH_S = 0.25  # a detection further than this from a logged click is not that click
BEEP_HZ, BEEP_MS = 2000, 40


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32768.0, w.getframerate()


def detect_clicks(samples: np.ndarray, rate: int, min_gap_s: float = 0.2) -> list[float]:
    """Seconds into the audio of each beep's onset. Demodulate at the beep frequency and average over 5 ms: speech and
    laughter spread over many frequencies, so little of them survives in that narrow band; the beep is all in it."""
    n = np.arange(samples.size)
    z = samples * np.exp(-2j * np.pi * BEEP_HZ * n / rate)
    window = max(1, int(rate * 0.005))
    env = np.abs(np.convolve(z, np.ones(window) / window, mode="same"))
    threshold = max(8 * float(np.median(env)), 0.6 * float(np.percentile(env, 99.9)), 0.01)
    hits = np.flatnonzero(env > threshold)
    if hits.size == 0:
        return []
    starts = hits[np.r_[True, np.diff(hits) > min_gap_s * rate]]
    return (starts / rate).tolist()


def offsets(logged: list[float], detected_wall: list[float]) -> tuple[list[float], list[float], int]:
    """(logged times that matched, their offsets detected - logged, number that matched nothing)."""
    det = np.asarray(detected_wall)
    kept, diffs = [], []
    for t in logged:
        if det.size:
            nearest = det[np.argmin(np.abs(det - t))]
            if abs(nearest - t) <= MATCH_S:
                kept.append(t)
                diffs.append(float(nearest - t))
    return kept, diffs, len(logged) - len(kept)


def digest(log_path: str, clicks_path: str) -> str:
    """The cache key's stand-in for the two files, so a changed clicks file or another session reruns."""
    h = hashlib.sha256(pathlib.Path(clicks_path).read_bytes())
    h.update(read_session(pathlib.Path(log_path))[0]["session"].encode())
    return h.hexdigest()[:16]


def wav_for(log_path: str, wav_path: str | None = None) -> str:
    audio = read_session(pathlib.Path(log_path))[0]["audio"]
    if audio.get("path") is None:
        raise ValueError("this show was not recorded: tick the consent box and allow the microphone")
    return wav_path or str(pathlib.Path(log_path).parent / audio["path"])


def run(log_path: str, clicks_path: str, wav_path: str | None = None, key: str | None = None) -> dict:
    """`key` is `digest(...)`: it exists so run_or_load's cache sees the files' content, not only their paths."""
    wav_path = wav_for(log_path, wav_path)
    audio = read_session(pathlib.Path(log_path))[0]["audio"]
    logged = json.loads(pathlib.Path(clicks_path).read_text())
    samples, rate = read_wav(wav_path)
    detected = [audio["start_at"] + t for t in detect_clicks(samples, rate)]
    kept, diffs, missed = offsets(logged, detected)
    if len(diffs) < 3:
        raise ValueError(f"only {len(diffs)} clicks matched; check the mic level and the clicks file")
    span = kept[-1] - kept[0]
    slope = float(np.polyfit(np.asarray(kept) - kept[0], diffs, 1)[0])
    p95 = float(np.percentile(np.abs(diffs), 95))
    return {"n_logged": len(logged), "n_matched": len(diffs), "n_missed": missed, "span_s": span,
            "median_offset_s": float(np.median(diffs)), "p95_abs_offset_s": p95,
            "drift_slope_s_per_s": slope, "drift_over_show_s": slope * span,
            "pass_p95": p95 <= BAR_P95_S, "pass_drift": abs(slope * span) <= BAR_DRIFT_S,
            "pass_matched": len(diffs) >= BAR_MATCHED * len(logged)}


def table(res: dict) -> str:
    ok = lambda b: "yes" if b else "NO"  # noqa: E731
    return "\n".join([
        "| Clicks matched | p95 offset | Median offset | Drift over the show | p95 <= 100 ms | Drift <= 50 ms | >= 90% matched |",
        "|---|---|---|---|---|---|---|",
        f"| {res['n_matched']}/{res['n_logged']} | {res['p95_abs_offset_s'] * 1000:.0f} ms | "
        f"{res['median_offset_s'] * 1000:+.0f} ms | {res['drift_over_show_s'] * 1000:+.0f} ms | "
        f"{ok(res['pass_p95'])} | {ok(res['pass_drift'])} | {ok(res['pass_matched'])} |"])


def play_clicks(out: str, n: int, interval_s: float) -> None:
    import winsound  # Windows only; the operator machine is Windows

    times = []
    for _ in range(n):
        times.append(time.time())
        winsound.Beep(BEEP_HZ, BEEP_MS)
        time.sleep(interval_s)
    pathlib.Path(out).write_text(json.dumps(times))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    play = sub.add_parser("play")
    play.add_argument("--n", type=int, default=90)
    play.add_argument("--interval", type=float, default=10.0)
    play.add_argument("--out", default="clicks.json")
    check = sub.add_parser("check")
    check.add_argument("log")
    check.add_argument("clicks")
    check.add_argument("--wav", help="default: the file named in the log, beside it")
    a = parser.parse_args()
    if a.cmd == "play":
        play_clicks(a.out, a.n, a.interval)
        return
    try:
        wav = wav_for(a.log, a.wav)
        res = run_or_load("audio_clock", run, log_path=a.log, clicks_path=a.clicks, wav_path=wav,
                          key=digest(a.log, a.clicks))
    except (ValueError, OSError) as e:
        sys.exit(f"error: {e}")
    print(table(res))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
