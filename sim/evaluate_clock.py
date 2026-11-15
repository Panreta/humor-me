"""Exp 21: do logged times line up with the recorded audio?

Usage (needs a real mic; the user runs it, see docs/design.md "Audio and consent"):
    python -m sim.evaluate_clock play --n 90 --interval 10 --out clicks.json    # beside a 15-minute live show
    python -m sim.evaluate_clock check data/sessions/<id>.jsonl clicks.json

`play` beeps through the speakers and writes the wall-clock time before each beep; the show's recording hears them.
`check` finds each beep in the WAV, converts it to wall clock with the log's `show_start.audio.start_at`, and reports
the offset from the time `play` logged. Bars, fixed in docs/plans/live-loop.md (T10): p95 offset <= 100 ms and no
drift. "No drift" was not quantified there; the bar used here is a fitted drift over the show of at most 50 ms
(judgment call). `play` stamps the time before the call returns control to the sound card, so its own latency is in
the offset; run it once on its own with a loopback or a mic next to the speakers and read the constant part as that.
Importable: run(), detect_clicks(), play_clicks().
"""
import argparse
import json
import pathlib
import sys
import time
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.log import read_session  # noqa: E402

BAR_P95_S, BAR_DRIFT_S = 0.1, 0.05
MATCH_S = 1.0  # a detection further than this from a logged click is not that click


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32768.0, w.getframerate()


def detect_clicks(samples: np.ndarray, rate: int, min_gap_s: float = 1.0) -> list[float]:
    """Seconds into the audio of each click's first sample over a threshold set well above the room's noise."""
    level = np.abs(samples)
    threshold = max(8 * float(np.median(level)), 0.05)
    hits = np.flatnonzero(level > threshold)
    times: list[float] = []
    for i in hits:
        t = i / rate
        if not times or t - times[-1] >= min_gap_s:
            times.append(t)
    return times


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


def run(log_path: str, wav_path: str, clicks_path: str) -> dict:
    audio = read_session(pathlib.Path(log_path))[0]["audio"]
    if audio.get("path") is None:
        raise ValueError("this show was not recorded")
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
            "pass_p95": p95 <= BAR_P95_S, "pass_drift": abs(slope * span) <= BAR_DRIFT_S}


def play_clicks(out: str, n: int, interval_s: float) -> None:
    import winsound  # Windows only; the operator machine is Windows

    times = []
    for _ in range(n):
        times.append(time.time())
        winsound.Beep(2000, 40)
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
    log = pathlib.Path(a.log)
    wav = a.wav or log.parent / read_session(log)[0]["audio"]["path"]
    print(json.dumps(run(str(log), str(wav), a.clicks), indent=2))


if __name__ == "__main__":
    main()
