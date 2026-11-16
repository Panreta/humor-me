"""Exp 21: do logged times line up with the recorded audio?

Usage (needs a real mic; the user runs it, see docs/design.md "Audio and consent"):
    # 1. a SIMULATED room, GENERATOR=bank, consent box ticked, speakers audible to the mic,
    #    Windows microphone "Audio enhancements" OFF (the driver's echo cancellation removes the laptop's own output)
    # 2. beside a 15-minute show, on the same machine:
    python -m sim.evaluate_clock play --n 90 --interval 10 --out clicks.json
    # 3. afterwards:
    python -m sim.evaluate_clock check data/sessions/<id>.jsonl clicks.json

`play` plays a 2 kHz, 100 ms tone through the speakers and saves the wall-clock time before each one; the show's
recording hears them. `check` finds each tone in the WAV with a matched filter for that tone (phase-independent, so
speech and laughter do not win), converts it to wall clock with the log's `show_start.audio.start_at`, and compares it
with the time `play` logged. Beeps whose search window is not inside the recording are counted as outside, not missed.

Bar, re-tuned by the user on 2026-11-15 after the first real run (decisions.md): drift over the show at most 50 ms;
p95 of |offset - median offset| (the spread around a constant) at most 100 ms; at least 90% of the in-recording beeps
matched. The constant offset is reported, not barred: it includes PlaySound's output latency, which this setup cannot
separate from the recorder's start time. The old bar (p95 |offset| at most 100 ms) is still computed, as `pass_old_p95`.
Results are cached in reports/metrics/audio_clock.json (numbers only), keyed by session id and a digest of the clicks
file. Importable: run(), cached(), table(), plot(), detect_clicks(), play_clicks().
"""
import argparse
import hashlib
import io
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

BAR_SPREAD_S, BAR_DRIFT_S, BAR_MATCHED = 0.1, 0.05, 0.9
OLD_BAR_P95_S = 0.1  # the first bar, on the absolute offset; kept as a comparison
MATCH_S = 0.5  # a detection further than this from a logged beep is not that beep (the constant offset was ~0.25 s)
NOISE_FACTOR = 8  # a beep is a peak at least this many times the median of the matched-filter output
BEEP_HZ, BEEP_MS = 2000, 100  # 100 ms: a 40 ms tone was hard to hear


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "expected mono 16-bit"
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2") / 32768.0, w.getframerate()


def detections(samples: np.ndarray, rate: int, min_gap_s: float = 0.2) -> tuple[list[float], list[float], float]:
    """(onset seconds, peak sizes, median of the filter output) for every tone found. Matched filter for the beep: the
    magnitude of the 2 kHz component summed over exactly one beep's length, so the output peaks where a beep starts
    whatever its phase. Speech and laughter spread over many frequencies and add little to one narrow component; the
    threshold is relative to the recording's own noise, because a laptop mic records the beeps at ~0.003."""
    length = max(1, round(rate * BEEP_MS / 1000))
    if samples.size <= length:
        return [], [], 0.0
    z = samples * np.exp(-2j * np.pi * BEEP_HZ * np.arange(samples.size) / rate)
    total = np.concatenate(([0], np.cumsum(z)))
    env = np.abs(total[length:] - total[:-length]) / length  # env[i]: a beep starting at sample i
    noise = float(np.median(env))
    hits = np.flatnonzero(env > NOISE_FACTOR * noise)
    if hits.size == 0:
        return [], [], noise
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > min_gap_s * rate) + 1)
    peaks = [g[np.argmax(env[g])] for g in groups]
    return [float(i / rate) for i in peaks], [float(env[i]) for i in peaks], noise


def detect_clicks(samples: np.ndarray, rate: int, min_gap_s: float = 0.2) -> list[float]:
    return detections(samples, rate, min_gap_s)[0]


def offsets(logged: list[float], detected_wall: list[float], strength: list[float] | None = None):
    """(logged times that matched, their offsets detected - logged, number that matched nothing). Of the detections
    within MATCH_S of a logged time, the strongest wins, so speech near a beep does not take its place."""
    det = np.asarray(detected_wall)
    power = np.asarray(strength if strength is not None else np.ones(len(detected_wall)))
    kept, diffs = [], []
    for t in logged:
        near = np.flatnonzero(np.abs(det - t) <= MATCH_S) if det.size else []
        if len(near):
            kept.append(t)
            diffs.append(float(det[near[np.argmax(power[near])]] - t))
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


def run(log_path: str, clicks_path: str, wav_path: str | None = None) -> dict:
    wav_path = wav_for(log_path, wav_path)
    audio = read_session(pathlib.Path(log_path))[0]["audio"]
    samples, rate = read_wav(wav_path)
    start, end = audio["start_at"], audio["start_at"] + samples.size / rate
    every = json.loads(pathlib.Path(clicks_path).read_text())
    logged = [t for t in every if start <= t and t + MATCH_S <= end]  # a beep whose search window left the file is no miss
    times, power, noise = detections(samples, rate)
    kept, diffs, missed = offsets(logged, [start + t for t in times], power)
    if len(diffs) < 3:
        raise ValueError(f"only {len(diffs)} beeps matched; check the mic level, Windows microphone enhancements and the clicks file")
    span = kept[-1] - kept[0]
    slope = float(np.polyfit(np.asarray(kept) - kept[0], diffs, 1)[0])
    median = float(np.median(diffs))
    spread = np.abs(np.asarray(diffs) - median)
    abs_p95 = float(np.percentile(np.abs(diffs), 95))
    wall = np.asarray(times) + start
    peaks = [power[int(np.argmin(np.abs(wall - (t + d))))] / noise for t, d in zip(kept, diffs)]
    p95_spread = float(np.percentile(spread, 95))
    return {"n_logged": len(logged), "n_outside": len(every) - len(logged), "n_matched": len(diffs), "n_missed": missed,
            "span_s": span, "median_offset_s": median, "p95_abs_offset_s": abs_p95, "p95_spread_s": p95_spread,
            "max_spread_s": float(spread.max()), "drift_slope_s_per_s": slope, "drift_over_show_s": slope * span,
            "peak_over_noise_median": float(np.median(peaks)), "peak_over_noise_min": float(min(peaks)),
            "pass_spread": p95_spread <= BAR_SPREAD_S, "pass_drift": abs(slope * span) <= BAR_DRIFT_S,
            "pass_matched": len(diffs) >= BAR_MATCHED * len(logged), "pass_old_p95": abs_p95 <= OLD_BAR_P95_S,
            "t_s": [t - kept[0] for t in kept], "offset_ms": [d * 1000 for d in diffs]}


def cached(session: str, key: str, log_path: str, clicks_path: str, wav_path: str | None = None,
           rerun: bool = False) -> dict:
    """run() through run_or_load. The cache is keyed by the session id and `key` (`digest(...)`) only, so the committed
    JSON holds numbers, never paths to the recording; the files are read only when there is no cached result."""
    return run_or_load("audio_clock", lambda session, key: run(log_path, clicks_path, wav_path), rerun=rerun,
                       session=session, key=key)


def table(res: dict) -> str:
    ok = lambda b: "yes" if b else "NO"  # noqa: E731
    return "\n".join([
        "| Beeps matched (in recording) | Outside the recording | Median offset | p95 spread around it | Drift over the show | Spread <= 100 ms | Drift <= 50 ms | >= 90% matched |",
        "|---|---|---|---|---|---|---|---|",
        f"| {res['n_matched']}/{res['n_logged']} | {res['n_outside']} | {res['median_offset_s'] * 1000:+.0f} ms | "
        f"{res['p95_spread_s'] * 1000:.0f} ms | {res['drift_over_show_s'] * 1000:+.0f} ms | "
        f"{ok(res['pass_spread'])} | {ok(res['pass_drift'])} | {ok(res['pass_matched'])} |"])


def plot(res: dict, ax) -> None:
    """Offset of each beep against its time in the show, with the median and the +-100 ms band the bar allows."""
    median = res["median_offset_s"] * 1000
    ax.axhspan(median - BAR_SPREAD_S * 1000, median + BAR_SPREAD_S * 1000, color="0.9", zorder=0)
    ax.axhline(median, color="0.5", lw=1)
    ax.scatter(res["t_s"], res["offset_ms"], s=28, zorder=3)
    ax.set_xlabel("seconds into the show")
    ax.set_ylabel("heard time minus logged time (ms)")


def beep_wav(rate: int = 48_000, volume: float = 0.5) -> bytes:
    """A BEEP_HZ tone as an in-memory WAV, with 2 ms ramps so it starts without a click at other frequencies."""
    t = np.arange(int(rate * BEEP_MS / 1000)) / rate
    ramp = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.002)
    pcm = (volume * ramp * np.sin(2 * np.pi * BEEP_HZ * t) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def play_clicks(out: str, n: int, interval_s: float) -> None:
    # winsound.Beep goes to a legacy beep path that many laptop drivers ignore (silent on the operator's laptop);
    # PlaySound uses the normal output device, the same one the robot's voice plays through.
    import winsound  # Windows only; the operator machine is Windows

    tone, times = beep_wav(), []
    for i in range(n):
        times.append(time.time())
        winsound.PlaySound(tone, winsound.SND_MEMORY)
        pathlib.Path(out).write_text(json.dumps(times))  # each beep saved at once, so Ctrl+C keeps what played
        print(f"beep {i + 1}/{n}", flush=True)
        time.sleep(max(0.0, interval_s - BEEP_MS / 1000))


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
        session = read_session(pathlib.Path(a.log))[0]["session"]
        res = cached(session, digest(a.log, a.clicks), a.log, a.clicks, wav)
    except (ValueError, OSError) as e:
        sys.exit(f"error: {e}")
    print(table(res))
    print(json.dumps({k: v for k, v in res.items() if k not in ("t_s", "offset_ms")}, indent=2))


if __name__ == "__main__":
    main()
