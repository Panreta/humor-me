"""Exp 26 (D-110): does one bit still fit its 12 s when it is written as three parallel calls, one thread each?

Usage (Ollama up, `OLLAMA_NUM_PARALLEL` at least 3, `OLLAMA_CONTEXT_LENGTH=8192`, nothing else loaded):
    python -m sim.thread_timing [--rerun]        # -> reports/metrics/thread_timing.json

20 bits on the show's own writer client (`generate.writer_client`, bonsai-2:4b on Ollama), after `generate.warm_up`,
on 20 seeded prompts that are not Exp 24's (ids `t00` to `t19`, another seed), with `premises_already_told` empty. No
safety filter and no voice: this times the writer only. Numbers only, no joke text; a premise is text the model wrote,
so only whether two threads of a bit share one is kept.

Bar, fixed before any run: at most 1 of 20 bits loses a thread (a call that timed out or failed), no bit has zero
threads, and the writer is still 100% on the GPU at context 8192 after the run (parallel calls take more KV cache).
Reported beside it, with no bar: bit wall time p50/p95, per-call p50/p95 next to Exp 24's single-call p50, how many bits queued their calls
(slowest call at least twice the fastest: the server ran them one after another), and how many bits have two threads with the same premise (calls do not see each other, so it can happen).
"""
import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import generate, llm  # noqa: E402
from sim import cache  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim.env import AGES, ENERGIES, VENUES  # noqa: E402

NAME = "thread_timing"
N_BITS = 20
SEED = "exp26"  # Exp 24's prompts use the integer seed 24 and ids p00 to p64
MAX_LOST_BITS = 1
# Client-side call times include Ollama's queue wait. With one slot the three calls last about 1x, 2x and 3x one
# call, so the slowest is about 3 times the fastest; run in parallel they last about the same. Two is the midpoint.
QUEUED_SPREAD = 2.0


def prompts(n: int = N_BITS, seed: str = SEED) -> list[dict]:
    """Seeded like Exp 24's: modes round-robin, each context drawn from its own stream, one per bit. Ids `t00`..."""
    modes = tuple(generate.MODES)
    return [{"id": f"t{i:02d}", "mode": modes[i % len(modes)],
             "context": (lambda r: {"venue": r.choice(VENUES), "energy": r.choice(ENERGIES), "age": r.choice(AGES)})(
                 random.Random(f"{seed}:{i}"))} for i in range(n)]


def writer_model() -> str:
    return llm.setting("JOKE_WRITER_MODEL") or llm.OLLAMA_MODEL


def digest() -> str:
    """What the timed bit is made of: a changed prompt, model, request flag, timing or retry constant starts a new
    record. Code changes elsewhere (the generator's logic, the client) are NOT covered: rerun with `--rerun` after one."""
    parts = [generate.PROMPT_ID, generate.SYSTEM, generate.TEENS_RULE, json.dumps(generate.MODES, sort_keys=True),
             json.dumps(generate.REPLY_SCHEMA, sort_keys=True), json.dumps(generate.COUNTED_REPLY_SCHEMA, sort_keys=True),
             "reask_bad_shape_once",  # D-111: the call re-asks a bad shape inside its deadline
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.RETRIES,
                  generate.WRITER_CONTEXT, generate.HANG_GRACE_S, llm.MIN_ATTEMPT_S)),
             writer_model(), json.dumps(llm.OLLAMA_NO_THINK, sort_keys=True), "json_schema=True", SEED, str(N_BITS)]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8]


class Recorder:
    """Wraps `client.chat` to note each call's start (seconds into the bit), duration and outcome. The calls of a bit
    run on several threads, so every write holds a lock. The reply is parsed here the way the generator parses it, so a
    bad shape is a failure reason too."""

    def __init__(self, client):
        self.calls, self.lock, self.t0 = [], threading.Lock(), 0.0
        inner = client.chat

        def chat(*args, **kwargs):
            start = time.perf_counter()
            reason, premise = None, None
            try:
                reply = inner(*args, **kwargs)
                try:
                    premise = generate.parse(reply)[0]
                except generate.GenerationError as e:
                    reason = f"GenerationError: {e}"[:80]
                return reply
            except llm.LLMError as e:
                reason = f"LLMError: {e}"[:80]
                raise
            finally:
                end = time.perf_counter()
                with self.lock:
                    self.calls.append({"start_s": start - self.t0, "seconds": end - start, "ok": reason is None,
                                       "reason": reason, "premise": premise})
        client.chat = chat

    def begin(self) -> None:
        with self.lock:
            self.calls, self.t0 = [], time.perf_counter()


def time_bit(gen, rec: Recorder, prompt: dict) -> dict:
    rec.begin()
    start = time.perf_counter()
    failure = None
    try:
        threads = len(gen.generate(prompt["mode"], 0, 0, prompt["context"]).texts)
    except Exception as e:  # noqa: BLE001 - a dead provider or a bad reply is the result, not a crash
        threads, failure = 0, f"{type(e).__name__}"
    wall = time.perf_counter() - start
    calls = sorted(rec.calls, key=lambda c: c["start_s"])
    premises = [c["premise"].strip().lower() for c in calls if c["premise"] is not None]
    return {"prompt": prompt["id"], "wall_s": wall, "threads": threads, "failure": failure,
            "calls": [{k: c[k] for k in ("start_s", "seconds", "ok", "reason")} for c in calls],
            "same_premise": len(set(premises)) < len(premises)}


def placement_reading(client, http) -> dict:
    share, context, others = generate.placement(client, http) if http else generate.placement(client)
    return {"gpu_share": share, "context_length": context, "others_loaded": others}


def measure(client, bits: list[dict], http=None) -> dict:
    """Warm the writer, read where Ollama put it, time every bit, read it again."""
    gen = generate.ThreadGenerator(client)
    warm_s = generate.warm_up(gen, http) if http else generate.warm_up(gen)
    before = placement_reading(client, http)
    rec = Recorder(client)  # after warm-up, which runs on a copy and is not a counted call
    timed = [time_bit(generate.ThreadGenerator(client), rec, p) for p in bits]
    after = placement_reading(client, http)
    return {"model": client.model, "warmup_s": warm_s, "placement_after_warmup": before, "placement_after_run": after,
            "num_parallel_in_shell": os.environ.get("OLLAMA_NUM_PARALLEL"),
            "num_parallel_note": "read from this shell only: the server's own value is not exposed by Ollama's API",
            "bits": timed}


def spread(bit: dict) -> float | None:
    """Slowest over fastest delivered call of a bit; None with fewer than two calls that answered."""
    secs = [c["seconds"] for c in bit["calls"] if c["ok"]]
    return max(secs) / min(secs) if len(secs) >= 2 and min(secs) > 0 else None


def exp24_call_p50() -> float | None:
    """Exp 24's single-call p50 for bonsai (one call, 3 threads), the yardstick for a one-thread call. None when the
    record is absent."""
    try:
        data = json.loads((cache.METRICS / "model_comparison.json").read_text(encoding="utf-8"))
        return data["results"]["models"]["bonsai-2-4b"]["latency_p50_s"]
    except (OSError, ValueError, KeyError):
        return None


def summarize(raw: dict) -> dict:
    bits = raw["bits"]
    walls = [b["wall_s"] for b in bits]
    calls = [c for b in bits for c in b["calls"]]
    lat = [c["seconds"] if c["ok"] else float("inf") for c in calls]
    lost = [b["prompt"] for b in bits if b["threads"] < generate.K]
    zero = [b["prompt"] for b in bits if b["threads"] == 0]
    spreads = [spread(b) for b in bits]
    spreads = [x for x in spreads if x is not None]
    queued = sum(x >= QUEUED_SPREAD for x in spreads)
    reasons: dict[str, int] = {}
    for c in calls:
        if not c["ok"]:
            reasons[c["reason"]] = reasons.get(c["reason"], 0) + 1
    after = raw["placement_after_run"]
    on_gpu = all(p["gpu_share"] is not None and p["gpu_share"] >= 1.0
                 and p["context_length"] in (None, generate.WRITER_CONTEXT)
                 for p in (raw["placement_after_warmup"], after))
    return {**raw, "n": len(bits), "bits_losing_a_thread": len(lost), "bits_with_zero_threads": len(zero),
            "bit_p50_s": cg.percentile(walls, 50), "bit_p95_s": cg.percentile(walls, 95),
            "call_p50_s": cg.percentile(lat, 50), "call_p95_s": cg.percentile(lat, 95),
            "failure_reasons": reasons, "bits_queued": queued,
            "spread_median": cg.percentile(spreads, 50), "exp24_call_p50_s": exp24_call_p50(),
            "bits_same_premise": sum(b["same_premise"] for b in bits), "writer_on_gpu": on_gpu,
            "bar": {"max_lost_bits": MAX_LOST_BITS, "lost_ok": len(lost) <= MAX_LOST_BITS, "zero_ok": not zero,
                    "gpu_ok": on_gpu}}


def run(n_bits: int = N_BITS, seed: str = SEED, digest_: str = "") -> dict:
    assert digest_ == digest(), "the prompt, timing or request changed since this call was built"
    client = generate.writer_client()
    if not getattr(client, "local", False):
        raise SystemExit("Exp 26 times the local writer: set JOKE_WRITER_PROVIDER=ollama")
    return summarize(measure(client, prompts(n_bits, seed)))


def params() -> dict:
    return {"n_bits": N_BITS, "seed": SEED, "digest_": digest()}


def stored() -> dict | None:
    return cache.load(NAME, params())


def table(res: dict) -> str:
    b = res["bar"]
    rows = ["| Check | Result | Bar |", "|---|---|---|",
            f"| Bits that lost a thread | {res['bits_losing_a_thread']} of {res['n']} | at most {b['max_lost_bits']} |",
            f"| Bits with no thread | {res['bits_with_zero_threads']} | 0 |",
            f"| Writer 100% on the GPU at context {generate.WRITER_CONTEXT} | {'yes' if res['writer_on_gpu'] else 'no'} | yes |",
            f"| Bit wall time p50 / p95 | {cg.fmt(res['bit_p50_s'])} s / {cg.fmt(res['bit_p95_s'])} s | beside the bar |",
            f"| Call p50 / p95 (Exp 24's one-call p50: {cg.fmt(res['exp24_call_p50_s'])} s) | {cg.fmt(res['call_p50_s'])} s / {cg.fmt(res['call_p95_s'])} s | beside the bar |",
            f"| Bits that queued (slowest call at least {QUEUED_SPREAD:.0f}x the fastest); median ratio {cg.fmt(res['spread_median'])} | {res['bits_queued']} of {res['n']} | beside the bar |",
            f"| Bits with two threads on the same premise | {res['bits_same_premise']} of {res['n']} | beside the bar |"]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("JOKE_WRITER_PROVIDER", "ollama")
    res = cache.run_or_load(NAME, run, rerun=a.rerun, **params())
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
