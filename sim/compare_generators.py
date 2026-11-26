"""Exp 24: which model should write the live show's bits? Claude Haiku 5.5, or a local Ollama model (bonsai-2:4b, qwen3:4b-instruct; D-101).
A local model is eligible only if Ollama puts it 100% on the GPU at context 8192 (it must fit beside Kokoro, which runs on
the CPU). A `reference` model (D-97) would be measured and on the ballot but never picked; none ships since D-104.

Usage (needs the Anthropic and Gemini keys in backend/.env and Ollama running, see backend/README.md "Comparing generator models"):
    python -m sim.compare_generators run --env-file backend/.env            # generate + safety, resumable, paced
    python -m sim.compare_generators export                                  # blind ballot for the user, local only
    python -m sim.compare_generators vote                                    # rank each prompt in the terminal
    python -m sim.compare_generators judge --export-bundle <dir> | --import-replies <file>   # a judge taught by your votes (Explorer)
    python -m sim.compare_generators score [--rerun]                         # numbers -> reports/metrics/model_comparison.json

Each prompt is one bit, generated exactly as a show does it (`generate.GeminiGenerator`, one call, 3 threads of 5
jokes, the production timeout, retries and deadline; a local model through Ollama's OpenAI-compatible endpoint), then
judged by the same safety filter for every model (Claude Haiku 5.5 with thinking off, as in a show since D-100). Latency, failures, safety pass rate, tokens and estimated cost
are measured; funniness is the user's blind ranking of one safe thread per model per prompt. Generated text, votes and
the ballot key stay under `data/model_comparison/` (gitignored); only numbers reach `reports/metrics`.

The bar is fixed in docs/plans/live-loop.md (T13) before any run (changed after review, before any data):
  eligible = p95 latency over ALL calls <= the generator's deadline (a timeout counts as above it) AND the 95% Wilson
  upper bound of the failure rate over EVERY attempt <= 15%, measured on 60 distinct prompts (30 are voted on);
  AND (local models, D-97) /api/ps reports 100% on the GPU, failing closed when it is missing or implausible;
  among the eligible, the best mean rank wins if the paired 95% CI of (next - best) excludes 0;
  otherwise funniness is called inconclusive and the cheaper (then faster) of the two is picked.
A model with no safe thread for a prompt is ranked last for it: a show would play the bank there.
A judge calibrated on the user's votes (sim/judge_generators.py) may add its rankings of prompts 31-60 if it passes its own bar.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "model_comparison"
sys.path.insert(0, str(ROOT / "backend"))
from app import generate, llm, safety  # noqa: E402
from sim import cache  # noqa: E402
from sim.blind import plain  # noqa: E402
from sim.env import AGES, ENERGIES, VENUES  # noqa: E402
from sim.evaluate_safety import wilson  # noqa: E402

DEADLINE_S = generate.GENERATOR_TIMING[1]  # wall time a bit may take over all attempts: what the bridge lines cover
BAR_FAILURE_UPPER = 0.15  # on the 95% Wilson upper bound of the failure rate, so it needs enough attempts to be met
N_PROMPTS = 60  # generated per model: every one counts toward the failure rate and the latency p95
N_BALLOT = 30  # the first of them are also voted on; power reasoning in docs/agent/decisions-detail.md D-94
SEED = 24
N_BOOT = 10_000
BITS_PER_5_SHOWS = 40  # 8 bits x 5 shows, the demo's horizon
# The filter is fixed across candidates (only the generator varies) and pinned here, not read from .env, so a run cannot
# change filters halfway. D-100: the show's filter, Haiku without thinking, off Gemini's free tier and its 503s.
SAFETY_PROVIDER, SAFETY_MODEL, SAFETY_THINKING = "anthropic", "claude-haiku-5-5", False
METRICS_NAME = "model_comparison"
SCHEMA_VERSION = "attempts-v2"


OLLAMA_URL = "http://127.0.0.1:11434/v1"  # Ollama's OpenAI-compatible endpoint: no key
OLLAMA_PROVIDER = "ollama"
LOCAL_JSON_MODE = "json_schema"  # D-101: local models are sent the reply schema, as Claude is; "json_object" is any JSON


@dataclass(frozen=True)
class Spec:
    key: str
    provider: str  # app.llm.from_env provider, or "ollama" (a local llm.Client, built by `make_client`)
    model: str  # for ollama the default tag; `OLLAMA_MODEL_ENV` overrides it at call time
    price_in: float  # USD per million tokens
    price_out: float
    bucket: str  # requests that share a quota, so they are spaced together
    gap_s: float  # minimum seconds between the end of one request and the start of the next in the bucket
    model_env: str = ""  # env var that overrides `model` (the tag `ollama list` shows may differ from the default)
    reference: bool = False  # measured and on the ballot, never eligible to be picked (D-97)

    @property
    def local(self) -> bool:
        return self.provider == OLLAMA_PROVIDER

    def tag(self) -> str:
        return (os.environ.get(self.model_env) if self.model_env else None) or self.model

    def base_url(self) -> str:
        return (os.environ.get("OLLAMA_URL") or OLLAMA_URL) if self.local else ""


# Claude's price is from the user (D-93); its output tokens include its thinking. The two local models cost $0, so the
# cost tie-break (decide) favors them: that is a property of the rule, not a finding. All prices are part of the scored
# result's key. D-96 explains why the Gemini writers were dropped. Local calls are sequential and need no spacing
# (one GPU), but keep the pacer's interface.
MODELS = {
    "qwen3-4b-instruct": Spec("qwen3-4b-instruct", OLLAMA_PROVIDER, "qwen3:4b-instruct", 0.0, 0.0, "ollama", 0.0, "QWEN_OLLAMA_TAG"),
    "bonsai-2-4b": Spec("bonsai-2-4b", OLLAMA_PROVIDER, "bonsai-2:4b", 0.0, 0.0, "ollama", 0.0, "BONSAI_OLLAMA_TAG"),
    "claude-haiku-5-5": Spec("claude-haiku-5-5", "anthropic", "claude-haiku-5-5", 0.10, 0.50, "anthropic", 1.0),
}  # D-104: gemma4:12b, the reference, was dropped: about 10 tokens/s with part of it on the CPU, so no bit fit 12 s
SAFETY_BUCKET = "anthropic-safety"  # the filter's own requests; a paid tier, so no free-tier spacing (Exp 19 reruns: 0 retries)
SAFETY_GAP_S = 0.0
MODE_ORDER = tuple(generate.MODES)


# --- prompts and cache -------------------------------------------------------------------------------------------

def prompts(n: int = N_PROMPTS, seed: int = SEED) -> list[dict]:
    """The same list for every model, and a prefix of a longer list: modes round-robin, each context drawn from its own
    seeded stream and never repeated for a mode. The first bit of a show, so `premises_already_told` is empty."""
    out, seen = [], set()
    for i in range(n):
        mode, rng = MODE_ORDER[i % len(MODE_ORDER)], random.Random(f"{seed}:{i}")
        while True:
            ctx = {"venue": rng.choice(VENUES), "energy": rng.choice(ENERGIES), "age": rng.choice(AGES)}
            if (mode, *ctx.values()) not in seen:
                break
        seen.add((mode, *ctx.values()))
        out.append({"id": f"p{i:02d}", "mode": mode, "context": ctx})
    return out


def gen_digest(seed: int = SEED) -> str:
    """Part of the cache path: editing the generator's or the filter's prompt, timing, retries or model starts a fresh
    comparison, not a mixed one."""
    parts = [generate.PROMPT_ID, generate.SYSTEM, generate.TEENS_RULE, json.dumps(generate.MODES, sort_keys=True),
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.RETRIES)),
             safety.PROMPT_ID, safety.SYSTEM, str((generate.FILTER_TIMING, generate.FILTER_RETRIES, SAFETY_PROVIDER, SAFETY_MODEL, SAFETY_THINKING)),
             json.dumps([[k, m.provider, m.tag(), m.base_url()] for k, m in MODELS.items()]),  # the candidates
             LOCAL_JSON_MODE, SCHEMA_VERSION, str(seed)]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8]


def run_dir(digest: str | None = None, root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or DATA) / f"gen-{digest or gen_digest()}"


def key_path(directory: pathlib.Path) -> pathlib.Path:
    """The ballot key lives apart from the ballot and the votes, so opening the run folder never shows it."""
    return directory.parent / "key" / f"{directory.name}.json"


def read_record(path: pathlib.Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_record(path: pathlib.Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # a Ctrl-C mid-write must not leave a half file that resume would trust


def final(rec: dict | None) -> dict | None:
    """The attempt a bit is judged on: the last successful one, else the last one made."""
    if not rec or not rec["attempts"]:
        return None
    return next((a for a in reversed(rec["attempts"]) if a["ok"]), rec["attempts"][-1])


# --- generating --------------------------------------------------------------------------------------------------

class Pacer:
    """Spaces requests that share a quota. Waiting happens before the timed call, never inside it."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.sleep, self.clock, self.last = sleep, clock, {}

    def wait(self, bucket: str, gap_s: float) -> None:
        if bucket in self.last and (left := self.last[bucket] + gap_s - self.clock()) > 0:
            self.sleep(left)

    def mark(self, bucket: str) -> None:
        self.last[bucket] = self.clock()


class ConfigError(RuntimeError):
    """The run stops instead of caching a pile of failures that only mean a wrong key, model name or setting."""


def failure_kind(e: Exception) -> str:
    """`config`: a key, model name or request the provider will never accept. `infra`: the provider was unavailable or
    slow, so the same request may work later. `model`: the model answered and the answer was unusable (a refusal, a
    cut-off, a reply that is not the shape asked for). Only config and infra failures are worth repeating."""
    if not isinstance(e, (llm.LLMError, generate.GenerationError)):
        return "config"  # nothing the clients raise is outside these two: it is a bug or a setting
    if isinstance(e, generate.GenerationError):
        return "model"
    if "credentials" in str(e) or (e.status is not None and 400 <= e.status < 500 and not e.retryable):
        return "config"
    return "infra" if e.retryable or (e.status is not None and e.status >= 500) else "model"


def is_timeout(reason: str | None) -> bool:
    return bool(reason) and any(w in reason.lower() for w in ("timeout", "deadline"))


def generate_bit(spec: Spec, prompt: dict, client, pacer: Pacer, now=time.perf_counter, first: bool = False) -> dict:
    """One call as a show makes it. Latency is wall time over all attempts, retries and backoff included. On the first
    call for a model (`first`) a config failure raises `ConfigError` instead of becoming data."""
    pacer.wait(spec.bucket, spec.gap_s)
    gen = generate.GeminiGenerator(client)
    before = (client.attempts, client.input_tokens, client.output_tokens)
    start = now()
    try:
        texts = [gen.generate(prompt["mode"], 0, j, prompt["context"]).texts for j in range(generate.JOKES)]
        threads, failure = [[texts[j][i] for j in range(generate.JOKES)] for i in range(generate.K)], None
    except Exception as e:  # the generator stores LLMError and GenerationError and re-raises them per joke
        threads, failure = None, e
    latency = now() - start
    pacer.mark(spec.bucket)
    kind = None if failure is None else failure_kind(failure)
    if first and spec.local and kind == "infra" and "ConnectError" in str(failure):
        kind = "config"  # nothing is listening: a setting, not an outage
    if first and kind == "config":
        hint = (f" (is Ollama running? `ollama serve`, and `ollama pull {spec.tag()}`)" if spec.local else "")
        raise ConfigError(f"{spec.key}: {type(failure).__name__}: {failure}{hint}")
    reason = None if failure is None else f"{type(failure).__name__}: {failure}"[:200]
    return {"ok": failure is None, "kind": kind, "reason": reason, "latency_s": latency,
            "timeout": is_timeout(reason) or (failure is not None and latency >= DEADLINE_S - llm.MIN_ATTEMPT_S),
            "requests": client.attempts - before[0], "input_tokens": client.input_tokens - before[1],
            "output_tokens": client.output_tokens - before[2], "threads": threads, "safety": None, "safety_tries": 0}


def judge_bit(att: dict, prompt: dict, safety_client, pacer: Pacer, now=time.perf_counter) -> dict:
    """Run the safety filter over all 15 lines. `safety[i][j]` is [ok, reason] for joke j of thread i. A verdict that
    only says the filter's own call failed is infrastructure, flagged so resume retries it."""
    age = prompt["context"]["age"]
    flat = [t for thread in att["threads"] for t in thread]
    pacer.wait(SAFETY_BUCKET, SAFETY_GAP_S)
    start = now()
    safety.prefetch(flat, age, safety_client)
    att["safety_latency_s"] = now() - start
    pacer.mark(SAFETY_BUCKET)
    verdicts = [[list(safety.check(t, age)) for t in thread] for thread in att["threads"]]
    if att.get("safety_rules") != safety.RULES_VERSION:
        att["safety_tries"] = 0  # new rules: the infrastructure retry budget starts again
    att["safety"] = verdicts
    att["safety_rules"] = safety.RULES_VERSION
    att["safety_infra_error"] = any(r.startswith("llm:error") for thread in verdicts for ok, r in thread if not ok)
    att["safety_tries"] += 1
    return att


MAX_SAFETY_TRIES = 3


def needs_safety(att: dict | None) -> bool:
    """A bit is (re)checked when it has no verdicts, when its verdicts came from older rules (D-105: only the safety
    step is redone, never the generation), or when the filter's own call failed and retries remain."""
    if not att or not att["ok"]:
        return False
    if att["safety"] is None or att.get("safety_rules") != safety.RULES_VERSION:
        return True
    return bool(att.get("safety_infra_error")) and att["safety_tries"] < MAX_SAFETY_TRIES


WARMUP_TIMEOUT_S = 300.0  # a cold load took 126 s for bonsai-2:4b and 35 s for mimo-v2.6:9b, since dropped (D-97), far past a bit's 12 s


def warmup_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "warmup.json"


class WarmupError(ConfigError):
    """A warm-up failed twice for a reason that is not a setting: the run stops, because the next timed call would pay
    the cold load and count it as the model's failure."""


def warm_up(spec: Spec, client, directory: pathlib.Path, now=time.perf_counter) -> bool:
    """One untimed request so the model is loaded, and alone, before its first timed call. Not an attempt: it is not in
    the model's records, latency or failures. Its wall time goes to `warmup.json` as `warmup_s` (the cold-load cost a show
    would pay if the writer were not preloaded), every try in `history`. Nothing listening or a missing model stops the
    run like a first call does; any other failure is retried once, and a second one stops the run (`WarmupError`)."""
    times, failure = [], None
    for _ in range(2):
        start = now()
        try:
            client.chat("Reply with JSON only.", '{"ready": true}', json_reply=True)
            failure = None
        except Exception as e:
            failure = e
        times.append(now() - start)
        if failure is None or failure_kind(failure) == "config" or "ConnectError" in str(failure):
            break
    rec = read_record(warmup_path(directory)) or {}
    history = rec.get(spec.key, {}).get("history", [])
    rec[spec.key] = {"warmup_s": times[-1], "ok": failure is None, "history": history + times}
    write_record(warmup_path(directory), rec)
    if failure is not None:
        hint = f"is Ollama running? `ollama serve`, and `ollama pull {spec.tag()}`"
        if failure_kind(failure) == "config" or "ConnectError" in str(failure):
            raise ConfigError(f"{spec.key} warm-up: {type(failure).__name__}: {failure} ({hint})")
        raise WarmupError(f"{spec.key} warm-up failed twice ({type(failure).__name__}: {failure}); nothing cached for this "
                          "model's block. Rerun `run` to try again")
    return True


def call_groups(models: list[str]) -> list[list[str]]:
    """Local models one by one, each loaded once (Ollama keeps a single one of them in VRAM, so alternating reloads
    every call); the hosted models together, prompt by prompt, at the place of the first of them."""
    out, remote = [], None
    for m in models:
        if MODELS[m].local:
            out.append([m])
        elif remote is None:
            remote = [m]
            out.append(remote)
        else:
            remote.append(m)
    return out


def run_comparison(n: int, models: list[str], make_client: Callable[[Spec], object], safety_client, *,
                   redo_failed: bool = False, directory: pathlib.Path | None = None, pacer: Pacer | None = None,
                   log: Callable[[str], None] = print, http: Callable[[str], dict] | None = None,
                   make_warm_client: Callable[[Spec], object] | None = None) -> dict:
    """Model-major for each local model, prompt-major among the hosted ones (`call_groups`), so a local model is loaded
    once. A file per model and prompt holds every attempt made, appended, never overwritten: `redo_failed` adds an
    attempt where the last one failed for a config or infrastructure reason, and the failure rate counts all of them.
    A local model gets one untimed warm-up in each run before its first call, and its GPU placement is measured right
    after it, when it is the only one loaded (D-97); a local model whose placement is not ok is warmed up and measured
    again at the start of its block even when every prompt is cached. Returns counts of what was done."""
    directory, pacer = directory or run_dir(), pacer or Pacer()
    clients = {m: make_client(MODELS[m]) for m in models}
    first = set(models)  # models whose first call this run has not happened yet
    warm = {m: False for m in models if MODELS[m].local}  # warmed up in this run
    tries = {m: 0 for m in warm}  # placement measurements this run, per local model
    done = {"generated": 0, "judged": 0, "cached": 0}
    def warm_and_measure(m: str) -> None:
        warm[m] = True
        client = make_warm_client(MODELS[m]) if make_warm_client else clients[m]
        if warm_up(MODELS[m], client, directory):
            tries[m] += 1
            record_placement(directory, MODELS[m], http or http_json, log)

    for group in call_groups(models):
        for m in group:  # a placement that is not ok is read again even when every prompt is cached: no timed call
            if m in warm and load_placement(directory).get(m, {}).get("status") != "ok":
                warm_and_measure(m)
        for prompt in prompts(n):
            for m in group:
                path = directory / m / f"{prompt['id']}.json"
                rec = read_record(path) or {"model": m, "prompt": prompt["id"], "attempts": []}
                last = rec["attempts"][-1] if rec["attempts"] else None
                if last is None or (redo_failed and not last["ok"] and last["kind"] in ("config", "infra")):
                    if m in warm and not warm[m]:
                        warm_and_measure(m)
                    rec["attempts"].append(generate_bit(MODELS[m], prompt, clients[m], pacer, first=m in first))
                    first.discard(m)
                    done["generated"] += 1
                    write_record(path, rec)
                    if m in tries and rec["attempts"][-1]["ok"] and tries[m] < MAX_PLACEMENT_TRIES:
                        tries[m] += 1
                        record_placement(directory, MODELS[m], http or http_json, log)
                else:
                    done["cached"] += 1
                att = final(rec)
                if needs_safety(att):
                    judge_bit(att, prompt, safety_client, pacer)
                    done["judged"] += 1
                    write_record(path, rec)
                log(f"{prompt['id']} {m}: {'ok' if att['ok'] else 'FAILED ' + str(att['reason'])} {att['latency_s']:.1f}s")
    return done


def make_client(spec: Spec):
    """The production generator's timing and retries, so latency and failures are what a show would see."""
    timeout, deadline = generate.GENERATOR_TIMING
    if spec.local:
        return llm.Client(spec.base_url(), "", spec.tag(), timeout, retries=generate.RETRIES, deadline_s=deadline,
                          extra_body=llm.OLLAMA_NO_THINK, json_schema=LOCAL_JSON_MODE == "json_schema")
    return llm.from_env(timeout, role="joke_writer", provider=spec.provider, model=spec.model,
                        retries=generate.RETRIES, deadline_s=deadline)


def make_warmup_client(spec: Spec):
    """The same local client with a timeout long enough for a cold load, and no retries."""
    return llm.Client(spec.base_url(), "", spec.tag(), WARMUP_TIMEOUT_S, retries=0, deadline_s=WARMUP_TIMEOUT_S,
                      extra_body=llm.OLLAMA_NO_THINK, json_schema=LOCAL_JSON_MODE == "json_schema")


def make_safety_client():
    """The production filter's timing and retries, the same filter for every candidate (D-100's, as in a show)."""
    timeout, deadline = generate.FILTER_TIMING
    return llm.from_env(timeout, role="safety_check", provider=SAFETY_PROVIDER, model=SAFETY_MODEL,
                        retries=generate.FILTER_RETRIES, deadline_s=deadline, thinking=SAFETY_THINKING)


# --- GPU placement (D-97) ----------------------------------------------------------------------------------------

GATE_CONTEXT = 8192  # the context the fit-beside-Kokoro gate is defined at
MAX_PLACEMENT_TRIES = 5  # measurements per local model per run: an unknown reading is retried, an ok one is kept
MIN_SIZE_SHARE_OF_FILE = 0.5  # a loaded model reported smaller than half its file is a bad report, not a small model


def http_json(url: str, timeout: float = 5.0) -> dict:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - the user's own Ollama server
        return json.loads(r.read().decode("utf-8"))


def ollama_root(spec: Spec) -> str:
    base = spec.base_url().rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _tag_of(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def _find(entries: list[dict], tag: str) -> dict | None:
    return next((e for e in entries if tag in {_tag_of(str(e.get("name", ""))), _tag_of(str(e.get("model", "")))}), None)


def measure_placement(spec: Spec, http: Callable[[str], dict] = http_json) -> dict:
    """Where Ollama put the model, from its own `/api/ps`. Fails closed: `status` is "ok" only when the model is listed,
    the report is plausible against the file size in `/api/tags` (Ollama 0.40.18 once reported 1 GB for an 8 GB model
    and once listed nothing right after a successful call), it was loaded at the gate's context, and a partial offload
    can be blamed on the model itself rather than on another loaded one. Anything else is "unknown"."""
    tag, root = spec.tag(), ollama_root(spec)
    rec = {"tag": tag, "status": "unknown", "size": None, "size_vram": None, "gpu_share": None, "file_size": None,
           "context_length": None, "context_ok": None, "others_loaded": 0, "note": ""}
    try:
        ps, tags = http(f"{root}/api/ps").get("models") or [], http(f"{root}/api/tags").get("models") or []
    except Exception as e:  # any failure to ask is "unknown", never a guess
        rec["note"] = f"could not ask Ollama: {type(e).__name__}"
        return rec
    mine, file = _find(ps, tag), _find(tags, tag)
    rec["others_loaded"] = sum(e is not mine for e in ps)
    if mine is None:
        rec["note"] = "model missing from /api/ps"
        return rec
    size, vram = mine.get("size"), mine.get("size_vram")
    rec["context_length"] = mine.get("context_length")
    if not isinstance(size, (int, float)) or not isinstance(vram, (int, float)) or size <= 0 or not 0 <= vram <= size:
        rec["note"] = "implausible /api/ps report"
        return rec
    rec.update(size=size, size_vram=vram, gpu_share=vram / size, file_size=(file or {}).get("size"))
    if not rec["file_size"]:
        rec["note"] = "model missing from /api/tags: the report cannot be checked"
    elif size < MIN_SIZE_SHARE_OF_FILE * rec["file_size"]:
        rec["note"] = "/api/ps size is below half the file size"
    elif rec["context_length"] is not None and rec["context_length"] != GATE_CONTEXT:
        rec["context_ok"] = False
        rec["note"] = f"loaded at context {rec['context_length']}, not {GATE_CONTEXT}"
    elif rec["gpu_share"] < 1.0 and rec["others_loaded"]:
        rec["note"] = "other models were loaded, so the partial offload cannot be blamed on this one"
    else:
        rec.update(status="ok", context_ok=None if rec["context_length"] is None else True)
    return rec


def placement_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "placement.json"


def load_placement(directory: pathlib.Path) -> dict[str, dict]:
    return read_record(placement_path(directory)) or {}


def describe_placement(key: str, rec: dict) -> str:
    """One plain line for the log: what was read and what it means for the gate."""
    ctx = rec.get("context_length")
    if rec["status"] != "ok":
        hint = ("; restart with OLLAMA_CONTEXT_LENGTH=8192 and rerun run" if rec.get("context_ok") is False
                else "; rerun run to measure again")
        return f"placement {key}: unknown: {rec['note']} -> not eligible{hint}"
    if rec["gpu_share"] >= 1.0:
        return f"placement {key}: 100% -> passes the GPU gate"
    return f"placement {key}: {rec['gpu_share']:.0%} on GPU at ctx {ctx} -> not eligible (partial GPU)"


def record_placement(directory: pathlib.Path, spec: Spec, http: Callable[[str], dict] = http_json,
                     log: Callable[[str], None] = lambda s: None) -> None:
    """Measured once and kept when ok; an unknown reading is replaced by the next measurement. Each new reading is logged."""
    recs = load_placement(directory)
    if recs.get(spec.key, {}).get("status") == "ok":
        return
    recs[spec.key] = measure_placement(spec, http)
    write_record(placement_path(directory), recs)
    log(describe_placement(spec.key, recs[spec.key]))


def placement_gate(spec: Spec | None, placement: dict | None) -> str | None:
    """Why a local model fails the gate, or None when it passes (hosted models have no placement to check)."""
    if spec is None or not spec.local:
        return None
    if not placement or placement.get("status") != "ok":
        return "placement unknown"
    return None if placement["gpu_share"] >= 1.0 else f"partial GPU {placement['gpu_share']:.0%}"


# --- blind ballot ------------------------------------------------------------------------------------------------

LETTERS = "ABCD"  # up to four models a prompt, references included; three ship since D-104


def playable_thread(rec: dict | None) -> int | None:
    """The first thread whose five lines all passed the filter, with the filter's own calls not failing: the thread a
    show would play unchanged. None when the model has no such thread for this prompt."""
    att = final(rec)
    if not att or not att["ok"] or att["safety"] is None or att.get("safety_infra_error"):
        return None
    return next((i for i, thread in enumerate(att["safety"]) if all(ok for ok, _ in thread)), None)


def read_votes(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["prompt"]: r["ranking"].strip().upper() for r in csv.DictReader(f)}


def write_votes(path: pathlib.Path, votes: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "ranking"])
        w.writerows(sorted(votes.items()))


def export_ballot(models: list[str], n: int, directory: pathlib.Path | None = None, seed: int = SEED) -> dict:
    """Writes ballot.md and votes.csv under the run directory and the key apart from them. A model with no playable
    thread for a prompt is not shown and is ranked last for it (a show would play the bank), so no prompt is dropped;
    a prompt with fewer than two models shown needs no vote. Incremental: prompts already in the key keep their
    letters and votes. A prompt not yet generated for every model waits (`pending`): export after `run` finishes."""
    directory = directory or run_dir()
    directory.mkdir(parents=True, exist_ok=True)
    kp, votes_path = key_path(directory), directory / "votes.csv"
    key = json.loads(kp.read_text(encoding="utf-8")) if kp.exists() else {}
    votes, pending = read_votes(votes_path), []
    by_id = {p["id"]: p for p in prompts(max([n] + [int(k[1:]) + 1 for k in key]), seed)}
    for pid in prompts(n, seed):
        pid = pid["id"]
        if pid in key:
            continue
        recs = {m: read_record(directory / m / f"{pid}.json") for m in models}
        if any(r is None or not r["attempts"] for r in recs.values()):
            pending.append(pid)
            continue
        picks = {m: playable_thread(recs[m]) for m in models}
        shown = [m for m in models if picks[m] is not None]
        random.Random(f"{seed}:ballot:{pid}").shuffle(shown)
        key[pid] = {"letters": dict(zip(LETTERS, shown)), "threads": {m: picks[m] for m in shown},
                    "last": [m for m in models if picks[m] is None]}
        if len(shown) >= 2:
            votes.setdefault(pid, "")
    lines = ["# Blind ballot: rank the options for each setting, funniest first (e.g. BAC)\n"]
    for pid in sorted(key):
        if len(key[pid]["letters"]) < 2:
            continue
        ctx = by_id[pid]["context"]
        lines.append(f"\n## {pid}: {ctx['venue']}, {ctx['energy']} energy, {ctx['age']} audience\n")
        for letter, m in key[pid]["letters"].items():
            thread = final(read_record(directory / m / f"{pid}.json"))["threads"][key[pid]["threads"][m]]
            lines.append(f"\n**{letter}**\n" + "\n".join(f"{j + 1}. {plain(t)}" for j, t in enumerate(thread)) + "\n")
    kp.parent.mkdir(parents=True, exist_ok=True)
    kp.write_text(json.dumps(key, indent=1, sort_keys=True), encoding="utf-8")
    (directory / "ballot.md").write_text("".join(lines), encoding="utf-8")
    write_votes(votes_path, votes)
    return {"in_key": len(key), "to_vote": len(votes), "ranked_last": {p: k["last"] for p, k in sorted(key.items()) if k["last"]},
            "pending": pending}


def parse_ranking(text: str, letters: str = LETTERS) -> str | None:
    """'bac' -> 'BAC' when it names each of `letters` once; anything else is None."""
    s = "".join(c for c in text.upper() if c.isalpha())
    return s if sorted(s) == sorted(letters) else None


def vote_interactively(directory: pathlib.Path | None = None, read=input, out=print) -> int:
    """Shows each unranked prompt and saves after every answer, so quitting loses nothing."""
    directory = directory or run_dir()
    blocks, current = {}, None
    for line in (directory / "ballot.md").read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("## "):
            current = line[3:].split(":")[0]
        if current:
            blocks[current] = blocks.get(current, "") + line
    key = json.loads(key_path(directory).read_text(encoding="utf-8"))
    votes_path, count = directory / "votes.csv", 0
    votes = read_votes(votes_path)
    todo = [p for p, r in sorted(votes.items()) if not parse_ranking(r, "".join(key[p]["letters"]))]
    for i, pid in enumerate(todo, 1):
        letters = "".join(key[pid]["letters"])
        out(f"\n[{i}/{len(todo)}]\n{blocks[pid]}")
        while True:
            answer = read(f"ranking of {letters}, best first (e.g. {letters[::-1]}), s to skip, q to quit: ").strip().lower()
            if answer in ("s", "q") or parse_ranking(answer, letters):
                break
            out(f"use each of {', '.join(letters)} once")
        if answer == "q":
            break
        if answer != "s":
            votes[pid] = parse_ranking(answer, letters)
            write_votes(votes_path, votes)
            count += 1
    return count


# --- scoring -----------------------------------------------------------------------------------------------------

def percentile(xs: list[float], q: float) -> float | None:
    """Nearest-rank percentile: the ceil(q/100 x n)-th smallest value. It accepts `inf` (a timeout is above any deadline)
    without interpolating into it, so the p95 is infinite exactly when more than 5% of the calls are."""
    xs = sorted(xs)
    return xs[max(math.ceil(q / 100 * len(xs)), 1) - 1] if xs else None


def call_latency(a: dict) -> float:
    """What a call counts as in the latency bar: its wall time, or infinity when it timed out or used the deadline up."""
    return math.inf if a["timeout"] else a["latency_s"]


def call_stats(attempts: list[dict], spec: Spec) -> dict:
    """Per-model numbers from every attempt made (a retried failure stays in). Latency is over ALL attempts, a timeout
    counting as above the deadline. Cost covers every attempt's tokens, since a provider bills failed ones too."""
    ok = [a for a in attempts if a["ok"]]
    judged = [a for a in ok if a["safety"] is not None and not a.get("safety_infra_error")]
    lines = [v for a in judged for thread in a["safety"] for v in thread]
    threads = [t for a in judged for t in a["safety"]]
    tin, tout = sum(a["input_tokens"] for a in attempts), sum(a["output_tokens"] for a in attempts)
    cost = (tin * spec.price_in + tout * spec.price_out) / 1e6
    n, k = len(attempts), len(attempts) - len(ok)
    reasons: dict[str, int] = {}
    for a in attempts:
        if not a["ok"]:
            reasons[a["reason"][:60]] = reasons.get(a["reason"][:60], 0) + 1
    lat = [call_latency(a) for a in attempts]
    p50, p95 = percentile(lat, 50), percentile(lat, 95)
    return {"model": spec.tag(), "n_attempts": n, "n_failed": k, "failure_rate": k / n if n else None,
            "failure_ci95": wilson(k, n), "failure_reasons": reasons,
            "n_timeouts": sum(a["timeout"] for a in attempts),
            "latency_p50_s": None if p50 in (None, math.inf) else p50,
            "latency_p95_s": None if p95 in (None, math.inf) else p95,
            "p95_over_deadline": p95 == math.inf,
            "latency_max_ok_s": max((a["latency_s"] for a in ok), default=None),
            "requests_per_attempt": sum(a["requests"] for a in attempts) / n if n else None,
            "n_judged": len(judged), "line_pass_rate": sum(v[0] for v in lines) / len(lines) if lines else None,
            "thread_pass_rate": sum(all(v[0] for v in t) for t in threads) / len(threads) if threads else None,
            "playable_rate": (sum(any(all(v[0] for v in t) for t in a["safety"]) for a in judged) / len(judged)
                              if judged else None),
            "input_tokens_per_attempt": tin / n if n else None, "output_tokens_per_attempt": tout / n if n else None,
            "price_per_mtok": [spec.price_in, spec.price_out],
            "cost_per_bit_usd": cost / n if n else None,
            "cost_5_shows_usd": cost / n * BITS_PER_5_SHOWS if n else None}


def boot_idx(n_items: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """One resampling of prompts shared by every statistic, so models and gaps are bootstrapped on the same draws."""
    return np.random.default_rng(seed).integers(0, n_items, (n_boot, n_items))


def boot_ci(x: np.ndarray, idx: np.ndarray) -> list[float]:
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rerank(r: dict[str, float], keep: list[str]) -> dict[str, float]:
    """Ranks among the models in `keep`, order preserved; equal ranks share their average place."""
    out, order, i = {}, sorted(keep, key=lambda m: r[m]), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and r[order[j + 1]] == r[order[i]]:
            j += 1
        for m in order[i:j + 1]:
            out[m] = (i + j) / 2 + 1
        i = j + 1
    return out


def restrict(ranks: dict[str, dict[str, float]], keep: list[str]) -> dict[str, dict[str, float]]:
    return {pid: rerank(r, keep) for pid, r in ranks.items()}


def ranks_from_votes(votes: dict[str, str], key: dict, models: list[str]) -> dict[str, dict[str, float]]:
    """{prompt: {model: rank}} (1 = funniest), over the prompts the key covers, for `models` only. The models shown are
    ranked by the vote (a prompt with one shown model needs none); models with no safe thread share the last places.
    A prompt whose vote is missing or invalid is left out, and so is one that lacks a requested model."""
    out = {}
    for pid, k in key.items():
        letters = k["letters"]
        if len(letters) >= 2:
            ranking = parse_ranking(votes.get(pid, ""), "".join(letters))
            if not ranking:
                continue
            order = [letters[c] for c in ranking]
        else:
            order = list(letters.values())
        full = assemble_ranks({m: float(i) for i, m in enumerate(order, 1)}, k["last"])
        if set(models) <= set(full):
            out[pid] = rerank(full, models)
    return out


def assemble_ranks(shown: dict[str, float], last: list[str]) -> dict[str, float]:
    """Ranks for every model of a prompt: the shown ones as given (1..s, or means of such ranks), the models with no
    safe thread tied for the last places."""
    n = len(shown) + len(last)
    return {**shown, **{m: (len(shown) + 1 + n) / 2 for m in last}}


def rank_stats(ranks: dict[str, dict[str, float]], models: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    ranks = restrict(ranks, models)
    pids = sorted(ranks)
    if not pids:
        return {"n": 0, "mean_rank": {}, "ci95": {}, "diff": {}}
    idx = boot_idx(len(pids), n_boot, seed)
    arr = {m: np.array([ranks[p][m] for p in pids], float) for m in models}
    return {"n": len(pids), "mean_rank": {m: float(arr[m].mean()) for m in models},
            "ci95": {m: boot_ci(arr[m], idx) for m in models},
            "diff": {f"{a}|{b}": {"mean": float((arr[a] - arr[b]).mean()), "ci95": boot_ci(arr[a] - arr[b], idx)}
                     for a in models for b in models if a < b}}


def diff_ci(ranks: dict[str, dict[str, float]], worse: str, better: str, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired bootstrap over prompts of mean(rank worse - rank better): positive means `better` is funnier."""
    pids = sorted(ranks)
    d = np.array([ranks[p][worse] - ranks[p][better] for p in pids], float)
    return float(d.mean()), boot_ci(d, boot_idx(len(pids), n_boot, seed))


def eligibility(stats: dict, deadline_s: float = DEADLINE_S, bar_upper: float = BAR_FAILURE_UPPER,
                placement: dict[str, dict] | None = None) -> dict[str, dict]:
    """Per model: the bar terms, whether all hold, and why not. Latency: p95 over all calls at most the deadline, a
    timeout counting as above it. Failures: the 95% Wilson upper bound over every attempt at most `bar_upper`. A model
    with no attempt is not eligible. D-97: a reference model is never eligible, and a local one is only if Ollama put
    it 100% on the GPU (a missing `placement` is unknown, which fails)."""
    out = {}
    for m, s in stats.items():
        spec, p95, ci = MODELS.get(m), s["latency_p95_s"], s.get("failure_ci95")
        gate = None if spec and spec.reference else placement_gate(spec, (placement or {}).get(m))  # a reference needs no fit
        out[m] = {"latency_ok": not s.get("p95_over_deadline", False) and p95 is not None and p95 <= deadline_s,
                  "failure_ok": ci is not None and ci[1] <= bar_upper, "reference": bool(spec and spec.reference),
                  "placement_ok": gate is None}
        out[m]["reasons"] = [r for r, bad in (("reference", out[m]["reference"]), (gate, gate is not None),
                                              ("latency over the deadline", not out[m]["latency_ok"]),
                                              ("failure rate over the bar", not out[m]["failure_ok"])) if bad]
        out[m]["eligible"] = not out[m]["reasons"]
    return out


def decide(stats: dict, ranks: dict[str, dict[str, float]], n_boot: int = N_BOOT, seed: int = SEED,
           deadline_s: float = DEADLINE_S, bar_upper: float = BAR_FAILURE_UPPER,
           placement: dict[str, dict] | None = None) -> dict:
    """The pre-fixed bar. Ranks are redone among the eligible models before comparing them."""
    elig = eligibility(stats, deadline_s, bar_upper, placement)
    ok = sorted(m for m, e in elig.items() if e["eligible"])
    out = {"eligibility": elig, "eligible": ok, "pick": None, "best": None, "second": None, "funniness_clear": None,
           "diff": None, "reason": ""}
    if not ok:
        out["reason"] = "no model is eligible (latency, failures, GPU placement); the generator stays as it is"
        return out
    if len(ok) == 1:
        out.update(pick=ok[0], reason="the only eligible model")
        return out
    if not ranks:
        out["reason"] = "waiting for the user's votes"
        return out
    re_ranked = restrict(ranks, ok)
    mean = {m: float(np.mean([r[m] for r in re_ranked.values()])) for m in ok}
    # D-108: an exact tie in mean rank goes to the faster model; the name only settles a tie in speed too
    best, second = sorted(ok, key=lambda m: (mean[m], stats[m]["latency_p50_s"] or math.inf, m))[:2]
    d, ci = diff_ci(re_ranked, second, best, n_boot, seed)
    clear = ci[0] > 0
    out.update(best=best, second=second, diff={"mean": d, "ci95": ci}, funniness_clear=clear, mean_rank_eligible=mean)
    if clear:
        out.update(pick=best, reason=f"{best} is funnier than {second}: the paired CI of the rank gap excludes 0")
    else:
        cheaper = min((best, second), key=lambda m: (stats[m]["cost_per_bit_usd"], stats[m]["latency_p50_s"] or 0, m))
        out.update(pick=cheaper, reason=f"funniness is inconclusive between {best} and {second} (the paired CI "
                                        "includes 0), so the cheaper, then faster, of the two")
    return out


# --- power -------------------------------------------------------------------------------------------------------

RANK_SD_NO_SIGNAL = math.sqrt(2 / 3)  # sd of one model's rank among 3 when every order is equally likely
RANK_GAP_SD_NO_SIGNAL = math.sqrt(2.0)  # sd of the gap between two models' ranks in that case


def halfwidth(n: int, sd: float = RANK_SD_NO_SIGNAL) -> float:
    """Half-width of a 95% interval for one model's mean rank over n prompts (normal approximation)."""
    return 1.96 * sd / math.sqrt(n)


def gap_halfwidth(n: int) -> float:
    """Half-width of the interval for the gap between two models' mean ranks: the one the decision turns on."""
    return halfwidth(n, RANK_GAP_SD_NO_SIGNAL)


def simulate_ranks(n: int, gap: float, rng: np.random.Generator, models: list[str]) -> dict[str, dict[str, float]]:
    """Votes from a voter for whom `models[0]` is `gap` latent sd funnier than every other model on the ballot
    (Thurstone: score = mean + N(0, 1) per model per prompt)."""
    k = len(models)
    mu = np.array([gap] + [0.0] * (k - 1))
    ranks = np.argsort(np.argsort(-(mu + rng.standard_normal((n, k))), axis=1), axis=1) + 1
    return {f"p{i:02d}": {m: float(ranks[i, j]) for j, m in enumerate(models)} for i in range(n)}


def simulate_decision_power(n: int, gap: float, reps: int = 300, n_boot: int = 1000, seed: int = SEED,
                            eligible: int = 3, references: int = 0) -> float:
    """How often the real `decide()` calls the truly funnier model clearly funnier: `eligible` models, all passing the
    bar, plus `references` ballot-only ones that are never eligible (the votes are re-ranked without them); best and
    second picked by observed mean rank, the gap interval a paired bootstrap over prompts. The defaults are D-94's
    three-model ballot; D-97 usually leaves two eligible beside two references."""
    rng = np.random.default_rng(seed)
    models = [f"m{i}" for i in range(eligible + references)]
    stats = {m: {"latency_p95_s": 5.0, "failure_ci95": [0.0, 0.05], "failure_rate": 0.0, "n_attempts": n,
                 "cost_per_bit_usd": 0.001, "latency_p50_s": 2.0} for m in models[:eligible]}  # references are not scored
    hits = 0
    for r in range(reps):
        d = decide(stats, simulate_ranks(n, gap, rng, models), n_boot=n_boot, seed=seed + r)
        hits += bool(d["funniness_clear"]) and d["pick"] == models[0]
    return hits / reps


# --- the experiment ----------------------------------------------------------------------------------------------

def load_attempts(directory: pathlib.Path, models: list[str], n: int) -> dict[str, list[dict]]:
    """Every attempt made for each model over the first `n` prompts."""
    out = {}
    for m in models:
        recs = [read_record(directory / m / f"{p['id']}.json") for p in prompts(n)]
        out[m] = [a for r in recs if r for a in r["attempts"]]
    return out


def _digest(paths) -> str:
    body = b"".join(p.read_bytes() for p in paths if p.exists())
    return hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest()[:12]


def ballot_digest(directory: pathlib.Path) -> str:
    """The user's votes and the key, never the text: what the judge was calibrated on."""
    return _digest((key_path(directory), directory / "votes.csv"))


def placement_digest(directory: pathlib.Path) -> str:
    """A new or changed GPU measurement recomputes the experiment: it can change who is eligible."""
    return _digest((placement_path(directory),))


def judge_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "judge_result.json"


def votes_digest(directory: pathlib.Path) -> str:
    """The votes, the key and the judge's result: a changed vote or a new judge run recomputes the experiment."""
    return _digest((key_path(directory), directory / "votes.csv", judge_path(directory)))


def judge_ranks(directory: pathlib.Path, models: list[str], n_ballot: int, n_prompts: int = N_PROMPTS) -> tuple[dict, dict]:
    """(ranks to add to the user's votes, the judge's summary). The ranks are the judge's for prompts after the
    ballot, only when its result is current (calibrated on these very votes, with this judge prompt and order scheme, for
    this many prompts), complete (every call answered) and passed its bar; otherwise empty."""
    from sim import judge_generators as jg  # lazy: the judge module imports this one
    jp = judge_path(directory)
    if not jp.exists():
        return {}, {"state": "not run"}
    res = json.loads(jp.read_text(encoding="utf-8"))
    if (res.get("ballot_digest") != ballot_digest(directory) or res.get("n_ballot") != n_ballot or res.get("n_prompts") != n_prompts
            or res.get("prompt_digest") != jg.prompt_digest() or res.get("scheme") != jg.SCHEME):
        return {}, {"state": "stale", "joined": False,
                    "reason": "the votes, the number of prompts, or the judge's prompt or order scheme changed since it ran"}
    if res.get("incomplete"):
        return {}, {"state": "incomplete", "joined": False, "reason": res["reason"]}
    summary = {k: v for k, v in res.items() if k not in ("judge_ranks", "loo")}
    summary["state"] = "joined" if res["joined"] else "not joined"
    if not res["joined"]:
        return {}, summary
    ids = {p["id"] for p in prompts(n_prompts)[n_ballot:]}
    ranks = {pid: rerank(r, models) for pid, r in res["judge_ranks"].items() if pid in ids and set(models) <= set(r)}
    return ranks, summary


def current_params(n: int = N_PROMPTS, models: tuple[str, ...] = tuple(MODELS), n_ballot: int = N_BALLOT) -> dict:
    return {"gen": gen_digest(), "votes": votes_digest(run_dir()),
            "placement": placement_digest(run_dir()), "n_prompts": n, "n_ballot": n_ballot,
            "models": list(models), "seed": SEED,
            "n_boot": N_BOOT, "deadline_s": DEADLINE_S, "bar_failure_upper": BAR_FAILURE_UPPER,
            "prices": {m: [MODELS[m].price_in, MODELS[m].price_out] for m in models}}


def run(gen: str = "", votes: str = "", n_prompts: int = N_PROMPTS, n_ballot: int = N_BALLOT,
        models: list[str] | None = None, seed: int = SEED,
        n_boot: int = N_BOOT, deadline_s: float = DEADLINE_S, bar_failure_upper: float = BAR_FAILURE_UPPER,
        prices: dict | None = None, placement: str = "") -> dict:
    """Scores the saved generations and votes. Numbers only: no joke text and no letters leave the data directory."""
    models = models or list(MODELS)
    assert gen == gen_digest(seed), "the generator or filter prompt, model or timing changed since these calls were made"
    directory = run_dir(gen)
    assert votes == votes_digest(directory), "votes or key changed since this call was built"
    assert placement == placement_digest(directory), "GPU placement changed since this call was built"
    placed = load_placement(directory)
    attempts = load_attempts(directory, models, n_prompts)
    stats = {m: call_stats(attempts[m], MODELS[m]) for m in models}
    kp = key_path(directory)
    key = json.loads(kp.read_text(encoding="utf-8")) if kp.exists() else {}
    key = {p: k for p, k in key.items() if p in {q["id"] for q in prompts(n_ballot, seed)}}
    user_ranks = ranks_from_votes(read_votes(directory / "votes.csv"), key, models)
    extra, judge = judge_ranks(directory, models, n_ballot, n_prompts)  # empty unless the judge agreed with the user
    ranks = {**user_ranks, **extra}
    by_id = {p["id"]: p for p in prompts(n_prompts, seed)}
    return {"models": stats, "n_prompts": n_prompts, "n_ballot": n_ballot, "n_key": len(key), "n_scored": len(ranks),
            "n_user": len(user_ranks), "n_judge": len(extra), "judge": judge,
            "placement": {m: placed[m] for m in models if m in placed},
            "n_ranked_last": {m: sum(m in k["last"] for k in key.values()) for m in models},
            "votes": rank_stats(ranks, models, n_boot, seed), "votes_user_only": rank_stats(user_ranks, models, n_boot, seed),
            "per_prompt": [{"prompt": p, "mode": by_id[p]["mode"], "rater": "user" if p in user_ranks else "judge",
                            "ranks": ranks[p]} for p in sorted(ranks)],
            "decision": decide(stats, ranks, n_boot, seed, deadline_s, bar_failure_upper, placed)}


def stored() -> dict | None:
    path = cache.METRICS / f"{METRICS_NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def fmt(x, spec="{:.2f}", none="-"):
    return none if x is None else spec.format(x)


def table(res: dict) -> str:
    rows = ["| Model | Attempts | Failed (95% upper) | p50 | p95 | Line pass | Playable | Mean rank (95% CI) | Cost, 5 shows | GPU | Eligible |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    mean, ci = res["votes"].get("mean_rank", {}), res["votes"].get("ci95", {})
    for m, s in res["models"].items():
        e, pl = res["decision"]["eligibility"][m], res.get("placement", {}).get(m)
        gpu = "-" if not MODELS[m].local else (f"{pl['gpu_share']:.0%}" if pl and pl["status"] == "ok" else "unknown")
        rank = f"{mean[m]:.2f} ({ci[m][0]:.2f} to {ci[m][1]:.2f})" if m in mean else "-"
        p95 = "over deadline" if s["p95_over_deadline"] else fmt(s["latency_p95_s"], "{:.1f} s")
        rows.append(f"| {m} | {s['n_attempts']} | {s['n_failed']} ({s['failure_ci95'][1]:.0%}) | "
                    f"{fmt(s['latency_p50_s'], '{:.1f} s')} | {p95} | {fmt(s['line_pass_rate'], '{:.0%}')} | "
                    f"{fmt(s['playable_rate'], '{:.0%}')} | {rank} | {fmt(s['cost_5_shows_usd'], '${:.3f}')} | "
                    f"{gpu} | {'yes' if e['eligible'] else 'no: ' + ', '.join(e['reasons'])} |")
    return "\n".join(rows)


def verdict(res: dict) -> str:
    d = res["decision"]
    return f"Pick: {d['pick'] or 'none'}. {d['reason']}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["run", "export", "vote", "judge", "score"])
    ap.add_argument("--env-file")
    ap.add_argument("--n", type=int, default=N_PROMPTS, help="prompts to generate per model (all count toward failures and latency)")
    ap.add_argument("--ballot", type=int, default=N_BALLOT, help="how many of the first prompts are voted on")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--redo-failed", action="store_true",
                    help="add an attempt where the last one failed for a key, setting or provider-availability reason")
    ap.add_argument("--judge-model", default=None, help="the judge (default qwen2.5-7b-instruct, run on Explorer)")
    ap.add_argument("--judge-provider", default=None, choices=["explorer", "ollama", "anthropic", "gemini"],
                    help="judge: ollama uses any Ollama model (default qwen3.8:27b) through its /v1 endpoint or an Explorer job")
    ap.add_argument("--export-bundle", metavar="DIR", help="judge: write the calls for Explorer (calls.jsonl) to DIR and stop")
    ap.add_argument("--import-replies", metavar="FILE", help="judge: read the cluster's replies.jsonl back and score the judge")
    ap.add_argument("--smoke", type=int, default=0, help="with --export-bundle: only the calls of the first N prompts")
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file, override=True)
    models = a.models.split(",")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        sys.exit(f"unknown models {unknown}; choose from {list(MODELS)}")
    if a.command == "run":
        if "claude-haiku-5-5" in models and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set: add a paid key to backend/.env (or pass --models without it)")
        try:
            print(run_comparison(a.n, models, make_client, make_safety_client(), redo_failed=a.redo_failed,
                                make_warm_client=make_warmup_client))
        except ConfigError as e:
            sys.exit(f"stopped, nothing cached for the failing call: {e}")
    elif a.command == "export":
        print(export_ballot(models, a.ballot))
        print(f"ballot: {run_dir() / 'ballot.md'}  votes: {run_dir() / 'votes.csv'}")
    elif a.command == "judge":
        from sim import judge_generators
        judge_generators.main(models, a.n, a.ballot, a.judge_model, a.export_bundle, a.import_replies, a.smoke, a.judge_provider)
    elif a.command == "vote":
        print(f"{vote_interactively()} votes saved to {run_dir() / 'votes.csv'}")
    else:
        res = cache.run_or_load(METRICS_NAME, run, rerun=a.rerun, **current_params(a.n, tuple(models), a.ballot))
        print(table(res))
        print(verdict(res))


if __name__ == "__main__":
    main()
