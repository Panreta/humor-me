"""Exp 24 pre-screen of local models (D-101): which small Ollama models could write a bit at all, with the reply schema
sent (`json_schema`) and without it (`json_object`).

Usage (Ollama running with OLLAMA_CONTEXT_LENGTH=8192, the tags pulled):
    python -m sim.screen_local [--rerun]        # both arms -> reports/metrics/local_screen_<mode>.json

Five bits per model on prompts 60 to 64, outside the 60 of the real run, each generated exactly as the harness does it:
the harness's warm-up client, `measure_placement`, and `generate_bit` at the production timing. Each model is unloaded
after its arm so the next loads alone. Numbers only: no generated text, and failure reasons are kept as the exception's
class name. Five bits is a smoke screen, not a measurement: 5 of 5 has a Wilson 95% interval of about 57 to 100%.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import generate_v3 as generate, llm  # noqa: E402  frozen gen-v3, as Exp 24 ran it (D-110)
from sim import cache  # noqa: E402
from sim import compare_generators as cg  # noqa: E402

TAGS = ["bonsai-2:4b", "qwen3.5:4b-q4_K_M", "ministral-3:3b", "gemma3:4b", "qwen3:4b-instruct"]
PROMPT_IDS = [f"p{i}" for i in range(60, 65)]
MODES = ("json_object", "json_schema")
NAME = "local_screen"


def prompt_digest() -> str:
    """What the screened bit is made of: a changed prompt, timing or request starts a new record."""
    parts = [generate.PROMPT_ID, generate.SYSTEM, generate.TEENS_RULE, json.dumps(generate.MODES, sort_keys=True),
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.RETRIES)),
             json.dumps(llm.OLLAMA_NO_THINK, sort_keys=True), str(cg.GATE_CONTEXT)]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8]


def unload(spec: cg.Spec) -> None:
    """Ollama drops a model when asked to keep it for 0 seconds, so the next one loads alone."""
    import urllib.request
    req = urllib.request.Request(f"{cg.ollama_root(spec)}/api/generate", method="POST",
                                 data=json.dumps({"model": spec.tag(), "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()  # noqa: S310 - the user's own Ollama server
    except Exception:  # noqa: BLE001 - an unload that fails only risks a reload, which the placement read would show
        pass


def make_client(spec: cg.Spec, mode: str) -> llm.Client:
    timeout, deadline = generate.GENERATOR_TIMING
    return llm.Client(spec.base_url(), "", spec.tag(), timeout, retries=generate.RETRIES, deadline_s=deadline,
                      extra_body=llm.OLLAMA_NO_THINK, json_schema=mode == "json_schema")


def screen_one(tag: str, mode: str, prompt_ids: list[str], http=cg.http_json) -> dict:
    spec = cg.Spec(tag, cg.OLLAMA_PROVIDER, tag, 0.0, 0.0, "ollama", 0.0)
    by_id = {p["id"]: p for p in cg.prompts(max(int(i[1:]) for i in prompt_ids) + 1)}
    out = {"warmup_ok": None, "warmup_s": None, "placement": None, "bits": []}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out["warmup_ok"] = cg.warm_up(spec, cg.make_warmup_client(spec), pathlib.Path(tmp))
            out["warmup_s"] = cg.read_record(cg.warmup_path(pathlib.Path(tmp)))[tag]["warmup_s"]
    except cg.ConfigError as e:
        out["warmup_ok"], out["warmup_error"] = False, type(e).__name__
        return out
    pl = cg.measure_placement(spec, http)
    out["placement"] = {k: pl[k] for k in ("status", "size", "size_vram", "gpu_share", "context_length", "note")}
    client, pacer = make_client(spec, mode), cg.Pacer()
    for pid in prompt_ids:
        a = cg.generate_bit(spec, by_id[pid], client, pacer)
        out["bits"].append({"prompt": pid, "ok": a["ok"], "kind": a["kind"], "latency_s": a["latency_s"], "timeout": a["timeout"],
                            "error": None if a["ok"] else a["reason"].split(":")[0]})
    return out


def summarize(raw: dict) -> dict:
    lat = [b["latency_s"] for b in raw["bits"]]
    errors: dict[str, int] = {}
    for b in raw["bits"]:
        if not b["ok"]:
            errors[b["error"]] = errors.get(b["error"], 0) + 1
    return {**raw, "n": len(lat), "valid": sum(b["ok"] for b in raw["bits"]), "p50_s": cg.percentile(lat, 50),
            "max_s": max(lat, default=None), "timeouts": sum(b["timeout"] for b in raw["bits"]), "errors": errors,
            "fits_gpu": bool(raw["placement"] and raw["placement"]["status"] == "ok" and raw["placement"]["gpu_share"] >= 1.0)}


def run(tags: list[str] = TAGS, json_mode: str = "json_schema", prompt_ids: list[str] = PROMPT_IDS, prompt_digest: str = "") -> dict:
    """One arm: every tag, five bits each, unloaded afterwards. Numbers only."""
    assert prompt_digest == globals()["prompt_digest"](), "the prompt, timing or request changed since this call was built"
    out = {}
    for tag in tags:
        spec = cg.Spec(tag, cg.OLLAMA_PROVIDER, tag, 0.0, 0.0, "ollama", 0.0)
        try:
            out[tag] = summarize(screen_one(tag, json_mode, prompt_ids))
        finally:
            unload(spec)
    return {"json_mode": json_mode, "models": out}


def params(mode: str) -> dict:
    return {"tags": TAGS, "json_mode": mode, "prompt_ids": PROMPT_IDS, "prompt_digest": prompt_digest()}


def stored(mode: str) -> dict | None:
    return cache.load(f"{NAME}_{mode}", params(mode))


def table(res: dict) -> str:
    rows = ["| Model | GPU | Valid | p50 | max | Errors |", "|---|---|---|---|---|---|"]
    for tag, m in res["models"].items():
        gpu = "-" if not m["placement"] else (f"{m['placement']['gpu_share']:.0%}" if m["placement"]["gpu_share"] is not None else "unknown")
        rows.append(f"| {tag} | {gpu} | {m.get('valid', 0)}/{m.get('n', 0)} | {cg.fmt(m.get('p50_s'), '{:.2f} s')} | "
                    f"{cg.fmt(m.get('max_s'), '{:.2f} s')} | {', '.join(f'{k} x{v}' for k, v in m.get('errors', {}).items()) or '-'} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    for tag in TAGS:
        unload(cg.Spec(tag, cg.OLLAMA_PROVIDER, tag, 0.0, 0.0, "ollama", 0.0))  # start with nothing loaded
    for mode in MODES:
        res = cache.run_or_load(f"{NAME}_{mode}", run, rerun=a.rerun, **params(mode))
        print(f"\n{mode}\n{table(res)}", flush=True)


if __name__ == "__main__":
    main()
