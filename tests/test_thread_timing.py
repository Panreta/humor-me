"""Exp 26 (D-110): the bookkeeping of the per-thread timing check, with a fake Ollama and no model."""
import json
import pathlib
import sys
import threading
import time

import httpx
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import generate, llm  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim import thread_timing as tt  # noqa: E402

GOOD = json.dumps({"premise": "A robot at a bar", "jokes": [f"line {j}" for j in range(5)]})


def writer(handler):
    return llm.Client("http://local/v1", "", "bonsai-2:4b", 10.0, retries=0, deadline_s=12.0, local=True,
                      extra_body=llm.OLLAMA_NO_THINK, json_schema=True, transport=httpx.MockTransport(handler))


def reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})


def ps(share=1.0, context=8192):
    return lambda url: {"models": [{"name": "bonsai-2:4b", "model": "bonsai-2:4b", "size": 100,
                                    "size_vram": int(100 * share), "context_length": context}]}


def test_the_bits_are_disjoint_from_exp_24s_and_seeded():
    mine, theirs = tt.prompts(), cg.prompts(65)
    assert len(mine) == 20 and tt.prompts() == mine
    assert not {p["id"] for p in mine} & {p["id"] for p in theirs}
    assert tt.SEED != cg.SEED and tt.prompts(5, "other") != mine[:5]
    assert {p["mode"] for p in mine} == set(generate.MODES)


def test_it_runs_the_shows_writer_client_and_the_v4_generator(monkeypatch):
    for name in ("JOKE_WRITER_PROVIDER", "JOKE_WRITER_MODEL", "OLLAMA_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JOKE_WRITER_PROVIDER", "ollama")
    c = generate.writer_client()
    fields = ("base_url", "model", "timeout", "retries", "deadline_s", "extra_body", "json_schema")
    assert {f: getattr(c, f) for f in fields} == {f: getattr(cg.make_client(cg.MODELS["bonsai-2-4b"]), f) for f in fields}
    assert generate.PROMPT_ID == "gen-v4"


def test_a_good_run_passes_the_bar_and_keeps_per_call_numbers_but_no_text():
    c = writer(lambda request: reply(GOOD))
    s = tt.summarize(tt.measure(c, tt.prompts(4), ps()))
    assert s["n"] == 4 and s["bits_losing_a_thread"] == 0 and s["bits_with_zero_threads"] == 0
    assert all(b["threads"] == 3 and len(b["calls"]) == 3 for b in s["bits"])
    assert s["bar"] == {"max_lost_bits": 1, "lost_ok": True, "zero_ok": True, "gpu_ok": True}
    assert s["bits_same_premise"] == 4  # the fake gives every call the same premise: case-insensitive match
    assert "bar" not in json.dumps(s["bits"]) and "line 0" not in json.dumps(s) and "robot" not in json.dumps(s)
    assert s["warmup_s"] is not None and s["placement_after_run"]["gpu_share"] == 1.0


def test_one_failed_call_loses_a_thread_two_lost_bits_break_the_bar_and_a_dead_bit_has_no_threads():
    n = {"i": 0}
    lock = threading.Lock()

    def flaky(request):
        with lock:
            n["i"] += 1
            i = n["i"]
        return reply("not json") if i > 1 and i % 3 == 1 else reply(GOOD)  # warm-up is call 1; then every 3rd call fails
    s = tt.summarize(tt.measure(writer(flaky), tt.prompts(6), ps()))
    assert s["bits_losing_a_thread"] >= 2 and s["bar"]["lost_ok"] is False
    assert all(b["threads"] >= 1 for b in s["bits"]) and s["bits_with_zero_threads"] == 0
    assert any("GenerationError" in r for r in s["failure_reasons"])
    dead = tt.summarize(tt.measure(writer(lambda request: reply("not json")), tt.prompts(2), ps()))
    assert dead["bits_with_zero_threads"] == 2 and dead["bar"]["zero_ok"] is False


def test_losing_exactly_one_bit_is_within_the_bar():
    calls = {"n": 0}
    lock = threading.Lock()

    def one_bad(request):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        return reply("not json") if n == 3 else reply(GOOD)
    s = tt.summarize(tt.measure(writer(one_bad), tt.prompts(5), ps()))
    assert s["bits_losing_a_thread"] == 1 and s["bar"]["lost_ok"] is True


@pytest.mark.parametrize("share, context", [(0.9, 8192), (1.0, 4096)])
def test_a_writer_pushed_off_the_gpu_by_the_parallel_calls_fails_the_bar(share, context):
    reads = {"n": 0}

    def http(url):  # warm-up reads once, the run reads before and after: only the last reading is bad
        reads["n"] += 1
        return ps()(url) if reads["n"] <= 2 else ps(share, context)(url)
    s = tt.summarize(tt.measure(writer(lambda request: reply(GOOD)), tt.prompts(1), http))
    assert s["writer_on_gpu"] is False and s["bar"]["gpu_ok"] is False
    assert s["placement_after_warmup"]["gpu_share"] == 1.0 and s["placement_after_run"]["context_length"] == context


def test_overlapping_calls_are_counted_as_concurrent_and_serial_ones_are_not():
    def slow(request):
        time.sleep(0.15)
        return reply(GOOD)
    s = tt.summarize(tt.measure(writer(slow), tt.prompts(2), ps()))
    assert s["bits_concurrent"] == 2 and all(b["wall_s"] < 0.4 for b in s["bits"])
    raw = tt.measure(writer(lambda request: reply(GOOD)), tt.prompts(1), ps())
    raw["bits"][0]["wall_s"], raw["bits"][0]["calls"] = 0.9, [{"start_s": 0.3 * i, "seconds": 0.3, "ok": True, "reason": None} for i in range(3)]
    assert tt.summarize(raw)["bits_concurrent"] == 0  # wall time equals the calls' sum: they ran one after another


def test_percentiles_are_nearest_rank_and_a_failed_call_counts_as_infinite_latency():
    raw = {"bits": [{"prompt": f"t{i}", "wall_s": float(i + 1), "threads": 3, "failure": None, "same_premise": False,
                     "calls": [{"start_s": 0.0, "seconds": 1.0, "ok": True, "reason": None}] * 3} for i in range(20)],
           "placement_after_warmup": {"gpu_share": 1.0, "context_length": 8192, "others_loaded": []},
           "placement_after_run": {"gpu_share": 1.0, "context_length": 8192, "others_loaded": []}}
    s = tt.summarize(raw)
    assert (s["bit_p50_s"], s["bit_p95_s"]) == (10.0, 19.0) and s["call_p95_s"] == 1.0
    for b in raw["bits"][:2]:  # 6 of 60 calls failed: more than 5%, so the p95 is infinite
        b["calls"] = [{"start_s": 0.0, "seconds": 9.0, "ok": False, "reason": "LLMError: x"}] * 3
    assert tt.summarize(raw)["call_p95_s"] == float("inf")


def test_the_digest_follows_the_prompt_and_a_stale_call_is_refused(monkeypatch):
    before = tt.digest()
    monkeypatch.setattr(generate, "SYSTEM", generate.SYSTEM + " x")
    assert tt.digest() != before
    with pytest.raises(AssertionError):
        tt.run(digest_=before)
    assert tt.params()["n_bits"] == 20


def test_a_hosted_writer_is_refused(monkeypatch):
    monkeypatch.setenv("JOKE_WRITER_PROVIDER", "gemini")
    monkeypatch.setenv("JOKE_WRITER_MODEL", "m")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://g/v1")
    with pytest.raises(SystemExit):
        tt.run(digest_=tt.digest())
