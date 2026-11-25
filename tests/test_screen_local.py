"""Exp 24 local pre-screen (D-101): the bookkeeping, with no Ollama."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from sim import compare_generators as cg  # noqa: E402
from sim import screen_local as sl  # noqa: E402


def bit(ok=True, latency=5.0, timeout=False, error=None):
    return {"prompt": "p60", "ok": ok, "kind": None if ok else "model", "latency_s": latency, "timeout": timeout, "error": error}


def raw(bits, share=1.0, status="ok"):
    return {"warmup_ok": True, "warmup_s": 3.0, "bits": bits,
            "placement": {"status": status, "size": 1, "size_vram": 1, "gpu_share": share, "context_length": 8192, "note": ""}}


def test_the_screened_prompts_are_outside_the_runs_sixty_and_are_the_same_bits_the_run_would_draw():
    ids = [p["id"] for p in cg.prompts(cg.N_PROMPTS)]
    assert not set(sl.PROMPT_IDS) & set(ids) and len(sl.PROMPT_IDS) == 5
    assert [p["id"] for p in cg.prompts(65)[60:]] == sl.PROMPT_IDS  # prefix-stable: the same list the harness draws from


def test_summary_counts_valid_bits_percentiles_timeouts_and_error_classes_over_all_attempts():
    s = sl.summarize(raw([bit(), bit(latency=6.0), bit(ok=False, latency=12.03, timeout=True, error="LLMError"),
                          bit(ok=False, latency=1.9, error="GenerationError"), bit(ok=False, latency=2.0, error="GenerationError")]))
    assert (s["n"], s["valid"], s["timeouts"]) == (5, 2, 1) and s["max_s"] == 12.03 and s["p50_s"] == 5.0
    assert s["errors"] == {"LLMError": 1, "GenerationError": 2} and s["fits_gpu"] is True


def test_a_model_fits_only_when_the_gate_reading_is_ok_and_complete():
    assert sl.summarize(raw([bit()], share=0.88))["fits_gpu"] is False
    assert sl.summarize(raw([bit()], share=None, status="unknown") | {"placement": {"status": "unknown", "gpu_share": None}})["fits_gpu"] is False


def test_the_digest_follows_the_prompt_and_a_stale_call_is_refused(monkeypatch):
    before = sl.prompt_digest()
    monkeypatch.setattr(sl.generate, "SYSTEM", sl.generate.SYSTEM + " x")
    assert sl.prompt_digest() != before
    with pytest.raises(AssertionError):
        sl.run(tags=[], prompt_digest=before)


def test_stored_numbers_hold_no_text_only_error_class_names():
    s = sl.summarize(raw([bit(ok=False, error="GenerationError")]))
    assert "thread" not in json.dumps(s) and set(s["bits"][0]) == {"prompt", "ok", "kind", "latency_s", "timeout", "error"}
