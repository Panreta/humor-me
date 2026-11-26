"""Exp 24 harness with fake providers: pairing, resume and appended attempts, pacing, the blind ballot, scoring and the
bar. No network."""
import json
import math
import pathlib
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import llm, safety  # noqa: E402
from sim import compare_generators as cg  # noqa: E402

MODELS = ["qwen3-4b-instruct", "bonsai-2-4b", "claude-haiku-5-5"]
GEMMA = "ref-local"  # a stand-in reference (D-97): on the ballot, never picked; none ships since D-104
SHIPPED = dict(cg.MODELS)  # before the fixture below adds the stand-in


REF_SPEC = cg.Spec(GEMMA, cg.OLLAMA_PROVIDER, "ref-local:1b", 0.0, 0.0, "ollama", 0.0, reference=True)


@pytest.fixture(autouse=True)
def stand_in_reference(monkeypatch):
    monkeypatch.setitem(cg.MODELS, GEMMA, REF_SPEC)


TAG = {"qwen3-4b-instruct": "alpha", "bonsai-2-4b": "bravo", "claude-haiku-5-5": "charlie", GEMMA: "delta"}  # text must not name a model
BY_ID = {{**cg.MODELS, GEMMA: REF_SPEC}[k].model: t for k, t in TAG.items()}


def reply(tag):
    return json.dumps({"premises": ["a", "b", "c"], "threads": [[f"{tag} thread {i} line {j}." for j in range(5)] for i in range(3)]})


class FakeModel:
    """A provider: answers with a canned bit, or raises what `fail` says (an exception, or a list used one per call);
    counts attempts and tokens like the real clients."""

    def __init__(self, model, fail=None, tokens=(1000, 2000), vary=False):
        self.model, self.fail, self.tokens, self.vary = model, fail, tokens, vary
        self.attempts = self.input_tokens = self.output_tokens = 0
        self.calls = 0
        self.events = None  # a shared list: the order in which the harness touches models, warm-ups and /api/ps

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        if self.events is not None:
            self.events.append(("call", self.model))
        self.attempts += 1
        self.calls += 1
        fail = self.fail.pop(0) if isinstance(self.fail, list) and self.fail else (None if isinstance(self.fail, list) else self.fail)
        if fail:
            raise fail
        self.input_tokens += self.tokens[0]
        self.output_tokens += self.tokens[1]
        return reply(BY_ID[self.model] + (f"v{self.calls}" if self.vary else ""))


class FakeFilter:
    """The safety filter's model: approves every line, or errors like a failed provider call."""

    def __init__(self, error=None, reject=()):
        self.error, self.reject, self.calls = error, reject, 0

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        self.calls += 1
        if self.error:
            raise self.error
        items = json.loads(user)
        return json.dumps({"verdicts": [{"id": i["id"], "ok": not any(r in i["text"] for r in self.reject), "reason": "r"}
                                        for i in items]})


class FakeWarm:
    """The warm-up client: answers anything, or raises `fail`, and never touches the model's own call counters."""

    def __init__(self, model, events=None, fail=None):
        self.model, self.events, self.fail = model, events, fail

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        if self.events is not None:
            self.events.append(("warm", self.model))
        fail = self.fail.pop(0) if isinstance(self.fail, list) and self.fail else (None if isinstance(self.fail, list) else self.fail)
        if fail:
            raise fail
        return '{"ready": true}'


class NoWait(cg.Pacer):
    def __init__(self):
        super().__init__(sleep=lambda s: None)


@pytest.fixture(autouse=True)
def fresh_safety():
    safety.clear_cache()


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "DATA", tmp_path)
    return tmp_path


class FakeOllama:
    """Ollama's /api/ps and /api/tags: every tag in `share` is loaded with that share of a 6 GB file on the GPU.
    `ps` replaces the /api/ps body (empty, implausible); `raises` makes the server unreachable."""

    FILE = 6_000_000_000

    def __init__(self, share=None, ps=None, raises=None, context=cg.GATE_CONTEXT, tags=None):
        self.share = {cg.MODELS[m].tag(): 1.0 for m in MODELS[:2]} if share is None else share
        self.ps, self.raises, self.context, self.urls = ps, raises, context, []
        self.events = None
        self.tags = list(self.share) if tags is None else tags

    def __call__(self, url):
        self.urls.append(url)
        if self.events is not None and url.endswith("/api/ps"):
            self.events.append(("ps", url))
        if self.raises:
            raise self.raises
        if url.endswith("/api/tags"):
            return {"models": [{"name": t, "size": self.FILE} for t in self.tags]}
        assert url.endswith("/api/ps") and "/v1" not in url  # the native API, not the OpenAI-compatible one
        if self.ps is not None:
            return self.ps
        return {"models": [{"name": t, "size": self.FILE, "size_vram": int(self.FILE * sh), "context_length": self.context}
                           for t, sh in self.share.items()]}


def run(n=6, models=MODELS, clients=None, filt=None, events=None, warm_fail=None, log_lines=None, **kw):
    clients = {} if clients is None else clients
    http = kw.pop("http", None) or FakeOllama()
    http.events = events

    def client(spec):
        c = clients.setdefault(spec.key, FakeModel(spec.model))
        c.events = events
        return c

    done = cg.run_comparison(n, models, client, filt or FakeFilter(), pacer=NoWait(), log=(log_lines.append if log_lines is not None else (lambda s: None)),
                             http=http, make_warm_client=lambda spec: FakeWarm(spec.model, events, warm_fail), **kw)
    return done, clients


def rec_of(m, pid):
    return cg.read_record(cg.run_dir() / m / f"{pid}.json")


def test_prompts_are_seeded_prefix_stable_balanced_and_distinct():
    p60 = cg.prompts(cg.N_PROMPTS)
    p30 = cg.prompts(cg.N_BALLOT)
    assert (cg.N_PROMPTS, cg.N_BALLOT) == (60, 30)
    assert p60 == cg.prompts(60) and cg.prompts(72)[:60] == p60 and p60[:30] == p30  # the ballot is a prefix
    for ps, per_mode in ((p30, 5), (p60, 10)):  # both the voted set and the whole set are balanced by mode
        assert {m: sum(p["mode"] == m for p in ps) for m in cg.MODE_ORDER} == {m: per_mode for m in cg.MODE_ORDER}
    assert len({(p["mode"], *p["context"].values()) for p in p60}) == 60
    assert all(p["context"]["venue"] in cg.VENUES and p["context"]["age"] in cg.AGES for p in p60)


@pytest.mark.parametrize("attr, value", [("SYSTEM", "x"), ("FILTER_TIMING", (1.0, 2.0)), ("FILTER_RETRIES", 9),
                                         ("GENERATOR_TIMING", (1.0, 2.0))])
def test_digest_follows_the_generator_and_filter_settings(monkeypatch, attr, value):
    before = cg.gen_digest()
    monkeypatch.setattr(cg.generate, attr, value if attr != "SYSTEM" else cg.generate.SYSTEM + " changed")
    assert cg.gen_digest() != before


def test_digest_follows_the_filter_prompt_and_model(monkeypatch):
    before = cg.gen_digest()
    monkeypatch.setattr(cg.safety, "SYSTEM", cg.safety.SYSTEM + " changed")
    assert cg.gen_digest() != before
    monkeypatch.undo()
    for name, other in (("SAFETY_MODEL", "other"), ("SAFETY_PROVIDER", "gemini"), ("SAFETY_THINKING", True)):
        monkeypatch.setattr(cg, name, other)
        assert cg.gen_digest() != before, name  # no folder mixes two filters
        monkeypatch.undo()


def test_the_filter_client_is_haiku_without_thinking_whatever_the_env_says(monkeypatch):
    from app import llm_anthropic
    for name in ("SAFETY_CHECK_PROVIDER", "SAFETY_CHECK_MODEL", "SAFETY_CHECK_THINKING", "JOKE_WRITER_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SAFETY_CHECK_PROVIDER", "gemini")  # a .env pointing elsewhere must not change the run's filter
    monkeypatch.setenv("SAFETY_CHECK_THINKING", "on")
    c = cg.make_safety_client()
    assert isinstance(c, llm_anthropic.Client) and c.model == "claude-haiku-5-5" and c.thinking is False
    assert (c.timeout, c.deadline_s, c.retries) == (*cg.generate.FILTER_TIMING, cg.generate.FILTER_RETRIES)


def test_the_candidates_are_haiku_and_two_local_ollama_models_and_the_digest_follows_them(monkeypatch):
    assert list(SHIPPED) == MODELS
    qwen, bonsai, haiku = (cg.MODELS[m] for m in MODELS)
    assert not (qwen.reference or bonsai.reference or haiku.reference)  # the gate decides about qwen, not a flag
    assert (qwen.tag(), bonsai.tag(), haiku.provider) == ("qwen3:4b-instruct", "bonsai-2:4b", "anthropic")
    assert qwen.local and bonsai.local and not haiku.local and (qwen.price_in, qwen.price_out) == (0.0, 0.0)
    assert qwen.base_url() == cg.OLLAMA_URL and haiku.base_url() == ""
    before = cg.gen_digest()
    monkeypatch.setenv("QWEN_OLLAMA_TAG", "qwen3:4b-instruct-q8")  # the tag `ollama list` shows may differ
    assert qwen.tag() == "qwen3:4b-instruct-q8" and cg.gen_digest() != before
    monkeypatch.delenv("QWEN_OLLAMA_TAG")
    monkeypatch.setenv("OLLAMA_URL", "http://other:11434/v1")
    assert cg.gen_digest() != before
    monkeypatch.delenv("OLLAMA_URL")  # not undo(): that would also drop the stand-in reference
    assert cg.gen_digest() == before


def test_a_local_client_has_no_key_the_production_timing_and_asks_for_no_thinking(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://box:11434/v1")
    c = cg.make_client(cg.MODELS["bonsai-2-4b"])
    assert isinstance(c, llm.Client) and (c.base_url, c.api_key, c.model) == ("http://box:11434/v1", "", "bonsai-2:4b")
    assert (c.timeout, c.deadline_s, c.retries) == (*cg.generate.GENERATOR_TIMING, cg.generate.RETRIES)
    assert c.extra_body == llm.OLLAMA_NO_THINK
    assert c.json_schema is True and cg.make_warmup_client(cg.MODELS["bonsai-2-4b"]).json_schema is True  # D-101


def test_the_json_mode_is_part_of_the_digest_so_no_folder_mixes_modes(monkeypatch):
    before = cg.gen_digest()
    monkeypatch.setattr(cg, "LOCAL_JSON_MODE", "json_object")
    assert cg.gen_digest() != before
    assert cg.make_client(cg.MODELS["bonsai-2-4b"]).json_schema is False  # the constant drives the client too


def test_the_shipped_candidates_are_haiku_bonsai_and_qwen_with_no_reference():
    assert [m for m, s in SHIPPED.items() if not s.reference] == ["qwen3-4b-instruct", "bonsai-2-4b", "claude-haiku-5-5"]
    assert not [m for m, s in SHIPPED.items() if s.reference]  # D-104 dropped gemma4:12b
    assert "mimo-v2.6-9b" not in SHIPPED and "gemma4-12b" not in SHIPPED
    assert cg.MODELS["qwen3-4b-instruct"].tag() == "qwen3:4b-instruct" and cg.MODELS["qwen3-4b-instruct"].price_in == 0.0


def test_nothing_listening_on_a_models_first_call_is_a_config_stop_with_the_ollama_hint(data):
    down = llm.LLMError("transport: ConnectError", retryable=True)
    with pytest.raises(cg.ConfigError, match=r"ollama serve.*ollama pull qwen3:4b-instruct"):
        run(n=2, models=["qwen3-4b-instruct"], clients={"qwen3-4b-instruct": FakeModel("qwen3:4b-instruct", fail=down)})
    assert not (cg.run_dir() / "qwen3-4b-instruct").exists()


def test_a_connection_failure_after_the_first_call_is_infra_data_and_never_stops_a_hosted_model(data):
    down = llm.LLMError("transport: ConnectError", retryable=True)
    run(n=2, models=["qwen3-4b-instruct"], clients={"qwen3-4b-instruct": FakeModel("qwen3:4b-instruct", fail=[None, down])})
    assert rec_of("qwen3-4b-instruct", "p01")["attempts"][0]["kind"] == "infra"
    with pytest.raises(cg.ConfigError) as e:  # a hosted model keeps the old rule: a transport error is an outage there
        run(n=1, models=["claude-haiku-5-5"], clients={"claude-haiku-5-5": FakeModel("claude-haiku-5-5", fail=ValueError("x"))})
    assert "ollama" not in str(e.value)
    run(n=1, models=["claude-haiku-5-5"], clients={"claude-haiku-5-5": FakeModel("claude-haiku-5-5", fail=down)})
    assert rec_of("claude-haiku-5-5", "p00")["attempts"][0]["kind"] == "infra"


def test_every_model_gets_the_same_prompts_and_one_attempt_per_prompt(data):
    done, _ = run(n=4)
    assert done == {"generated": 12, "judged": 12, "cached": 0}
    for m in MODELS:
        recs = [rec_of(m, p["id"]) for p in cg.prompts(4)]
        assert [r["prompt"] for r in recs] == ["p00", "p01", "p02", "p03"]
        for r in recs:
            (a,) = r["attempts"]
            assert a["ok"] and len(a["threads"]) == 3 and all(len(t) == 5 for t in a["threads"])
            assert (a["input_tokens"], a["output_tokens"], a["requests"], a["kind"]) == (1000, 2000, 1, None)


def test_resume_calls_nothing_again_and_extending_adds_only_new_prompts(data):
    run(n=3)
    clients = {}
    done, _ = run(n=5, clients=clients)
    assert done["cached"] == 9 and done["generated"] == 6
    assert all(c.calls == 2 for c in clients.values())


def test_failure_kinds():
    E = llm.LLMError
    assert cg.failure_kind(E("http 400", 400)) == "config"
    assert cg.failure_kind(E("http 401", 401)) == "config"
    assert cg.failure_kind(E("http 404", 404)) == "config"
    assert cg.failure_kind(E("no anthropic credentials")) == "config"
    assert cg.failure_kind(ValueError("bug")) == "config"
    assert cg.failure_kind(E("http 408", 408, retryable=True)) == "infra"
    assert cg.failure_kind(E("http 429", 429)) == "infra"
    assert cg.failure_kind(E("http 503", 503)) == "infra"
    assert cg.failure_kind(E("transport: ReadTimeout", retryable=True)) == "infra"
    assert cg.failure_kind(E("refused by the model")) == "model"
    assert cg.failure_kind(E("reply cut off at max_tokens")) == "model"
    assert cg.failure_kind(cg.generate.GenerationError("malformed reply")) == "model"


@pytest.mark.parametrize("err", [llm.LLMError("http 400", 400), llm.LLMError("http 404", 404), ValueError("x"),
                                 llm.LLMError("http 401", 401)])
def test_a_config_failure_on_a_models_first_call_stops_the_run_and_caches_nothing(data, err):
    with pytest.raises(cg.ConfigError):
        run(n=2, models=["bonsai-2-4b"], clients={"bonsai-2-4b": FakeModel("bonsai-2:4b", fail=err)})
    assert not (cg.run_dir() / "bonsai-2-4b").exists()


def test_a_config_failure_after_the_first_call_is_data_not_a_stop(data):
    flaky = {"claude-haiku-5-5": FakeModel("claude-haiku-5-5", fail=[None, llm.LLMError("http 400", 400)])}
    run(n=2, models=["claude-haiku-5-5"], clients=flaky)
    assert rec_of("claude-haiku-5-5", "p00")["attempts"][0]["ok"]
    a = rec_of("claude-haiku-5-5", "p01")["attempts"][0]
    assert not a["ok"] and a["kind"] == "config"


def test_redo_failed_appends_attempts_and_only_for_infra_and_config_failures(data):
    E = llm.LLMError
    bad = {"bonsai-2-4b": FakeModel("bonsai-2:4b", fail=[None, E("http 503", 503), E("refused by the model")])}
    run(n=3, models=["bonsai-2-4b"], clients=bad)
    assert [a["kind"] for a in rec_of("bonsai-2-4b", "p01")["attempts"]] == ["infra"]
    assert [a["kind"] for a in rec_of("bonsai-2-4b", "p02")["attempts"]] == ["model"]
    done, _ = run(n=3, models=["bonsai-2-4b"], clients=bad)  # without the flag nothing is repeated
    assert done["generated"] == 0
    done, _ = run(n=3, models=["bonsai-2-4b"], clients={"bonsai-2-4b": FakeModel("bonsai-2:4b")}, redo_failed=True)
    assert done["generated"] == 1  # the refusal is the model's answer and stands
    infra = rec_of("bonsai-2-4b", "p01")["attempts"]
    assert [a["ok"] for a in infra] == [False, True] and infra[0]["reason"].endswith("http 503")  # appended, not replaced
    assert len(rec_of("bonsai-2-4b", "p02")["attempts"]) == 1
    assert cg.final(rec_of("bonsai-2-4b", "p01")) == infra[1]


def test_a_filter_outage_is_retried_on_resume_not_scored_as_rejections(data):
    run(n=1, models=["qwen3-4b-instruct"], filt=FakeFilter(error=llm.LLMError("http 503", 503)))
    rec = rec_of("qwen3-4b-instruct", "p00")
    assert cg.final(rec)["safety_infra_error"] and cg.playable_thread(rec) is None
    safety.clear_cache()
    done, _ = run(n=1, models=["qwen3-4b-instruct"], filt=FakeFilter())
    att = cg.final(rec_of("qwen3-4b-instruct", "p00"))
    assert done["generated"] == 0 and done["judged"] == 1 and not att["safety_infra_error"] and att["safety_tries"] == 2


def test_verdicts_from_older_rules_are_redone_without_regenerating(data, monkeypatch):
    # D-105: a rules fix must reach bits already judged, and only the safety step may run again.
    run(n=2, models=["qwen3-4b-instruct"], filt=FakeFilter())
    att = cg.final(rec_of("qwen3-4b-instruct", "p00"))
    assert att["safety_rules"] == safety.RULES_VERSION and not cg.needs_safety(att)
    done, _ = run(n=2, models=["qwen3-4b-instruct"], filt=FakeFilter())
    assert done == {"generated": 0, "judged": 0, "cached": 2}  # same rules: nothing to redo
    monkeypatch.setattr(safety, "RULES_VERSION", "rules-next")
    safety.clear_cache()
    threads_before = att["threads"]
    done, _ = run(n=2, models=["qwen3-4b-instruct"], filt=FakeFilter())
    att = cg.final(rec_of("qwen3-4b-instruct", "p00"))
    assert done == {"generated": 0, "judged": 2, "cached": 2} and att["threads"] == threads_before
    assert att["safety_rules"] == "rules-next" and att["safety_tries"] == 1  # the retry budget restarts with new rules
    assert len(rec_of("qwen3-4b-instruct", "p00")["attempts"]) == 1  # no new attempt: generation is untouched


def test_a_bit_judged_before_rules_were_versioned_is_redone():
    old = {"ok": True, "safety": [[[True, "llm:ok"]]], "safety_infra_error": False, "safety_tries": 1}
    assert cg.needs_safety(old) and not cg.needs_safety({**old, "safety_rules": safety.RULES_VERSION})
    assert not cg.needs_safety({**old, "ok": False})  # a failed generation has nothing to judge


def test_latency_is_wall_time_around_the_call_and_excludes_pacing():
    ticks = iter([10.0, 14.5])
    sleeps = []
    pacer = cg.Pacer(sleep=sleeps.append, clock=lambda: 0.0)
    spec = cg.MODELS["claude-haiku-5-5"]  # the local models need no spacing: gap 0
    pacer.mark(spec.bucket)
    a = cg.generate_bit(spec, cg.prompts(1)[0], FakeModel("claude-haiku-5-5"), pacer, now=lambda: next(ticks))
    assert a["latency_s"] == 4.5 and sleeps == [spec.gap_s] and not a["timeout"]


def test_a_timeout_or_a_failure_that_used_the_deadline_up_is_marked_as_a_timeout():
    spec, pacer = cg.MODELS["bonsai-2-4b"], NoWait()
    slow = FakeModel("bonsai-2:4b", fail=llm.LLMError("transport: ReadTimeout", retryable=True))
    assert cg.generate_bit(spec, cg.prompts(1)[0], slow, pacer, now=iter([0.0, 3.0]).__next__)["timeout"]
    spent = FakeModel("bonsai-2:4b", fail=llm.LLMError("http 429", 429))
    assert cg.generate_bit(spec, cg.prompts(1)[0], spent, pacer, now=iter([0.0, 11.9]).__next__)["timeout"]
    quick = FakeModel("bonsai-2:4b", fail=llm.LLMError("http 429", 429))
    assert not cg.generate_bit(spec, cg.prompts(1)[0], quick, pacer, now=iter([0.0, 2.0]).__next__)["timeout"]


def test_pacer_spaces_one_bucket_and_leaves_others_alone():
    clock, sleeps = [100.0], []
    p = cg.Pacer(sleep=sleeps.append, clock=lambda: clock[0])
    p.wait("a", 7.0)
    assert sleeps == []
    p.mark("a")
    clock[0] += 2.0
    p.wait("a", 7.0)
    p.wait("b", 7.0)
    assert sleeps == [5.0]


def test_a_filter_rejection_shows_in_pass_rates_and_the_playable_thread(data):
    run(n=1, models=["qwen3-4b-instruct"], filt=FakeFilter(reject=("thread 0 line 3",)))
    rec = rec_of("qwen3-4b-instruct", "p00")
    assert cg.playable_thread(rec) == 1  # thread 0 lost a line, thread 1 is the first whole one
    s = cg.call_stats(rec["attempts"], cg.MODELS["qwen3-4b-instruct"])
    assert s["line_pass_rate"] == pytest.approx(14 / 15) and s["thread_pass_rate"] == pytest.approx(2 / 3)
    assert s["playable_rate"] == 1.0


def att(ok=True, latency=2.0, tin=1000, tout=2000, reason=None, timeout=False, kind=None):
    return {"ok": ok, "latency_s": latency, "input_tokens": tin, "output_tokens": tout, "requests": 1, "reason": reason,
            "timeout": timeout, "kind": kind, "safety": [[[True, "ok"]] * 5] * 3 if ok else None}


def test_call_stats_count_every_attempt_and_cost_covers_failed_ones():
    spec = cg.MODELS["claude-haiku-5-5"]
    s = cg.call_stats([att(latency=x) for x in (1, 2, 3, 4)]
                      + [att(ok=False, latency=2.0, reason="LLMError: http 529", tin=500, tout=0)], spec)
    assert (s["n_attempts"], s["n_failed"], s["failure_rate"]) == (5, 1, 0.2)
    assert s["latency_p50_s"] == 2.0 and s["latency_p95_s"] == 4.0  # nearest rank; a quick failure is just a call
    cost = (4500 * 0.10 + 8000 * 0.50) / 1e6
    assert s["cost_per_bit_usd"] == pytest.approx(cost / 5) and s["cost_5_shows_usd"] == pytest.approx(cost / 5 * 40)
    assert s["failure_reasons"] == {"LLMError: http 529": 1} and s["failure_ci95"] == cg.wilson(1, 5)


def test_latency_p95_is_over_all_calls_and_a_timeout_counts_as_over_the_deadline():
    spec = cg.MODELS["bonsai-2-4b"]
    fine = [att(latency=3.0) for _ in range(19)]
    one = cg.call_stats(fine + [att(ok=False, latency=12.0, timeout=True, reason="LLMError: transport: ReadTimeout")], spec)
    assert one["latency_p95_s"] == 3.0 and not one["p95_over_deadline"] and one["n_timeouts"] == 1  # 1 in 20 is not the p95
    two = cg.call_stats(fine[:18] + [att(ok=False, latency=12.0, timeout=True, reason="LLMError: x")] * 2, spec)
    assert two["latency_p95_s"] is None and two["p95_over_deadline"]
    assert not cg.eligibility({"m": dict(two, failure_ci95=[0, 0.01])})["m"]["latency_ok"]
    assert cg.percentile([1.0, math.inf], 50) == 1.0 and cg.percentile([1.0, 2.0, math.inf], 50) == 2.0
    assert cg.percentile(list(range(1, 31)), 95) == 29 and cg.percentile([], 95) is None


def stats(**over):
    base = {m: {"latency_p95_s": 5.0, "p95_over_deadline": False, "n_attempts": 30, "failure_rate": 0.0,
                "failure_ci95": cg.wilson(0, 30), "cost_per_bit_usd": c, "latency_p50_s": 2.0}
            for m, c in zip(MODELS, (0.0005, 0.002, 0.0006))}
    for m, patch in over.items():
        base[next(k for k in base if re.sub(r"\W", "_", k) == re.sub(r"\W", "_", m))].update(patch)  # kwargs cannot hold "-" or "."
    return base


def with_failures(k, n=60):
    return {"n_attempts": n, "failure_rate": k / n, "failure_ci95": cg.wilson(k, n)}


def test_failure_bar_is_the_wilson_upper_bound_over_every_attempt():
    f = lambda k, n: cg.eligibility(stats(bonsai_2_4b=with_failures(k, n)))["bonsai-2-4b"]["failure_ok"]  # noqa: E731
    assert f(0, 30)  # upper 11.4%
    assert not f(1, 30)  # upper 16.7%: one failure in 30 attempts is too many to rule out 15%
    assert f(1, 60) and f(2, 60) and f(3, 60)  # more attempts, a tighter bound (8.9%, 11.4%, 13.8%)
    assert not f(4, 60)


def test_bar_edges_are_inclusive_on_latency():
    _, b, c = MODELS
    assert cg.eligibility(stats(bonsai_2_4b={"latency_p95_s": cg.DEADLINE_S}), placement=PLACED)[b]["eligible"]
    e = cg.eligibility(stats(bonsai_2_4b={"latency_p95_s": 12.01}, claude_haiku_5_5={"latency_p95_s": None}), placement=PLACED)
    assert not e[b]["eligible"] and not e[c]["eligible"]


def prep(n=6, models=MODELS, **kw):
    run(n=n, models=models, **kw)
    return cg.export_ballot(models, n), cg.run_dir()


def test_ballot_is_blind_shuffled_per_prompt_and_keyed_apart(data):
    out, d = prep(n=12)
    ballot = (d / "ballot.md").read_text(encoding="utf-8")
    kp = cg.key_path(d)
    key = json.loads(kp.read_text(encoding="utf-8"))
    assert out == {"in_key": 12, "to_vote": 12, "ranked_last": {}, "pending": []}
    assert kp.parent != d and not (d / "key.json").exists() and kp.parent.name == "key"
    assert not any(name in ballot.lower() for name in ("gemini", "claude", "haiku", "qwen", "bonsai", "ollama"))
    assert len({tuple(k["letters"].values()) for k in key.values()}) > 1  # not one fixed order
    assert all(sorted(k["letters"].values()) == sorted(MODELS) and k["last"] == [] for k in key.values())
    for k in key.values():  # the text under each letter is that model's thread
        for m in k["letters"].values():
            assert f"{TAG[m]} thread {k['threads'][m]} line 0." in ballot


def test_a_model_with_no_safe_thread_is_ranked_last_and_the_prompt_stays(data):
    bad = {"bonsai-2-4b": FakeModel("bonsai-2:4b", fail=[None, llm.LLMError("refused by the model")])}
    out, d = prep(n=3, clients=bad)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert out["ranked_last"] == {"p01": ["bonsai-2-4b"]} and len(key) == 3  # nothing dropped
    assert sorted(key["p01"]["letters"].values()) == ["claude-haiku-5-5", "qwen3-4b-instruct"] and key["p01"]["last"] == ["bonsai-2-4b"]
    ballot = (d / "ballot.md").read_text(encoding="utf-8")
    assert "**C**" not in ballot.split("## p01")[1].split("## p02")[0]  # two options shown, two letters
    votes = cg.read_votes(d / "votes.csv")
    votes["p00"], votes["p01"], votes["p02"] = "ABC", "BA", "CBA"
    ranks = cg.ranks_from_votes(votes, key, MODELS)
    one = ranks["p01"]
    first, second = (key["p01"]["letters"][c] for c in "BA")
    assert (one[first], one[second], one["bonsai-2-4b"]) == (1.0, 2.0, 3.0)
    assert len(ranks) == 3


def test_one_shown_model_needs_no_vote_and_none_shown_ties_everyone(data):
    E = llm.LLMError
    clients = {"bonsai-2-4b": FakeModel("bonsai-2:4b", fail=[E("refused by the model")] * 2),
               "claude-haiku-5-5": FakeModel("claude-haiku-5-5", fail=[E("refused by the model"), None])}
    out, d = prep(n=2, clients=clients)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert list(key["p00"]["letters"].values()) == ["qwen3-4b-instruct"] and "p00" not in cg.read_votes(d / "votes.csv")
    ranks = cg.ranks_from_votes({"p01": "AB"}, key, MODELS)
    assert ranks["p00"] == {"qwen3-4b-instruct": 1.0, "bonsai-2-4b": 2.5, "claude-haiku-5-5": 2.5}
    assert ranks["p01"]["bonsai-2-4b"] == 3.0 and sorted(ranks["p01"].values()) == [1.0, 2.0, 3.0]


def test_no_model_with_a_safe_thread_ties_everyone_in_the_middle(data):
    E = llm.LLMError
    clients = {m: FakeModel(cg.MODELS[m].model, fail=[E("refused by the model")]) for m in MODELS}
    _, d = prep(n=1, clients=clients)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert cg.ranks_from_votes({}, key, MODELS)["p00"] == {m: 2.0 for m in MODELS}


def test_prompts_not_generated_for_every_model_wait_instead_of_being_ranked_last(data):
    run(n=2, models=["qwen3-4b-instruct"])
    out = cg.export_ballot(MODELS, 2)
    assert out["pending"] == ["p00", "p01"] and out["in_key"] == 0


def test_two_models_export_and_score_without_a_key_error(data):
    two = ["qwen3-4b-instruct", "claude-haiku-5-5"]
    out, d = prep(n=4, models=two)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert out["to_vote"] == 4 and all(sorted(k["letters"]) == ["A", "B"] for k in key.values())
    ballot = (d / "ballot.md").read_text(encoding="utf-8")
    assert "**C**" not in ballot
    votes = {p: "AB" for p in key}
    cg.write_votes(d / "votes.csv", votes)
    ranks = cg.ranks_from_votes(votes, key, two)
    assert set(ranks["p00"]) == set(two) and sorted(ranks["p00"].values()) == [1.0, 2.0]
    assert cg.rank_stats(ranks, two)["mean_rank"].keys() == set(two)


def test_scoring_a_subset_of_a_three_model_ballot_reranks_to_the_subset(data):
    _, d = prep(n=3)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    pref = {"claude-haiku-5-5": 0, "bonsai-2-4b": 1, "qwen3-4b-instruct": 2}
    votes = {pid: "".join(sorted(k["letters"], key=lambda c: pref[k["letters"][c]])) for pid, k in key.items()}
    subset = ["qwen3-4b-instruct", "claude-haiku-5-5"]
    ranks = cg.ranks_from_votes(votes, key, subset)
    assert all(r == {"claude-haiku-5-5": 1.0, "qwen3-4b-instruct": 2.0} for r in ranks.values())
    s = cg.rank_stats(cg.ranks_from_votes(votes, key, MODELS), subset)
    assert s["mean_rank"] == {"qwen3-4b-instruct": 2.0, "claude-haiku-5-5": 1.0}


def test_rerank_gives_equal_ranks_their_average_place():
    assert cg.rerank({"a": 1.0, "b": 2.5, "c": 2.5}, ["a", "b", "c"]) == {"a": 1.0, "b": 2.5, "c": 2.5}
    assert cg.rerank({"a": 3.0, "b": 1.0, "c": 2.0}, ["a", "c"]) == {"c": 1.0, "a": 2.0}


def test_export_is_incremental_and_never_reshuffles_or_loses_votes(data):
    _, d = prep(n=3)
    old = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    votes = cg.read_votes(d / "votes.csv")
    votes["p01"] = "CAB"
    cg.write_votes(d / "votes.csv", votes)
    run(n=5)
    cg.export_ballot(MODELS, 5)
    assert cg.read_votes(d / "votes.csv") == {"p00": "", "p01": "CAB", "p02": "", "p03": "", "p04": ""}
    new = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert all(new[p] == old[p] for p in old) and len(new) == 5


def test_parse_ranking():
    assert cg.parse_ranking("bac", "ABC") == "BAC" and cg.parse_ranking(" c, a b ", "ABC") == "CAB"
    assert cg.parse_ranking("aab", "ABC") is None and cg.parse_ranking("ab", "ABC") is None and cg.parse_ranking("") is None
    assert cg.parse_ranking("dbca") == "DBCA" and cg.parse_ranking("bac") is None  # the default is the four-letter ballot
    assert cg.parse_ranking("ba", "AB") == "BA" and cg.parse_ranking("abc", "AB") is None


def test_voting_saves_after_every_answer_and_can_be_resumed(data):
    _, d = prep(n=4)
    answers = iter(["bad", "cab", "s", "abc", "q"])
    shown = []
    assert cg.vote_interactively(read=lambda prompt: next(answers), out=shown.append) == 2
    assert cg.read_votes(d / "votes.csv") == {"p00": "CAB", "p01": "", "p02": "ABC", "p03": ""}
    assert any("use each of A, B, C once" in s for s in shown)
    assert cg.vote_interactively(read=lambda prompt: "q", out=lambda s: None) == 0


def ranks_for(order_by_prompt):
    """One list of models per prompt, best first."""
    return {f"p{i:02d}": {m: float(r) for r, m in enumerate(order, 1)} for i, order in enumerate(order_by_prompt)}


def test_rank_stats_mean_ci_and_paired_gap():
    a, b, c = MODELS
    s = cg.rank_stats(ranks_for([[a, b, c]] * 20), MODELS)
    assert s["mean_rank"] == {a: 1.0, b: 2.0, c: 3.0} and s["ci95"][a] == [1.0, 1.0]
    assert s["diff"][f"{b}|{a}"]["mean"] == 1.0 and s["n"] == 20


PLACED = {m: {"status": "ok", "gpu_share": 1.0} for m in MODELS}  # every local model fits: the bar's other terms are tested


def decide(s, r, **kw):
    return cg.decide(s, r, **{"placement": PLACED, **kw})


def test_clear_winner_among_eligible_is_picked_even_if_costlier():
    a, b, c = MODELS
    d = decide(stats(), ranks_for([[b, c, a]] * 24))
    assert d["pick"] == b and d["best"] == b and d["second"] == c and d["funniness_clear"] is True


def test_overlapping_ranks_pick_the_cheaper_and_say_inconclusive():
    rng = np.random.default_rng(1)
    d = decide(stats(), ranks_for([list(rng.permutation(MODELS)) for _ in range(30)]))
    assert d["funniness_clear"] is False and "inconclusive" in d["reason"]
    assert d["pick"] == min((d["best"], d["second"]), key=lambda m: stats()[m]["cost_per_bit_usd"])


def test_ineligible_models_cannot_win_and_ranks_are_redone_among_the_rest():
    a, b, c = MODELS
    s = stats(bonsai_2_4b={"latency_p95_s": 12.5})  # the funniest, but past the 12 s deadline
    d = decide(s, ranks_for([[b, a, c]] * 20 + [[b, c, a]] * 4))
    assert d["eligible"] == sorted([a, c]) and d["pick"] == a and d["funniness_clear"] is True
    assert d["eligibility"][b]["eligible"] is False and d["eligibility"][b]["reasons"] == ["latency over the deadline"]


def test_one_eligible_model_wins_and_none_eligible_picks_nothing():
    a = MODELS[0]
    only = decide(stats(bonsai_2_4b=with_failures(5), claude_haiku_5_5=with_failures(5)), {})
    assert only["pick"] == a and only["reason"] == "the only eligible model"
    none = decide(stats(**{m.replace("-", "_"): with_failures(10) for m in MODELS}), {})
    assert none["pick"] is None and "stays" in none["reason"]
    waiting = decide(stats(), {})
    assert waiting["pick"] is None and "votes" in waiting["reason"]


def test_deadline_is_the_generators_current_one():
    from app import generate
    assert cg.DEADLINE_S == generate.GENERATOR_TIMING[1] == 12.0


def test_power_numbers_behind_the_prompt_count():
    assert cg.halfwidth(30) == pytest.approx(0.292, abs=1e-3) and cg.gap_halfwidth(30) == pytest.approx(0.506, abs=1e-3)
    assert cg.simulate_decision_power(30, 1.5, reps=100, n_boot=500) > 0.9
    assert cg.simulate_decision_power(30, 0.0, reps=100, n_boot=500) < 0.15  # no gap: a clear call is rare, and a fluke


def test_run_scores_saved_attempts_and_votes_and_stores_numbers_only(data, monkeypatch):
    monkeypatch.setattr(cg.cache, "METRICS", data / "metrics")
    run(n=60)
    cg.export_ballot(MODELS, 30)
    d = cg.run_dir()
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    pref = {"claude-haiku-5-5": 0, "bonsai-2-4b": 1, "qwen3-4b-instruct": 2}  # this voter: claude, flash, flash-lite
    votes = {pid: "".join(sorted(k["letters"], key=lambda c: pref[k["letters"][c]])) for pid, k in key.items()}
    cg.write_votes(d / "votes.csv", votes)
    params = cg.current_params(60, tuple(MODELS), 30)
    res = cg.cache.run_or_load(cg.METRICS_NAME, cg.run, **params)
    assert (res["n_prompts"], res["n_ballot"], res["n_key"], res["n_scored"]) == (60, 30, 30, 30) and res["votes"]["mean_rank"]["claude-haiku-5-5"] == 1.0
    assert res["n_ranked_last"] == {m: 0 for m in MODELS}
    assert res["models"]["bonsai-2-4b"]["n_attempts"] == 60 and res["models"]["bonsai-2-4b"]["failure_ci95"][1] < 0.07
    assert res["decision"]["pick"] == "claude-haiku-5-5" and res["decision"]["funniness_clear"] is True
    stored_text = (data / "metrics" / "model_comparison.json").read_text(encoding="utf-8")
    assert "thread " not in stored_text and "line 0" not in stored_text  # no generated text
    assert cg.stored() == json.loads(json.dumps(res))
    assert cg.table(res).count("\n") == 4 and cg.verdict(res).startswith("Pick:")
    cg.write_votes(d / "votes.csv", {**votes, "p00": "ABC" if votes["p00"] != "ABC" else "CBA"})
    with pytest.raises(AssertionError):  # a changed vote invalidates the call
        cg.run(**params)


def test_sixty_prompts_are_generated_and_only_the_first_thirty_reach_the_ballot(data):
    done, _ = run(n=60)
    assert done["generated"] == 180
    out = cg.export_ballot(MODELS, 30)
    key = json.loads(cg.key_path(cg.run_dir()).read_text(encoding="utf-8"))
    assert out["in_key"] == 30 and sorted(key) == [p["id"] for p in cg.prompts(30)]
    assert len(cg.read_votes(cg.run_dir() / "votes.csv")) == 30
    assert "p30" not in (cg.run_dir() / "ballot.md").read_text(encoding="utf-8")


def test_failures_and_latency_are_counted_over_all_sixty_calls_not_just_the_voted_thirty(data):
    E = llm.LLMError
    # a model that fails only on prompts 31 to 60 (calls 31..60): the vote never sees it, the gate does
    bad = {"bonsai-2-4b": FakeModel("bonsai-2:4b", fail=[None] * 30 + [E("refused by the model")] * 4)}
    run(n=60, models=["bonsai-2-4b"], clients=bad)
    d = cg.run_dir()
    attempts = cg.load_attempts(d, ["bonsai-2-4b"], 60)["bonsai-2-4b"]
    s = cg.call_stats(attempts, cg.MODELS["bonsai-2-4b"])
    assert (s["n_attempts"], s["n_failed"]) == (60, 4) and s["failure_ci95"] == cg.wilson(4, 60)
    assert not cg.eligibility({"m": s})["m"]["failure_ok"]
    assert cg.load_attempts(d, ["bonsai-2-4b"], 30)["bonsai-2-4b"].__len__() == 30  # the voted set alone would not show it


def test_what_the_sixty_attempt_failure_bar_rules_out():
    from math import comb

    def ruled_out(p, n, k_ok):
        return 1 - sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k_ok + 1))

    assert ruled_out(0.03, 60, 3) == pytest.approx(0.106, abs=0.002) and ruled_out(0.03, 30, 0) == pytest.approx(0.599, abs=0.002)
    assert 1 - ruled_out(0.10, 60, 3) == pytest.approx(0.137, abs=0.002)
    for k, n in ((3, 60), (4, 60)):
        assert (cg.wilson(k, n)[1] <= cg.BAR_FAILURE_UPPER) == (k == 3)


def test_latency_p95_is_over_all_sixty_calls_nearest_rank():
    spec = cg.MODELS["bonsai-2-4b"]
    slow = att(ok=False, latency=12.0, timeout=True, reason="LLMError: transport: ReadTimeout")
    three = [att(latency=3.0) for _ in range(57)] + [slow] * 3
    four = [att(latency=3.0) for _ in range(56)] + [slow] * 4
    assert cg.call_stats(three, spec)["latency_p95_s"] == 3.0  # the 57th of 60 is still a finite call
    assert cg.call_stats(four, spec)["p95_over_deadline"] and cg.call_stats(four, spec)["latency_p95_s"] is None


# --- D-97: the GPU gate and the reference model ------------------------------------------------------------------

BONSAI, QWEN, HAIKU = "bonsai-2-4b", "qwen3-4b-instruct", "claude-haiku-5-5"
PS_URL = "http://127.0.0.1:11434/api/ps"


def placed(m, **over):
    return {"status": "ok", "gpu_share": 1.0, **over}


def measured(m, http):
    return cg.measure_placement(cg.MODELS[m], http)


def test_a_model_fully_on_the_gpu_passes_the_gate_and_a_partial_one_fails_with_its_share():
    tag = cg.MODELS[QWEN].tag()
    full = measured(QWEN, FakeOllama(share={tag: 1.0}))
    assert (full["status"], full["gpu_share"], full["context_length"], full["context_ok"]) == ("ok", 1.0, 8192, True)
    part = measured(QWEN, FakeOllama(share={tag: 0.87}))  # the measured case: 5.60 of 6.39 GB
    assert (part["status"], round(part["gpu_share"], 2)) == ("ok", 0.87)  # a clean reading that fails the gate
    el = cg.eligibility(stats(), placement={QWEN: part, BONSAI: placed(BONSAI)})
    assert el[QWEN]["eligible"] is False and el[QWEN]["reasons"] == ["partial GPU 87%"] and el[QWEN]["placement_ok"] is False
    assert el[BONSAI]["eligible"] and el[HAIKU]["eligible"]  # a hosted model has no placement to fail
    assert cg.placement_gate(cg.MODELS[QWEN], placed(QWEN, gpu_share=0.999)) == "partial GPU 100%"  # no tolerance


@pytest.mark.parametrize("why,http", [
    ("empty /api/ps right after a successful generate", FakeOllama(ps={"models": []})),
    ("no models key", FakeOllama(ps={})),
    ("1 GB reported for a 6 GB file", FakeOllama(ps={"models": [{"name": "qwen3:4b-instruct", "size": 1_010_000_000, "size_vram": 1_010_000_000}]})),
    ("size_vram above size", FakeOllama(ps={"models": [{"name": "qwen3:4b-instruct", "size": 6_000_000_000, "size_vram": 7_000_000_000}]})),
    ("no size at all", FakeOllama(ps={"models": [{"name": "qwen3:4b-instruct"}]})),
    ("model absent from /api/tags: cannot check the report", FakeOllama(share={"qwen3:4b-instruct": 1.0}, tags=[])),
    ("server unreachable", FakeOllama(raises=ConnectionError("down"))),
    ("loaded at another context", FakeOllama(share={"qwen3:4b-instruct": 1.0}, context=32768)),
    ("partial, with another model loaded: not attributable", FakeOllama(share={"qwen3:4b-instruct": 0.87, "bonsai-2:4b": 1.0})),
])
def test_the_gate_fails_closed_when_the_placement_cannot_be_trusted(why, http):
    rec = measured(QWEN, http)
    assert rec["status"] == "unknown" and rec["note"], why
    assert cg.placement_gate(cg.MODELS[QWEN], rec) == "placement unknown"
    el = cg.eligibility(stats(), placement={QWEN: rec, BONSAI: placed(BONSAI)})
    assert el[QWEN]["eligible"] is False and el[QWEN]["reasons"] == ["placement unknown"]  # perfect latency and failures do not help


def test_a_missing_placement_is_unknown_and_a_full_model_beside_another_is_still_ok():
    assert cg.eligibility(stats())[BONSAI]["reasons"] == ["placement unknown"]  # never measured
    both = FakeOllama(share={"qwen3:4b-instruct": 1.0, "bonsai-2:4b": 1.0})
    assert measured(QWEN, both)["status"] == "ok"  # 100% with others loaded can only be better alone
    assert cg.placement_gate(None, None) is None  # a name the registry does not know has no placement term


def test_the_native_api_is_asked_not_the_openai_compatible_one(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://box:11434/v1")
    http = FakeOllama(share={"bonsai-2:4b": 1.0})
    measured(BONSAI, http)
    assert http.urls == ["http://box:11434/api/ps", "http://box:11434/api/tags"]


def test_a_reference_is_never_picked_even_when_it_is_funniest():
    s = stats()
    s[GEMMA] = {**s[BONSAI]}
    placement = {**PLACED, GEMMA: placed(GEMMA)}  # a reference that is flawless on every other term
    d = cg.decide(s, ranks_for([[GEMMA, QWEN, BONSAI, HAIKU]] * 30), placement=placement)
    assert d["eligibility"][GEMMA]["eligible"] is False and d["eligibility"][GEMMA]["reasons"] == ["reference"]
    assert GEMMA not in d["eligible"] and GEMMA not in (d["best"], d["second"])
    assert d["pick"] == QWEN  # the best eligible by rank once the reference is out of the ranking
    one = cg.decide({k: s[k] for k in (GEMMA, HAIKU)}, ranks_for([[GEMMA, HAIKU]] * 30), placement=placement)
    assert one["pick"] == HAIKU and one["reason"] == "the only eligible model"
    assert cg.decide({GEMMA: s[GEMMA]}, {}, placement=placement)["pick"] is None


def test_the_ballot_has_four_models_and_scoring_keeps_the_reference_out_of_the_pick(data, monkeypatch):
    monkeypatch.setattr(cg.cache, "METRICS", data / "metrics")
    models = MODELS + [GEMMA]
    run(n=60, models=models)
    cg.export_ballot(models, 30)
    d = cg.run_dir()
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert all(sorted(k["letters"]) == list("ABCD") and sorted(k["letters"].values()) == sorted(models) for k in key.values())
    pref = {GEMMA: 0, HAIKU: 1, BONSAI: 2, QWEN: 3}  # the voter's order; qwen was measured 87% on the GPU
    cg.write_votes(d / "votes.csv", {p: "".join(sorted(k["letters"], key=lambda c: pref[k["letters"][c]])) for p, k in key.items()})
    cg.write_record(cg.placement_path(d), {BONSAI: placed(BONSAI), QWEN: placed(QWEN, gpu_share=0.87, size=6.39e9, size_vram=5.6e9),
                                           GEMMA: {"status": "unknown", "gpu_share": None}})
    params = cg.current_params(60, tuple(models), 30)
    res = cg.cache.run_or_load(cg.METRICS_NAME, cg.run, **params)
    dec = res["decision"]
    assert res["votes"]["mean_rank"][GEMMA] == 1.0  # the reference is in the table...
    assert dec["eligible"] == sorted([BONSAI, HAIKU]) and dec["pick"] == HAIKU  # ...and never in the pick
    assert dec["eligibility"][QWEN]["reasons"] == ["partial GPU 87%"] and dec["eligibility"][GEMMA]["reasons"] == ["reference"]
    assert res["placement"][QWEN]["gpu_share"] == 0.87
    stored_text = (data / "metrics" / "model_comparison.json").read_text(encoding="utf-8")
    assert "partial GPU 87%" in stored_text and "thread " not in stored_text
    assert "no: partial GPU 87%" in cg.table(res) and "no: reference" in cg.table(res)
    cg.write_record(cg.placement_path(d), {BONSAI: placed(BONSAI), QWEN: placed(QWEN), GEMMA: placed(GEMMA)})
    with pytest.raises(AssertionError):  # a new measurement invalidates the scored result
        cg.run(**params)


def test_placement_is_measured_after_a_models_first_success_kept_when_ok_and_retried_when_unknown(data):
    three = [QWEN, BONSAI, HAIKU]
    http = FakeOllama()
    run(n=3, models=three, http=http)
    saved = cg.load_placement(cg.run_dir())
    assert set(saved) == {QWEN, BONSAI} and all(r["status"] == "ok" for r in saved.values())  # hosted models are not measured
    assert http.urls.count(PS_URL) == 2  # once each: an ok reading is kept
    x = cg.run_dir() / "x"
    flaky = FakeOllama(ps={"models": []})
    run(n=6, models=three, http=flaky, directory=x)
    assert {r["status"] for r in cg.load_placement(x).values()} == {"unknown"}
    assert flaky.urls.count(PS_URL) == 2 * cg.MAX_PLACEMENT_TRIES  # retried on each success, up to the cap
    again = FakeOllama()
    run(n=7, models=three, http=again, directory=x)
    assert {r["status"] for r in cg.load_placement(x).values()} == {"ok"}
    run(n=8, models=three, http=again, directory=x)
    assert again.urls.count(PS_URL) == 2  # nothing remeasured once ok


def test_a_failed_first_call_measures_nothing_and_the_next_success_does(data):
    http = FakeOllama()
    err = llm.LLMError("503 unavailable", status=503, retryable=True)
    run(n=2, models=[QWEN], clients={QWEN: FakeModel(cg.MODELS[QWEN].model, fail=[err] + [None] * 9)}, http=http)
    assert cg.load_placement(cg.run_dir())[QWEN]["status"] == "ok" and http.urls.count(PS_URL) == 1


def test_power_with_two_eligible_beside_two_references_is_computed_through_decide():
    three = cg.simulate_decision_power(30, 1.0, reps=100, n_boot=500)
    two = cg.simulate_decision_power(30, 1.0, reps=100, n_boot=500, eligible=2, references=2)
    assert 0.5 < three <= 1.0 and 0.0 <= two <= 1.0
    assert cg.simulate_decision_power(30, 0.0, reps=100, n_boot=500, eligible=2, references=2) < 0.2  # no gap: a fluke


# --- D-97 review: one local model in VRAM at a time, so order and warm-up matter ----------------------------------

def blocks(events, kind="call"):
    """The models in the order they were called, consecutive repeats collapsed."""
    import itertools
    return [m for m, _ in itertools.groupby(m for k, m in events if k == kind)]


def test_each_local_model_is_called_in_one_block_so_it_is_loaded_once(data):
    events = []
    run(n=4, models=[QWEN, HAIKU, BONSAI], events=events)
    names = [cg.MODELS[m].model for m in (QWEN, HAIKU, BONSAI)]
    assert blocks(events) == names  # not qwen, haiku, bonsai, qwen, ...: alternating would evict the other local model each time
    per = {m: sum(1 for k, x in events if k == "call" and x == cg.MODELS[m].model) for m in (QWEN, BONSAI)}
    assert per == {QWEN: 4, BONSAI: 4}  # every prompt, one call a bit, all done in the block
    events = []
    run(n=2, models=[BONSAI, QWEN, GEMMA], events=events, directory=cg.run_dir() / "other")
    assert blocks(events) == [cg.MODELS[m].model for m in (BONSAI, QWEN, GEMMA)]


def test_hosted_models_stay_together_prompt_by_prompt_at_the_place_of_the_first():
    assert cg.call_groups([QWEN, BONSAI, HAIKU, GEMMA]) == [[QWEN], [BONSAI], [HAIKU], [GEMMA]]
    assert cg.call_groups([HAIKU, QWEN]) == [[HAIKU], [QWEN]]
    assert [g for g in cg.call_groups(["claude-haiku-5-5"])] == [["claude-haiku-5-5"]]


def test_a_local_model_is_warmed_up_untimed_and_placement_is_read_right_after_it_before_any_call(data):
    events = []
    run(n=2, models=[QWEN], events=events)
    kinds = [k for k, _ in events]
    assert kinds[:2] == ["warm", "ps"] and kinds.index("call") > 1  # resident and alone when it is measured
    assert kinds.count("warm") == 1


def test_the_warm_up_is_not_an_attempt_and_adds_nothing_to_latency_tokens_or_failures(data):
    client = FakeModel(cg.MODELS[QWEN].model)
    cg.run_comparison(3, [QWEN], lambda spec: client, FakeFilter(), pacer=NoWait(), log=lambda s: None, http=FakeOllama())
    assert client.calls == 1 + 3  # no separate warm client: the warm-up goes through this one
    atts = cg.load_attempts(cg.run_dir(), [QWEN], 3)[QWEN]
    assert len(atts) == 3 and all(a["requests"] == 1 and a["input_tokens"] == 1000 for a in atts)  # the warm-up is excluded
    s = cg.call_stats(atts, cg.MODELS[QWEN])
    assert s["n_attempts"] == 3 and s["n_failed"] == 0
    w = cg.read_record(cg.warmup_path(cg.run_dir()))[QWEN]
    assert w["ok"] is True and w["warmup_s"] >= 0 and len(w["history"]) == 1
    assert not any("warm" in json.dumps(a) for a in atts)


def test_a_resume_that_makes_a_call_warms_up_again_and_a_fully_cached_one_does_not(data):
    events = []
    run(n=2, models=[QWEN], events=events)
    run(n=2, models=[QWEN], events=events)  # everything cached: no call, so nothing to warm up for
    assert [k for k, _ in events].count("warm") == 1
    run(n=3, models=[QWEN], events=events)  # one new prompt: the model may have been unloaded since
    assert [k for k, _ in events].count("warm") == 2
    assert len(cg.read_record(cg.warmup_path(cg.run_dir()))[QWEN]["history"]) == 2


def test_warm_up_failure_modes(data):
    down = llm.LLMError("transport: ConnectError", retryable=True)
    with pytest.raises(cg.ConfigError, match=r"warm-up.*ollama serve.*ollama pull qwen3:4b-instruct"):
        run(n=2, models=[QWEN], warm_fail=down)
    assert not (cg.run_dir() / QWEN / "p00.json").exists()  # nothing cached for the failing call
    missing = llm.LLMError("404 model not found", status=404)
    with pytest.raises(cg.ConfigError):
        run(n=2, models=[QWEN], warm_fail=missing)
    slow = llm.LLMError("transport: ReadTimeout", retryable=True)
    with pytest.raises(cg.WarmupError, match="failed twice"):  # a cold load must not be absorbed by a timed call
        run(n=2, models=[QWEN], warm_fail=slow)
    assert not (cg.run_dir() / QWEN / "p00.json").exists()
    assert len(cg.read_record(cg.warmup_path(cg.run_dir()))[QWEN]["history"]) == 4 and not cg.load_placement(cg.run_dir())  # one try per config stop, then both tries


def test_the_warm_up_client_has_a_cold_load_timeout_and_no_retries(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://box:11434/v1")
    c = cg.make_warmup_client(cg.MODELS[BONSAI])
    assert (c.base_url, c.model, c.retries) == ("http://box:11434/v1", "bonsai-2:4b", 0)
    assert c.timeout == c.deadline_s == cg.WARMUP_TIMEOUT_S >= 126  # the measured cold load of bonsai-2:4b was 126 s
    assert c.extra_body == llm.OLLAMA_NO_THINK


# --- D-97 review round 3: re-measure unknown placement, retry the warm-up -----------------------------------------

def test_a_fully_cached_rerun_measures_an_unknown_placement_with_no_timed_call_and_leaves_an_ok_one(data):
    events = []
    run(n=2, models=[QWEN], http=FakeOllama(ps={"models": []}))  # the first run ends with placement unknown
    assert cg.load_placement(cg.run_dir())[QWEN]["status"] == "unknown"
    before = len(cg.load_attempts(cg.run_dir(), [QWEN], 2)[QWEN])
    lines = []
    run(n=2, models=[QWEN], events=events, log_lines=lines)  # all cached; Ollama now reports properly
    assert [k for k, _ in events] == ["warm", "ps"]  # warmed up and measured, and not one timed call
    assert cg.load_placement(cg.run_dir())[QWEN]["status"] == "ok"
    assert len(cg.load_attempts(cg.run_dir(), [QWEN], 2)[QWEN]) == before  # nothing added to the attempts
    events.clear()
    run(n=2, models=[QWEN], events=events)
    assert events == []  # ok is kept: no warm-up, no read


def test_each_placement_is_logged_in_plain_words_with_the_hint_for_a_wrong_context(data):
    tag = cg.MODELS[QWEN].tag()
    lines = []
    run(n=1, models=[QWEN], http=FakeOllama(share={tag: 1.0}), log_lines=lines)
    assert f"placement {QWEN}: 100% -> passes the GPU gate" in lines
    assert any(l.startswith(f"placement {QWEN}") for l in lines)
    for share, ctx, want in ((0.88, 8192, f"placement {QWEN}: 88% on GPU at ctx 8192 -> not eligible (partial GPU)"),
                             (1.0, 4096, f"placement {QWEN}: unknown: loaded at context 4096, not 8192 -> not eligible; "
                                         "restart with OLLAMA_CONTEXT_LENGTH=8192 and rerun run"),
                             (1.0, None, None)):
        d = data / f"x{share}{ctx}"
        lines = []
        http = FakeOllama(share={tag: share}, context=ctx)
        run(n=1, models=[QWEN], http=http, directory=d, log_lines=lines)
        if want:
            assert want in lines
    d = data / "gone"
    lines = []
    run(n=1, models=[QWEN], http=FakeOllama(ps={"models": []}), directory=d, log_lines=lines)
    assert f"placement {QWEN}: unknown: model missing from /api/ps -> not eligible; rerun run to measure again" in lines


def test_a_warm_up_that_fails_once_is_retried_and_both_tries_are_in_the_history(data):
    slow = llm.LLMError("transport: ReadTimeout", retryable=True)
    events = []
    run(n=2, models=[QWEN], warm_fail=[slow], events=events)
    assert [k for k, _ in events].count("warm") == 2
    w = cg.read_record(cg.warmup_path(cg.run_dir()))[QWEN]
    assert w["ok"] is True and len(w["history"]) == 2
    assert len(cg.load_attempts(cg.run_dir(), [QWEN], 2)[QWEN]) == 2  # the run went on


def test_a_warm_up_that_fails_twice_stops_the_run_and_caches_nothing_for_that_block(data):
    slow = llm.LLMError("transport: ReadTimeout", retryable=True)
    with pytest.raises(cg.ConfigError, match="nothing cached"):
        run(n=2, models=[BONSAI, QWEN], warm_fail=[None, slow, slow])  # bonsai warms fine, qwen fails twice
    assert (cg.run_dir() / BONSAI / "p00.json").exists() and not (cg.run_dir() / QWEN).exists()
    assert len(cg.read_record(cg.warmup_path(cg.run_dir()))[QWEN]["history"]) == 2


def test_the_ballot_shows_no_punctuation_that_gives_a_model_away(data, monkeypatch):
    # D-105: only qwen wrote curly quotes, em dashes and ellipses; on the ballot they would unblind its bits.
    qwen = cg.MODELS[QWEN].model
    monkeypatch.setitem(BY_ID, qwen, BY_ID[qwen] + "’s “bit” — wait…")
    _, d = prep(n=2, models=MODELS, filt=FakeFilter())
    ballot = (d / "ballot.md").read_text(encoding="utf-8")
    assert "s \"bit\" - wait..." in ballot  # qwen's bit is shown, in plain form
    assert not set(ballot) & set("’‘“”—–…")
    stored = cg.final(rec_of(QWEN, "p00"))["threads"][0][0]
    assert "—" in stored  # display only: the stored bit keeps what the model wrote
