"""The generator, the retries and the show's render-ahead, bridge and bank fallback. A fake HTTP server and fake voice;
no real API or Kokoro. The timing numbers live in notebooks/decisions-live.ipynb (Exp 20)."""
import asyncio
import json
import pathlib
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import generate, generate_v3, llm, safety  # noqa: E402
from app.bank import BankGenerator  # noqa: E402
from app.director import EPSILON, load_director  # noqa: E402
from app.log import SessionLog, read_session  # noqa: E402
from app.rooms import ReactionConfig, RealClock, SimRoom  # noqa: E402
from app.show import UniformSelector, run_show  # noqa: E402
from app.tts import to_wav  # noqa: E402

CTX = {"venue": "bar", "energy": "high", "age": "mixed"}
FAST = ReactionConfig(holds={3: 0.01, 2: 0.01, 1: 0.01, 0: 0.01, None: 0.0}, timeout_s=0.5)


def reply(n_jokes=5, k=3, tag="x"):
    """The one accepted shape: thread-major, threads[i][j] is joke j of thread i."""
    return json.dumps({"premises": [f"p{i}" for i in range(k)], "threads": [[f"{tag} line {j}.{i}" for j in range(n_jokes)] for i in range(k)]})


def thread_reply(tag="x", premise="p"):
    """The one gen-v4 shape: a premise and 5 lines."""
    return json.dumps({"premise": premise, "jokes": [f"{tag} line {j}." for j in range(5)]})


JOKE_MAJOR = json.dumps({"premises": ["p"], "jokes": [[f"line {j}.{i}" for i in range(3)] for j in range(5)]})


def completion(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}]}


def server(*script):
    """Transport answering from a script of (status, body) or an exception to raise; repeats the last entry."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        step = script[min(len(seen) - 1, len(script) - 1)]
        if isinstance(step, Exception):
            raise step
        status, body = step
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler), seen


def client(*script, **options):
    transport, seen = server(*script)
    return llm.Client("http://fake", "k", "m", transport=transport, sleep=lambda s: None, **options), seen


# --- the client's retries ---

def test_a_429_is_retried_until_it_succeeds_and_every_attempt_is_counted():
    c, seen = client((429, {}), (503, {}), (200, completion("ok")), retries=2)
    assert c.chat("s", "u") == "ok"
    assert (c.attempts, c.retried, c.failed, len(seen)) == (3, 2, 0, 3)


def test_retries_are_bounded_and_the_last_error_is_raised():
    c, seen = client((429, {}), retries=2)
    with pytest.raises(llm.LLMError) as e:
        c.chat("s", "u")
    assert e.value.status == 429 and len(seen) == 3 and c.failed == 1


def test_a_client_error_and_a_provider_block_are_not_retried():
    for step in ((400, {}), (200, completion("x", finish="content_filter"))):
        c, seen = client(step, retries=3)
        with pytest.raises(llm.LLMError):
            c.chat("s", "u")
        assert len(seen) == 1


@pytest.mark.parametrize("raw", [
    "<think>hmm {not json}</think>\n" + thread_reply(),  # a reasoning block before the JSON
    "<think>\nmulti\nline\n</think>" + thread_reply(),
    "reasoning the template opened in the prompt</think>" + thread_reply(),  # only the closing tag is in the reply
    "<think>a</think><think>b</think>" + thread_reply(),
    thread_reply(),  # no block: untouched
])
def test_a_reasoning_block_never_breaks_the_json_parse(raw):
    c, _ = client((200, completion(raw)))
    assert generate.ThreadGenerator(c).generate("neurotic", 0, 0, {"venue": "club", "energy": "high", "age": "adults"}).texts


def test_a_reply_that_is_only_an_unfinished_reasoning_block_is_an_empty_reply():
    c, _ = client((200, completion("<think>still thinking about")))
    with pytest.raises(llm.LLMError, match="empty reply"):
        c.chat("s", "u")


def test_extra_body_fields_are_sent_with_every_request():
    c, seen = client((200, completion("ok")), extra_body=llm.OLLAMA_NO_THINK)
    c.chat("s", "u", json_reply=True)
    assert seen[0]["reasoning_effort"] == "none" and seen[0]["response_format"] == {"type": "json_object"}
    c, seen = client((200, completion("ok")))
    c.chat("s", "u")
    assert "reasoning_effort" not in seen[0]


SCHEMA = {"type": "object", "required": ["x"]}


def test_json_schema_is_sent_only_when_asked_and_only_for_a_schema():
    c, seen = client((200, completion("ok")), json_schema=True)
    c.chat("s", "u", json_reply=SCHEMA)
    assert seen[0]["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "schema": SCHEMA}}
    c.chat("s", "u", json_reply=True)  # no schema to send: any JSON
    assert seen[1]["response_format"] == {"type": "json_object"}
    c.chat("s", "u")
    assert "response_format" not in seen[2]
    c, seen = client((200, completion("ok")))  # the default, and what Gemini's client keeps: json_object even with a schema
    c.chat("s", "u", json_reply=SCHEMA)
    assert seen[0]["response_format"] == {"type": "json_object"}


def test_a_timeout_is_retried():
    c, seen = client(httpx.ReadTimeout("slow"), (200, completion("ok")), retries=1)
    assert c.chat("s", "u") == "ok" and len(seen) == 2


def test_the_deadline_stops_a_retry_that_would_overrun_it():
    c, seen = client((503, {}), retries=5, backoff_s=2.0, deadline_s=1.0)
    with pytest.raises(llm.LLMError):
        c.chat("s", "u")
    assert len(seen) == 1  # the first backoff alone (2 s) is past the 1 s deadline


def test_a_hanging_server_cannot_run_past_the_deadline():
    def handler(request):  # a server that never answers: the request's own timeout is what ends it
        time.sleep(request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("hung")
    c = llm.Client("http://fake", "k", "m", timeout=0.6, retries=3, backoff_s=0.05, deadline_s=1.2,
                   transport=httpx.MockTransport(handler))
    start = time.monotonic()
    with pytest.raises(llm.LLMError):
        c.chat("s", "u")
    assert time.monotonic() - start <= 1.2 + 0.2  # before the fix each retry got the full 0.6 s
    assert c.attempts >= 2  # it did retry, with the second attempt's timeout clipped to what was left


def test_default_client_never_retries():
    c, seen = client((503, {}))
    with pytest.raises(llm.LLMError):
        c.chat("s", "u")
    assert len(seen) == 1


# --- the frozen gen-v3 generator (Exp 24's harness; the show runs gen-v4 below) ---

def test_one_call_per_bit_serves_all_five_jokes_with_three_candidates_each():
    c, seen = client((200, completion(reply())))
    g = generate_v3.GeminiGenerator(c)
    batches = [g.generate("cynic", 0, j, CTX) for j in range(5)]
    assert len(seen) == 1
    assert [len(b.texts) for b in batches] == [3] * 5 and batches[2].texts[1] == "x line 2.1"
    assert seen[0]["temperature"] == generate_v3.TEMPERATURE and seen[0]["response_format"] == {"type": "json_object"}
    assert batches[0].prompt.startswith(seen[0]["messages"][0]["content"])  # the log holds what the model saw


def test_only_the_premise_of_the_played_thread_is_told_to_the_next_bit():
    c, seen = client((200, completion(reply())))
    g = generate_v3.GeminiGenerator(c)
    g.generate("cynic", 0, 0, CTX)
    g.played_thread(0, 1)
    g.generate("optimist", 1, 0, CTX)
    told = json.loads(seen[1]["messages"][1]["content"])["premises_already_told"]
    assert len(seen) == 2 and told == ["p1"]  # not p0 or p2, which the room never heard
    g.played_thread(5, 0)  # a bit that was never generated, or a thread with no premise, is ignored
    g.played_thread(1, 9)


@pytest.mark.parametrize("bad", [JOKE_MAJOR,  # the old shape is rejected, not guessed at
                                 json.dumps({"threads": ["Line one.", "Line two.", "Line three."]}),  # string rows
                                 json.dumps({"threads": "xyz"}), json.dumps({"threads": [[1] * 5] * 3}),
                                 "not json", reply(n_jokes=4), reply(n_jokes=6), reply(k=0), reply(k=2), reply(k=4),
                                 json.dumps({"threads": [["a"] * 5, ["b"] * 5, ["", "b", "b", "b", "b"]]}),
                                 json.dumps({"threads": [["a"] * 5, ["b"] * 5, [chr(0x1F642) + "!", "b", "b", "b", "b"]]}),
                                 json.dumps([1, 2])])
def test_a_malformed_reply_raises_for_every_joke_without_a_second_call(bad):
    c, seen = client((200, completion(bad)))
    g = generate_v3.GeminiGenerator(c)
    for joke in range(5):
        with pytest.raises(generate.GenerationError):
            g.generate("cynic", 0, joke, CTX)
    assert len(seen) == 1


def test_a_timeout_raises_for_every_joke_without_a_second_call():
    c, seen = client(httpx.ConnectTimeout("down"))
    g = generate_v3.GeminiGenerator(c)
    for joke in range(5):
        with pytest.raises(llm.LLMError):
            g.generate("cynic", 0, joke, CTX)
    assert len(seen) == 1


def test_fence_stripping_is_shared_with_the_safety_parser():
    fence = chr(96) * 3
    assert llm.strip_fence(fence + "json" + chr(10) + "{}" + chr(10) + fence) == "{}" and llm.strip_fence("  {}  ") == "{}"
    fenced = fence + "json" + chr(10) + json.dumps({"verdicts": [{"id": 0, "ok": True}]}) + chr(10) + fence
    assert safety._parse(fenced, 1)[0].ok


def test_a_thread_major_reply_becomes_joke_major_batches_with_thread_i_at_index_i():
    fence = chr(96) * 3
    premises, lines = generate_v3.parse(fence + "json" + chr(10) + reply() + chr(10) + fence)
    assert premises == ["p0", "p1", "p2"] and [len(r) for r in lines] == [3] * 5
    assert lines[4] == ["x line 4.0", "x line 4.1", "x line 4.2"] and lines[0][2] == "x line 0.2"


def test_the_prompt_holds_the_identity_the_rules_and_teens_only_for_teens():
    c, seen = client((200, completion(reply())))
    g = generate_v3.GeminiGenerator(c)
    g.generate("wordsmith", 0, 0, CTX)
    g.generate("wordsmith", 1, 0, {**CTX, "age": "teens"})
    adults, teens = (s["messages"][0]["content"] for s in seen)
    for text in (adults, teens):
        assert "slightly anxious robot" in text and "wordplay" in text
        assert "bodies or appearance" in text and "protected groups" in text and "named people" in text
    assert "teenagers" in teens and "teenagers" not in adults
    assert set(generate.MODES) >= {"neurotic", "cynic", "observer", "wordsmith", "optimist", "absurdist"}


@pytest.mark.parametrize("age", ["mixed", "teens"])
def test_every_bridge_line_passes_the_rules(age):
    assert all(safety.check_rules(t, age).ok for t in generate.BRIDGES)


# --- the gen-v4 generator: one thread per call, three calls in parallel on one prompt (D-110) ---

def parallel_client(*steps, wait_for=3, hold_s=3.0, **options):
    """A transport that holds every request until `wait_for` have arrived (or `hold_s` passes), so only calls that
    overlap in time get through together. `steps[i]` answers the i-th request to arrive: a reply text or an
    Exception to raise (the last one repeats). Returns (client, seen bodies, state with the peak held at once)."""
    seen, lock, gate = [], threading.Lock(), threading.Event()
    state = {"in": 0, "peak": 0}

    def handler(request):
        with lock:
            seen.append(json.loads(request.content))
            n = len(seen)
            state["in"] += 1
            state["peak"] = max(state["peak"], state["in"])
            if state["in"] >= wait_for:
                gate.set()
        gate.wait(hold_s)  # a sequential generator never reaches wait_for, and gets through only after hold_s
        with lock:
            state["in"] -= 1
        step = steps[min(n - 1, len(steps) - 1)]
        if isinstance(step, Exception):
            raise step
        return httpx.Response(200, json=completion(step))
    c = llm.Client("http://fake", "k", "m", transport=httpx.MockTransport(handler), sleep=lambda s: None, **options)
    return c, seen, state


def test_parse_accepts_only_one_premise_and_five_lines():
    assert generate.parse(thread_reply(premise=" a robot at a bar ")) == ("a robot at a bar", [f"x line {j}." for j in range(5)])
    fence = chr(96) * 3
    assert generate.parse(fence + "json" + chr(10) + thread_reply() + chr(10) + fence)[0] == "p"
    for bad in (reply(), JOKE_MAJOR, "not json", json.dumps([1]), json.dumps({"jokes": ["a"] * 5}),
                json.dumps({"premise": "p"}), json.dumps({"premise": 3, "jokes": ["a"] * 5}),
                json.dumps({"premise": "p", "jokes": "abcde"}), json.dumps({"premise": "p", "jokes": ["a"] * 4}),
                json.dumps({"premise": "p", "jokes": ["a"] * 6}), json.dumps({"premise": "p", "jokes": [1] * 5}),
                json.dumps({"premise": "p", "jokes": ["a", "a", "a", "a", ""]}),
                json.dumps({"premise": "p", "jokes": ["a", "a", "a", "a", chr(0x1F642) + "!"]})):
        with pytest.raises(generate.GenerationError):
            generate.parse(bad)


def test_three_identical_calls_overlap_in_time_and_serve_all_five_jokes_with_three_candidates_each():
    c, seen, state = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g = generate.ThreadGenerator(c)
    start = time.monotonic()
    batches = [g.generate("cynic", 0, j, CTX) for j in range(5)]
    assert time.monotonic() - start < 2.5  # the server holds 3 s if the calls come one after another
    assert len(seen) == 3 and state["peak"] == 3
    assert seen[0] == seen[1] == seen[2]  # same system and user message: independent samples of one prompt
    assert "premises_this_bit" not in seen[0]["messages"][1]["content"]
    assert json.loads(seen[0]["messages"][1]["content"]) == {"venue": "bar", "energy": "high", "premises_already_told": []}
    assert seen[0]["temperature"] == generate.TEMPERATURE and seen[0]["response_format"] == {"type": "json_object"}
    assert [len(b.texts) for b in batches] == [3] * 5
    assert sorted(batches[2].texts) == ["a line 2.", "b line 2.", "c line 2."]
    assert batches[0].prompt.startswith(seen[0]["messages"][0]["content"])  # the log holds what the model saw
    assert (c.attempts, c.failed) == (3, 0)


def test_threads_keep_the_order_of_their_calls_not_of_their_completion():
    n = {"i": 0}
    lock = threading.Lock()

    def handler(request):
        with lock:
            mine = n["i"]
            n["i"] += 1
        if mine == 0:
            time.sleep(0.3)  # the first call to arrive finishes last
        return httpx.Response(200, json=completion(thread_reply(tag=f"call{mine}", premise=f"p{mine}")))
    g = generate.ThreadGenerator(llm.Client("http://fake", "k", "m", transport=httpx.MockTransport(handler)))
    lines = [g.generate("cynic", 0, j, CTX).texts for j in range(5)]
    premises = g._bits[0][2]
    assert sorted(premises) == ["p0", "p1", "p2"]
    for j in range(5):  # premise i and line j of thread i come from the same call
        assert [t.split()[0] for t in lines[j]] == [f"call{p[1:]}" for p in premises]
    first = [t.split()[0] for t in lines[0]]
    assert all([t.split()[0] for t in row] == first for row in lines)


def test_a_failed_or_malformed_call_drops_only_its_thread():
    for broken in (httpx.ConnectTimeout("down"), "not json", reply(), json.dumps({"premise": "p", "jokes": ["a"] * 4})):
        c, seen, _ = parallel_client(thread_reply(tag="a", premise="pa"), broken, thread_reply(tag="c", premise="pc"))
        g = generate.ThreadGenerator(c)
        batches = [g.generate("cynic", 0, j, CTX) for j in range(5)]
        assert len(seen) == 3 and [len(b.texts) for b in batches] == [2] * 5
        assert len(g._bits[0][2]) == 2  # premises line up with the threads that came back
        premise_of = {"a": "pa", "c": "pc"}  # the premise each surviving call wrote, keyed by its lines' tag
        for thread in (0, 1):
            g.played_thread(0, thread)
            assert g._told[-1] == premise_of[batches[0].texts[thread].split()[0]]  # the played thread's, not another's
        assert len(g._told) == 2
        assert len(batches[0].threads_lost) == 1 and "line 0" not in batches[0].threads_lost[0]  # a reason, no text
        assert batches[4].threads_lost == batches[0].threads_lost


def test_with_no_thread_every_joke_raises_and_no_further_call_is_made():
    for bad in (httpx.ConnectTimeout("down"), "not json"):
        c, seen, _ = parallel_client(bad)
        g = generate.ThreadGenerator(c)
        for joke in range(5):
            with pytest.raises(generate.GenerationError):
                g.generate("cynic", 0, joke, CTX)
        assert len(seen) == 3
        g.played_thread(0, 0)
        assert g._told == []
        with pytest.raises(generate.GenerationError) as e:
            g.generate("cynic", 0, 0, CTX)
        assert len(e.value.threads_lost) == 3 and all(isinstance(x, str) for x in e.value.threads_lost)


class SpyClient:
    deadline_s, model, local = 12.0, "m", False

    def __init__(self):
        self.deadlines, self.lock = [], threading.Lock()

    def chat(self, system, user, **kw):
        with self.lock:
            self.deadlines.append(kw["deadline_s"])
        return thread_reply()


def test_the_whole_bit_shares_one_budget_and_no_call_starts_without_time():
    spy = SpyClient()
    generate.ThreadGenerator(spy).generate("cynic", 0, 0, CTX)
    assert spy.deadlines == [12.0] * 3  # the calls start together: each has the bit's whole budget, none its own
    spy = SpyClient()
    spy.deadline_s = 4.0  # a client with a tighter deadline of its own keeps it
    generate.ThreadGenerator(spy).generate("cynic", 0, 0, CTX)
    assert spy.deadlines == [4.0] * 3
    spy = SpyClient()
    with pytest.raises(generate.GenerationError):
        generate.ThreadGenerator(spy, budget_s=llm.MIN_ATTEMPT_S - 0.1).generate("cynic", 0, 0, CTX)
    assert spy.deadlines == []


def test_a_slow_server_ends_the_bit_at_the_shared_deadline_with_no_further_call():
    seen = []

    def handler(request):
        seen.append(1)
        time.sleep(request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("slow")
    c = llm.Client("http://fake", "k", "m", timeout=5.0, retries=2, backoff_s=0.05, deadline_s=5.0,
                   transport=httpx.MockTransport(handler))
    g = generate.ThreadGenerator(c, budget_s=0.8)
    start = time.monotonic()
    for joke in range(5):
        with pytest.raises(generate.GenerationError):
            g.generate("cynic", 0, joke, CTX)
    assert time.monotonic() - start < 1.6  # each call is cut at the 0.8 s the bit has, retries included
    assert len(seen) == 3  # one attempt per call: a retry would be cut off before it started
    time.sleep(1.0)  # past the bit's 0.8 s and every call's own deadline: nothing may start later
    assert len(seen) == 3
    for joke in range(5):
        with pytest.raises(generate.GenerationError):
            g.generate("cynic", 0, joke, CTX)
    assert len(seen) == 3  # the failure is cached for the bit's other jokes


def test_a_call_that_ignores_its_timeout_cannot_hold_the_bit_past_the_budget(monkeypatch):
    monkeypatch.setattr(generate, "HANG_GRACE_S", 0.3)

    class Hung(SpyClient):
        def chat(self, system, user, **kw):
            time.sleep(3.0)  # a client bug: it never honours deadline_s
            return thread_reply()
    g = generate.ThreadGenerator(Hung(), budget_s=0.5)
    start = time.monotonic()
    with pytest.raises(generate.GenerationError) as e:
        g.generate("cynic", 0, 0, CTX)
    assert time.monotonic() - start < 1.5 and len(e.value.threads_lost) == 3
    assert all("outlived" in x for x in e.value.threads_lost)


def test_client_counters_stay_exact_when_calls_run_on_many_threads():
    ok = llm.Client("http://fake", "k", "m", sleep=lambda s: None, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={**completion("ok"), "usage": {"prompt_tokens": 3, "completion_tokens": 2}})))
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda _: [ok.chat("s", "u") for _ in range(200)], range(16)))
    assert (ok.attempts, ok.input_tokens, ok.output_tokens) == (3200, 9600, 6400)
    bad = llm.Client("http://fake", "k", "m", sleep=lambda s: None, retries=1,
                     transport=httpx.MockTransport(lambda request: httpx.Response(503, json={})))

    def call(_):
        for _ in range(100):
            with pytest.raises(llm.LLMError):
                bad.chat("s", "u")
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(call, range(16)))
    assert (bad.attempts, bad.retried, bad.failed) == (3200, 1600, 1600)


def test_a_per_call_deadline_replaces_the_clients_own_for_that_call_only():
    c, seen = client((503, {}), retries=5, backoff_s=2.0, deadline_s=60.0)
    with pytest.raises(llm.LLMError):
        c.chat("s", "u", deadline_s=1.0)  # the first backoff (2 s) is past this call's 1 s
    assert len(seen) == 1
    c, seen = client((503, {}), (200, completion("ok")), retries=5, backoff_s=0.01, deadline_s=0.0001)
    assert c.chat("s", "u", deadline_s=30.0) == "ok" and len(seen) == 2


def test_only_the_premise_of_the_played_thread_is_told_to_the_next_bit():
    c, seen, _ = parallel_client(*(thread_reply(premise=f"p{i}") for i in range(6)))
    g = generate.ThreadGenerator(c)
    g.generate("cynic", 0, 0, CTX)
    first = g._bits[0][2]
    g.played_thread(0, 1)
    g.generate("optimist", 1, 0, CTX)
    told = json.loads(seen[3]["messages"][1]["content"])["premises_already_told"]
    assert told == [first[1]] and len(seen) == 6  # not the premises of the threads the room never heard
    assert all(json.loads(s["messages"][1]["content"])["premises_already_told"] == told for s in seen[3:])
    g.played_thread(5, 0)  # a bit that was never generated, or a thread with no premise, is ignored
    g.played_thread(1, 9)
    assert g._told == told


def test_the_prompt_holds_the_identity_the_rules_and_teens_only_for_teens():
    c, seen, _ = parallel_client(thread_reply())
    g = generate.ThreadGenerator(c)
    g.generate("wordsmith", 0, 0, CTX)
    g.generate("wordsmith", 1, 0, {**CTX, "age": "teens"})
    adults, teens = (s["messages"][0]["content"] for s in (seen[0], seen[3]))
    for text in (adults, teens):
        assert "slightly anxious robot" in text and "wordplay" in text and "one thread" in text
        assert "bodies or appearance" in text and "protected groups" in text and "named people" in text
        assert "premises_this_bit" not in text and "alternative threads" not in text
    assert "teenagers" in teens and "teenagers" not in adults


def test_a_bit_that_got_fewer_threads_still_plays_one_through_the_show(tmp_path):
    rows, voice = show(tmp_path, Gen(threads=2), Filter(), n_bits=2)
    for bit in range(2):
        plays = [r for r in rows if r["event"] == "joke_played" and r["bit"] == bit]
        assert len(plays) == 5 and len({thread_of(r["text"]) for r in plays}) == 1
        assert [r["pick_prob"] for r in plays] == [0.5] * 5
    assert all(len(r["items"]) == 2 for r in rows if r["event"] == "candidates")  # the log shows how many threads it got
    rows, _ = show(tmp_path / "lost", Gen(threads=2, lost=["LLMError: transport: ReadTimeout"]), Filter(), n_bits=1)
    cand = [r for r in rows if r["event"] == "candidates"]
    assert len(cand) == 5 and all(r["threads_lost"] == ["LLMError: transport: ReadTimeout"] for r in cand)
    rows, _ = show(tmp_path / "all_lost", Gen(fail=True, lost=["LLMError: a", "LLMError: b", "GenerationError: c"]),
                   Filter(), fallback=BankGenerator(), n_bits=1)
    cand = [r for r in rows if r["event"] == "candidates"]
    assert cand and all(r["threads_lost"] == ["LLMError: a", "LLMError: b", "GenerationError: c"] for r in cand)
    rows, _ = show(tmp_path / "none", Gen(), Filter(), n_bits=1)
    assert all(r["threads_lost"] == [] for r in rows if r["event"] == "candidates")  # always present, empty when none
    rows, _ = show(tmp_path / "one", Gen(threads=1), Filter(), n_bits=1)
    assert [r["pick_prob"] for r in rows if r["event"] == "joke_played"] == [1.0] * 5
    assert all(len(r["items"]) == 1 for r in rows if r["event"] == "candidates")


# --- D-111: the counted schema, the re-ask, and the connected prompt as a candidate ---

def per_call_server(bad_for, bad_reply="not json", delay_s=0.0, **options):
    """A client whose requests are answered by the pool thread that made them: a thread whose index is in `bad_for`
    gets `bad_reply` on its first `bad_for[index]` requests, then a good thread. Returns (client, request bodies)."""
    seen, lock, counts = [], threading.Lock(), {}

    def handler(request):
        who = threading.current_thread().name.rsplit("_", 1)[-1]
        with lock:
            seen.append(json.loads(request.content))
            counts[who] = counts.get(who, 0) + 1
            n = counts[who]
        if delay_s:
            time.sleep(delay_s)
        bad = who in bad_for and n <= bad_for[who]
        return httpx.Response(200, json=completion(bad_reply if bad else thread_reply(tag=f"call{who}", premise=f"p{who}")))
    return llm.Client("http://fake", "k", "m", transport=httpx.MockTransport(handler), sleep=lambda s: None,
                      **options), seen


def test_the_counted_schema_is_sent_on_the_ollama_path_and_only_there():
    c = lambda **o: llm.Client("http://fake", "", "m", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=completion(thread_reply()))), **o)
    ollama, plain = c(json_schema=True, local=True), c()
    sent = []
    for cl in (ollama, plain):
        inner = cl._once
        cl._once = lambda system, user, json_reply, temperature, timeout, inner=inner: (sent.append(json_reply), inner(system, user, json_reply, temperature, timeout))[1]
        generate.ThreadGenerator(cl).generate("cynic", 0, 0, CTX)
    assert sent[:3] == [generate.COUNTED_REPLY_SCHEMA] * 3 and sent[3:] == [generate.REPLY_SCHEMA] * 3
    jokes = generate.COUNTED_REPLY_SCHEMA["properties"]["jokes"]
    assert (jokes["minItems"], jokes["maxItems"]) == (generate.JOKES, generate.JOKES) == (5, 5)
    assert "minItems" not in generate.REPLY_SCHEMA["properties"]["jokes"]
    assert generate.REPLY_SCHEMA["properties"]["jokes"] == {"type": "array", "items": {"type": "string"}}  # the plain one is unchanged


def test_the_ollama_request_body_carries_the_counted_schema_in_its_response_format():
    c, seen = per_call_server({}, json_schema=True, local=True)
    generate.ThreadGenerator(c).generate("cynic", 0, 0, CTX)
    for body in seen:
        schema = body["response_format"]["json_schema"]["schema"]
        assert body["response_format"]["type"] == "json_schema"
        assert (schema["properties"]["jokes"]["minItems"], schema["properties"]["jokes"]["maxItems"]) == (5, 5)
    c, seen = per_call_server({})  # an OpenAI-compatible endpoint without json_schema gets json_object, as before
    generate.ThreadGenerator(c).generate("cynic", 0, 0, CTX)
    assert all(b["response_format"] == {"type": "json_object"} for b in seen)


def test_a_bad_shape_is_asked_for_once_more_and_the_thread_is_kept():
    for bad in ("not json", json.dumps({"premise": "p", "jokes": ["a"] * 4}), json.dumps({"premise": "p", "jokes": ["a"] * 6}),
                json.dumps({"premise": "p", "jokes": ["one string with all five lines packed together"]}),  # the show's [1, 1, 1] shape
                json.dumps({"premise": "p", "jokes": "one string, not an array"})):
        c, seen = per_call_server({"0": 1}, bad_reply=bad, retries=2)
        g = generate.ThreadGenerator(c)
        batches = [g.generate("cynic", 0, j, CTX) for j in range(5)]
        assert [len(b.texts) for b in batches] == [3] * 5 and batches[0].threads_lost == ()
        assert (len(seen), c.bad_replies, c.retried) == (4, 1, 0)  # one extra call, counted, and not a provider retry


def test_a_second_bad_reply_drops_only_that_thread_and_says_why():
    c, seen = per_call_server({"0": 2}, retries=2)
    g = generate.ThreadGenerator(c)
    batches = [g.generate("cynic", 0, j, CTX) for j in range(5)]
    assert [len(b.texts) for b in batches] == [2] * 5
    assert (len(seen), c.bad_replies) == (4, 2)  # 3 first calls and exactly one re-ask, never a third try
    assert len(batches[0].threads_lost) == 1 and batches[0].threads_lost[0].startswith("GenerationError: malformed reply")
    assert "line" not in batches[0].threads_lost[0]  # a reason, not text


def test_the_reask_stays_inside_the_bits_deadline():
    c, seen = per_call_server({"0": 1, "1": 1, "2": 1}, delay_s=0.3, retries=2)
    g = generate.ThreadGenerator(c, budget_s=0.5)  # the first reply lands at 0.3 s: a second 0.3 s call cannot fit
    start = time.monotonic()
    with pytest.raises(generate.GenerationError) as e:
        g.generate("cynic", 0, 0, CTX)
    assert len(seen) == 3 and len(e.value.threads_lost) == 3 and time.monotonic() - start < 1.0
    c, seen = per_call_server({"0": 1}, delay_s=0.05, retries=2)
    assert len(generate.ThreadGenerator(c, budget_s=5.0).generate("cynic", 0, 0, CTX).texts) == 3 and len(seen) == 4


def test_the_live_prompt_is_still_gen_v4_and_byte_for_byte_the_one_exp_26_timed():
    import hashlib
    assert generate.PROMPT_ID == "gen-v4" and generate.PROMPTS["gen-v4"] is generate.SYSTEM
    assert hashlib.sha256(generate.SYSTEM.encode()).hexdigest()[:16] == "c0941a33c8dbdda8"  # moves only with a decision
    c, seen = per_call_server({})
    g = generate.ThreadGenerator(c)
    g.generate("cynic", 0, 0, CTX)
    assert g.prompt_id == "gen-v4" and g.generator_id == "gen-v4:m"
    assert seen[0]["messages"][0]["content"] == generate.SYSTEM.format(
        mode="cynic", mode_note=generate.MODES["cynic"], jokes=5, teens="")
    assert g.generate("cynic", 0, 0, CTX).prompt_id == "gen-v4:cynic:0:0"


def test_the_connected_prompt_changes_only_the_structure_and_is_used_when_asked_for():
    v4, v5 = generate.SYSTEM, generate.SYSTEM_CONNECTED
    assert v4 != v5 and generate.PROMPTS["gen-v5"] is v5
    assert v4.replace(generate._STRUCTURE, "@") == v5.replace(generate._STRUCTURE_CONNECTED, "@")  # the rest is shared
    for must in ("one premise", "builds on the line right before", "callback to line 1", "must not start a new topic",
                 "slightly anxious robot", "bodies or appearance", "{teens}", "{mode}"):
        assert must in v5
    c, seen = per_call_server({})
    g = generate.ThreadGenerator(c, prompt_id="gen-v5")
    batch = g.generate("wordsmith", 0, 0, {**CTX, "age": "teens"})
    system = seen[0]["messages"][0]["content"]
    assert system == v5.format(mode="wordsmith", mode_note=generate.MODES["wordsmith"], jokes=5, teens=generate.TEENS_RULE)
    assert batch.prompt_id == "gen-v5:wordsmith:0:0" and g.generator_id == "gen-v5:m" and "teenagers" in system
    with pytest.raises(ValueError):
        generate.ThreadGenerator(c, prompt_id="gen-v9")


# --- the show around it ---

class FakeVoice:
    def __init__(self, delay=0.0):
        self.delay, self.events, self.lock, self.times = delay, [], threading.Lock(), {}

    def synthesize(self, text):
        with self.lock:
            self.events.append(("synth", text))
        time.sleep(self.delay)
        return to_wav(np.zeros(240))


class PickyVoice(FakeVoice):
    """Raises for lines the voice cannot say, the way Kokoro does for a line with no sound in it."""

    def __init__(self, cannot, **kw):
        super().__init__(**kw)
        self.cannot = cannot

    def synthesize(self, text):
        if self.cannot(text):
            raise ValueError(f"no audio produced for: {text!r}")
        return super().synthesize(text)


class FakePlayer:
    def __init__(self, voice, seconds=0.05):
        self.voice, self.seconds = voice, seconds

    async def play(self, wav, text):
        self.voice.events.append(("start", text))
        self.voice.times[("start", text)] = time.time()
        await asyncio.sleep(self.seconds)
        self.voice.events.append(("end", text))
        self.voice.times[("end", text)] = time.time()


class Filter:
    """Passes lines containing 'ok'; records the order of prefetch and check calls."""

    def __init__(self, rule=lambda t: "ok" in t, bank_rule=lambda t: True):
        self.rule, self.bank_rule, self.calls = rule, bank_rule, []

    def check_rules(self, text, age):  # what the bank fallback runs: the rules alone, no API call
        ok = self.bank_rule(text)
        return safety.Verdict(ok, "ok" if ok else "rule")

    def prefetch(self, texts, age):
        self.calls.append(("prefetch", list(texts)))

    def check(self, text, age):
        self.calls.append(("check", text))
        ok = self.rule(text)
        return safety.Verdict(ok, "ok" if ok else "no")


class Gen:
    def __init__(self, tag="ok", fail=False, delay=0.0, threads=3, lost=()):
        self.generator_id, self.tag, self.fail, self.delay, self.calls = f"gen-{tag}", tag, fail, delay, 0
        self.lost = tuple(lost)  # the reasons its dropped threads are logged with
        self.threads = threads  # how many threads the bit got: a bit whose other calls failed has fewer

    def generate(self, mode, bit, joke, context):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            err = llm.LLMError("http 503", 503)
            err.threads_lost = list(self.lost)
            raise err
        return generate.Batch("p", f"prompt {bit}", [f"{self.tag} {bit}.{joke}.{i}" for i in range(self.threads)], self.lost)


def show(tmp_path, gen, flt, voice=None, fallback=None, bridges=None, n_bits=3, seed=0):
    voice = voice or FakeVoice()
    log = SessionLog(tmp_path)
    asyncio.run(run_show(load_director([], seed=1), "sim", CTX, generator=gen, safety=flt,
                         selector=UniformSelector(seed), tts=voice, player=FakePlayer(voice), source=SimRoom(CTX, 0),
                         clock=RealClock(), log=log, n_bits=n_bits, config=FAST, fallback=fallback,
                         bridge_lines=bridges))
    return read_session(log.path), voice


def played(rows):
    return [r["text"] for r in rows if r["event"] == "joke_played"]


def test_every_line_that_reaches_the_voice_passed_the_filter_first(tmp_path):
    flt = Filter(rule=lambda t: t.endswith(".1") or t.endswith(".2"))  # candidate 0 of every joke is rejected
    rows, voice = show(tmp_path, Gen(), flt)
    spoken = {t for kind, t in voice.events if kind == "synth"}
    assert spoken == set(played(rows)) and len(spoken) == 15
    assert not any(t.endswith(".0") for t in spoken)
    for r in (r for r in rows if r["event"] == "candidates"):
        assert [i["verdict"] for i in r["items"]] == ["reject", "pass", "pass"]


def test_one_prefetch_per_bit_with_all_candidates_before_any_check(tmp_path):
    flt = Filter()
    show(tmp_path, Gen(), flt)
    kinds = [c[0] for c in flt.calls]
    assert kinds.count("prefetch") == 3
    for start in range(0, len(kinds), 16):  # one prefetch, then 5 jokes x 3 checks
        assert kinds[start] == "prefetch" and len(flt.calls[start][1]) == 15


def test_next_line_is_rendered_while_the_current_one_plays(tmp_path):
    rows, voice = show(tmp_path, Gen(), Filter(), voice=FakeVoice(delay=0.01))
    order = [e for e in voice.events if e[1].startswith("ok")]
    for bit in range(3):
        texts = [t for t in played(rows) if t.startswith(f"ok {bit}.")]
        for current, nxt in zip(texts, texts[1:]):
            assert order.index(("synth", nxt)) < order.index(("end", current))


def test_a_failed_generation_falls_back_to_the_bank_and_says_so(tmp_path):
    bank = BankGenerator()
    rows, _ = show(tmp_path, Gen(fail=True), Filter(), fallback=bank)
    assert rows[-1]["status"] == "finished" and len(played(rows)) == 15
    cands = [r for r in rows if r["event"] == "candidates"]
    assert len(cands) == 15 and {r["generator_id"] for r in cands} == {"gen-ok"}  # one row per joke, the generator's id
    assert all(len(r["items"]) == 1 and "hand-checked bank: generation failed: LLMError: http 503 (bank-v1)" in
               r["items"][0]["reason"] for r in cands)


def test_all_candidates_rejected_falls_back_to_the_bank_inside_the_same_row(tmp_path):
    flt = Filter(rule=lambda t: not t.startswith("ok 0.2"))
    rows, _ = show(tmp_path, Gen(), flt, fallback=BankGenerator())
    by_joke = {}
    for r in rows:
        if r["event"] == "candidates":
            assert (r["bit"], r["joke"]) not in by_joke  # never a second row for a joke
            by_joke[(r["bit"], r["joke"])] = r
    row = by_joke[(0, 2)]
    assert [i["verdict"] for i in row["items"]] == ["reject"] * 3 + ["pass"]  # generated first, then the bank line
    assert row["items"][3]["reason"].startswith("hand-checked bank: every candidate rejected (bank-v1); rules: ")
    assert row["generator_id"] == "gen-ok" and len(by_joke) == 15 and len(played(rows)) == 15
    chosen = next(r for r in rows if r["event"] == "joke_played" and (r["bit"], r["joke"]) == (0, 2))
    assert chosen["chosen"] == 3 and row["items"][3]["text"] == chosen["text"]  # chosen indexes into the one list


def test_without_a_fallback_a_failed_bit_is_skipped_but_the_log_says_why(tmp_path, caplog):
    rows, _ = show(tmp_path, Gen(fail=True), Filter())
    assert played(rows) == [] and rows[-1]["status"] == "finished"
    cands = [r for r in rows if r["event"] == "candidates"]
    assert len(cands) == 15  # every joke of every bit has a row (invariant 6)
    assert all(r["items"][0]["verdict"] == "reject" and "generation failed: LLMError: http 503" in r["items"][0]["reason"]
               for r in cands)
    assert "no fallback" in caplog.text


def test_bank_fallback_lines_go_through_the_rules_and_a_rejected_one_is_skipped(tmp_path):
    flt = Filter(bank_rule=lambda t: False)  # stands in for a rule hit on the bank's line
    rows, _ = show(tmp_path, Gen(fail=True), flt, fallback=BankGenerator(), n_bits=1)
    assert played(rows) == [] and rows[-1]["status"] == "finished"
    assert all(r["items"][0]["verdict"] == "reject" and "rules: rule" in r["items"][0]["reason"]
               for r in rows if r["event"] == "candidates") and len([r for r in rows if r["event"] == "candidates"]) == 5
    assert not [c for c in flt.calls if c[0] == "check"]  # the LLM-backed check never ran on a bank line


def test_the_real_rules_pass_every_line_of_the_bank():
    bank = BankGenerator()
    for mode in bank.bank:
        for bit in range(4):
            for joke in range(5):
                text = bank.generate(mode, bit, joke, CTX).texts[0]
                assert all(safety.check_rules(text, age).ok for age in ("mixed", "teens")), text


def test_bridge_plays_between_bits_but_never_before_the_first(tmp_path):
    rows, voice = show(tmp_path, Gen(), Filter(), bridges=["bridge a", "bridge b"])
    plays = [t for kind, t in voice.events if kind == "start"]
    assert plays[0].startswith("ok 0.0") and plays.count("bridge a") + plays.count("bridge b") >= 2
    for bit in (1, 2):
        first = next(i for i, t in enumerate(plays) if t.startswith(f"ok {bit}."))
        assert plays[first - 1].startswith("bridge")


def test_a_slow_bit_gets_more_bridge_lines_up_to_the_cap(tmp_path):
    from app.show import MAX_BRIDGES
    _, voice = show(tmp_path, Gen(delay=0.2), Filter(), bridges=["b1", "b2", "b3", "b4"], n_bits=2)
    plays = [t for kind, t in voice.events if kind == "start"]
    assert sum(t.startswith("b") for t in plays) == MAX_BRIDGES


def thread_of(text):
    return int(text.rsplit(".", 1)[1])  # Gen names a line "<tag> <bit>.<joke>.<thread>"


def test_a_bit_plays_one_thread_whenever_a_thread_fully_passes(tmp_path):
    # Thread 1 is rejected at joke 2, so threads 0 and 2 are the full ones. The closer must match the opener.
    flt = Filter(rule=lambda t: not t.endswith("0.2.1") and not t.endswith("1.2.1") and not t.endswith("2.2.1"))
    rows, _ = show(tmp_path, Gen(), flt, n_bits=3)
    by_bit = {}
    for r in (r for r in rows if r["event"] == "joke_played"):
        by_bit.setdefault(r["bit"], []).append(r)
    assert len(by_bit) == 3
    for plays in by_bit.values():
        assert len({thread_of(r["text"]) for r in plays}) == 1 and thread_of(plays[0]["text"]) != 1
        assert [r["pick_prob"] for r in plays] == [0.5] * 5  # two eligible threads: 0.5 each, 1 in total


def test_thread_picks_are_uniform_over_the_eligible_threads():
    from app.show import _pick_thread
    batches = [generate.Batch("p", "x", [f"line {j}.{i}" for i in range(3)]) for j in range(5)]
    passing = [[0, 2]] * 4 + [[0, 1, 2]]  # threads 0 and 2 pass everywhere, thread 1 misses four jokes
    selector = UniformSelector(3)
    picks = [_pick_thread(batches, passing, selector) for _ in range(400)]
    assert {t for t, _ in picks} == {0, 2} and {tuple(tied) for _, tied in picks} == {(0, 2)}
    assert 140 <= sum(t == 0 for t, _ in picks) <= 260
    assert _pick_thread(batches, [[]] * 5, selector) is None  # nothing passed anywhere


def test_pick_prob_is_the_marginal_probability_of_the_line_and_sums_to_one():
    from app.show import _marginal
    # Threads A=0 and B=1 are tied; A's candidate was rejected at this joke, B's and C=2's passed.
    passing, tied = [1, 2], [0, 1]
    assert _marginal(1, passing, tied) == pytest.approx(0.75)  # 1/2 B is picked + 1/2 A is picked and B borrowed (1/2)
    assert _marginal(2, passing, tied) == pytest.approx(0.25)  # only by borrowing
    assert sum(_marginal(i, passing, tied) for i in passing) == pytest.approx(1.0)
    full = [0, 1, 2]
    assert [_marginal(i, full, [0, 2]) for i in full] == [0.5, 0.0, 0.5]


def test_the_logged_pick_prob_of_a_line_is_its_marginal_over_many_shows(tmp_path):
    # Thread 0 fails joke 2, thread 1 fails joke 3, thread 2 fails jokes 0 and 1: threads 0 and 1 tie with 4 passes.
    def rule(text):
        thread, joke = thread_of(text), int(text.rsplit(".", 2)[1])
        return not ((thread == 0 and joke == 2) or (thread == 1 and joke == 3) or (thread == 2 and joke in (0, 1)))
    seen = {}
    for seed in range(8):
        rows, _ = show(tmp_path, Gen(), Filter(rule=rule), n_bits=1, seed=seed)
        for r in (r for r in rows if r["event"] == "joke_played"):
            seen.setdefault((r["joke"], thread_of(r["text"])), set()).add(r["pick_prob"])
    # Joke 2: thread 1's line is picked with 1/2 + (1/2)(1/2); thread 2's only by borrowing, 1/4.
    assert seen[(2, 1)] == {0.75} and seen[(2, 2)] == {0.25} and (2, 0) not in seen
    assert seen[(0, 0)] == {0.5} and seen[(0, 1)] == {0.5}  # tied threads, no borrowing at joke 0: 1/2 each


def test_without_a_full_thread_the_best_partial_one_is_filled_from_other_threads_then_the_bank(tmp_path):
    def rule(text):
        thread, joke = thread_of(text), int(text.rsplit(".", 2)[1])
        return not ((thread == 0 and joke == 1) or (thread == 1 and joke == 3) or joke == 4 or thread == 2)
    # thread 0 fails joke 1, thread 1 fails joke 3, thread 2 fails everything, and joke 4 fails in every thread.
    rows, _ = show(tmp_path, Gen(), Filter(rule=rule), fallback=BankGenerator(), n_bits=1)
    plays = [r for r in rows if r["event"] == "joke_played"]
    assert len(plays) == 5
    chosen = thread_of(plays[0]["text"])
    assert chosen in (0, 1)
    other = 1 - chosen
    failed = 1 if chosen == 0 else 3
    assert thread_of(plays[failed]["text"]) == other  # the rejected slot borrows the other thread's passing line
    assert not plays[4]["text"].startswith("ok")  # no thread passes joke 4: the bank's line


def test_the_first_line_is_rendered_before_the_bridge_hands_over(tmp_path):
    voice = FakeVoice(delay=0.5)
    gen = Gen()
    log = SessionLog(tmp_path)
    asyncio.run(run_show(load_director([], seed=1), "sim", CTX, generator=gen, safety=Filter(),
                         selector=UniformSelector(0), tts=voice, player=FakePlayer(voice, seconds=0.2),
                         source=SimRoom(CTX, 0), clock=RealClock(), log=log, n_bits=2, config=FAST,
                         bridge_lines=["b1", "b2"]))
    last_bridge_end = max(t for (kind, text), t in voice.times.items() if kind == "end" and text.startswith("b"))
    first_line_start = voice.times[("start", next(t for k, t in voice.events if k == "start" and t.startswith("ok 1."))) ]
    assert first_line_start - last_bridge_end < 0.15  # a 0.5 s render must not show up as dead air


def test_bridge_lines_are_not_repeated_while_unused_ones_remain(tmp_path):
    from app import show as show_module
    lines = [f"b{i}" for i in range(2 * show_module.MAX_BRIDGES + 4)]  # more than two waits can use
    _, voice = show(tmp_path, Gen(delay=0.2), Filter(), bridges=lines, n_bits=3)
    bridges = [t for kind, t in voice.events if kind == "start" and t.startswith("b")]
    assert len(bridges) > 2 and len(set(bridges)) == len(bridges) and bridges == lines[:len(bridges)]


def test_a_bridge_line_that_fails_to_render_never_stops_the_show(tmp_path):
    class BadBridgeVoice(FakeVoice):
        def synthesize(self, text):
            if text.startswith("b"):
                raise RuntimeError("tts down")
            return super().synthesize(text)
    rows, voice = show(tmp_path, Gen(delay=0.1), Filter(), voice=BadBridgeVoice(), bridges=["b1", "b2"], n_bits=3)
    assert rows[-1]["status"] == "finished" and len(played(rows)) == 15
    assert not [e for e in voice.events if e[0] == "start" and e[1].startswith("b")]


def test_two_concurrent_shows_sharing_one_engine_synthesize_one_line_at_a_time(tmp_path):
    from app.tts import SerialTTS

    class Engine:
        active = peak = 0

        def __init__(self):
            self.guard = threading.Lock()

        def synthesize(self, text):
            with self.guard:
                Engine.active += 1
                Engine.peak = max(Engine.peak, Engine.active)
            time.sleep(0.01)
            with self.guard:
                Engine.active -= 1
            return to_wav(np.zeros(240))

    async def two_shows(engine):
        async def one(name):
            voice = FakeVoice()
            voice.synthesize = engine.synthesize  # the shared, process-wide engine
            await run_show(load_director([], seed=1), "sim", CTX, generator=Gen(), safety=Filter(),
                           selector=UniformSelector(0), tts=engine, player=FakePlayer(voice), source=SimRoom(CTX, 0),
                           clock=RealClock(), log=SessionLog(tmp_path / name), n_bits=2, config=FAST,
                           bridge_lines=["b1", "b2", "b3"])
        await asyncio.gather(one("a"), one("b"))

    asyncio.run(two_shows(Engine()))
    assert Engine.peak > 1  # without the shared lock, two shows do synthesize at once
    Engine.active = Engine.peak = 0
    asyncio.run(two_shows(SerialTTS(Engine())))
    assert Engine.peak == 1


def test_a_line_the_voice_cannot_say_is_replaced_by_another_candidate_and_logged(tmp_path):
    voice = PickyVoice(lambda t: t.endswith(".0") or t.endswith(".1"), delay=0.0)  # threads 0 and 1 are unspeakable
    rows, _ = show(tmp_path, Gen(), Filter(), voice=voice, fallback=BankGenerator(), n_bits=2)
    assert rows[-1]["status"] == "finished" and len(played(rows)) == 10
    assert all(t.endswith(".2") for t in played(rows))  # only thread 2 can be spoken, so every joke borrows it
    row = next(r for r in rows if r["event"] == "candidates")
    assert len(row["items"]) == 3 and sum("voice failed: ValueError" in i["reason"] for i in row["items"]) <= 2


def test_when_every_generated_line_fails_to_voice_the_bank_is_used_inside_the_same_row(tmp_path):
    voice = PickyVoice(lambda t: t.startswith("ok"))
    rows, _ = show(tmp_path, Gen(), Filter(), voice=voice, fallback=BankGenerator(), n_bits=1)
    assert rows[-1]["status"] == "finished" and len(played(rows)) == 5 and not any(t.startswith("ok") for t in played(rows))
    cands = [r for r in rows if r["event"] == "candidates"]
    assert len(cands) == 5
    for r in cands:
        assert [i["verdict"] for i in r["items"]] == ["pass"] * 4
        assert all("voice failed" in i["reason"] for i in r["items"][:3])
        assert r["items"][3]["reason"].startswith("hand-checked bank: voice failed (bank-v1); rules: ")


def test_a_bit_whose_first_line_cannot_be_voiced_still_starts_and_the_show_finishes(tmp_path):
    voice = PickyVoice(lambda t: t.startswith("ok 0.0.") or t.startswith("bank"))
    rows, _ = show(tmp_path, Gen(), Filter(), voice=voice, n_bits=2)  # no bank: the joke is skipped, the rest play
    assert rows[-1]["status"] == "finished"
    first = [r for r in rows if r["event"] == "candidates" and r["bit"] == 0 and r["joke"] == 0]
    assert len(first) == 1 and all("voice failed" in i["reason"] for i in first[0]["items"])
    assert len(played(rows)) == 9  # joke 0 of bit 0 had nothing speakable; every other joke played


def test_a_line_that_cannot_be_voiced_and_has_no_bank_is_skipped_without_stopping_the_show(tmp_path):
    rows, _ = show(tmp_path, Gen(), Filter(), voice=PickyVoice(lambda t: True), n_bits=1)
    assert rows[-1]["status"] == "finished" and played(rows) == []
    assert len([r for r in rows if r["event"] == "candidates"]) == 5


def test_bridge_lines_are_ten_distinct_and_the_cache_key_notices_a_change(monkeypatch):
    from sim import evaluate_live as ev
    assert len(generate.BRIDGES) >= 10 and len(set(generate.BRIDGES)) == len(generate.BRIDGES)
    before = ev.digest()
    monkeypatch.setattr(generate, "BRIDGES", generate.BRIDGES[:-1])
    assert ev.digest() != before
    monkeypatch.undo()
    from app import show as show_module
    monkeypatch.setattr(show_module, "MAX_BRIDGES", show_module.MAX_BRIDGES + 1)
    assert ev.digest() != before


def test_live_safety_rejects_a_bit_when_its_verdict_call_dies_and_the_bank_takes_over(tmp_path):
    safety.clear_cache()
    dead, _ = client((503, {}))
    rows, _ = show(tmp_path, Gen(), generate.LiveSafety(dead), fallback=BankGenerator())
    assert played(rows) and all(r["generator_id"] == "bank-v1" for r in rows if r["event"] == "candidates"
                                 and r["items"][0]["verdict"] == "pass")
    safety.clear_cache()


def test_live_safety_passes_lines_the_model_cleared_after_prefetch():
    safety.clear_cache()
    texts = ["The microwave and I have an understanding.", "My battery is on a journey."]
    ok, _ = client((200, completion(json.dumps({"verdicts": [{"id": i, "ok": True, "reason": "r"} for i in range(2)]}))))
    s = generate.LiveSafety(ok)
    assert not s.check(texts[0], "mixed").ok  # no verdict yet
    s.prefetch(texts, "mixed")
    assert all(s.check(t, "mixed").ok for t in texts)
    safety.clear_cache()


def test_stack_choice_and_the_live_room_guard(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from app import main
    monkeypatch.delenv("GENERATOR", raising=False)
    gen, _, fallback, bridges = main.make_stack("live")
    assert isinstance(gen, BankGenerator) and fallback is None and bridges is None
    monkeypatch.setenv("GENERATOR", "llm")
    monkeypatch.setattr(safety, "BLOCKLIST_LOCAL", tmp_path / "missing.txt")
    with pytest.raises(safety.SafetyNotReady):
        main.make_stack("live")
    for name in ("JOKE_WRITER_PROVIDER", "JOKE_WRITER_MODEL", "SAFETY_CHECK_PROVIDER", "SAFETY_CHECK_MODEL", "GEMINI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)  # an exported shell variable must not change what this test builds
    monkeypatch.setenv("LLM_BASE_URL", "http://x")
    monkeypatch.setenv("LLM_MODEL", "m")
    gen, filt, fallback, bridges = main.make_stack("sim")
    assert isinstance(gen, generate.ThreadGenerator) and isinstance(fallback, BankGenerator) and bridges
    assert gen.client.retries == generate.RETRIES and filt.client.retries == generate.FILTER_RETRIES


# --- Exp 20's bookkeeping ---

def joke_row(joke, start, end, bit=0, text="x"):
    return {"event": "joke_played", "bit": bit, "joke": joke, "text": text, "audio_start_at": start, "audio_end_at": end}


def bridge_row(start, end, bit=1):
    return {"event": "bridge", "bit": bit, "text": "Give me a second.", "audio_start_at": start, "audio_end_at": end}


def gap_row(at):
    return {"event": "gap", "at": at}


def test_silences_measure_from_the_later_of_audio_end_and_hold_end():
    from sim import evaluate_live as ev
    rows = [joke_row(0, 10.0, 11.0), gap_row(12.0),                # first line: no silence before it
            joke_row(1, 12.5, 13.0), gap_row(20.0),                # hold ended 12.0, audio 11.0: 0.5 s
            bridge_row(20.25, 24.0),                               # hold ended 20.0
            joke_row(0, 25.0, 26.0, bit=1)]                        # after the bridge, which ended 24.0
    out = ev.silences(rows)
    assert out["within"] == [pytest.approx(0.5)] and out["between"] == [pytest.approx(0.25), pytest.approx(1.0)]


def test_a_hold_whose_timestamp_equals_the_next_start_is_not_counted_as_silence():
    from sim import evaluate_live as ev
    # A coarse clock stamps the gap row and the next play with the same instant: the gap is 0, not the hold length.
    rows = [joke_row(0, 10.0, 11.0), gap_row(12.0), joke_row(1, 12.0, 13.0), gap_row(20.0), joke_row(0, 20.0, 21.0, bit=1)]
    out = ev.silences(rows)
    assert out["within"] == [0.0] and out["between"] == [0.0]
    # and a clock that runs the other way round (the next play stamped before the row) cannot make it negative
    rows[2]["audio_start_at"] = 11.9
    assert ev.silences(rows)["within"] == [0.0]


def test_a_repeated_line_is_paired_with_its_own_hold_by_position_not_by_text():
    from sim import evaluate_live as ev
    rows = [joke_row(0, 10.0, 11.0, text="same"), gap_row(12.0), joke_row(1, 12.5, 13.0, text="x"), gap_row(20.0),
            joke_row(2, 20.5, 21.0, text="same"), gap_row(31.0), joke_row(3, 31.5, 32.0, text="y")]
    # Keyed by text, the third play would take the hold of the first "same" (12.0) or the last one (31.0).
    assert ev.silences(rows)["within"] == [pytest.approx(0.5)] * 3


def test_every_bridge_line_a_show_plays_is_a_bridge_row_in_play_order(tmp_path):
    rows, voice = show(tmp_path, Gen(delay=0.2), Filter(), bridges=[f"b{i}" for i in range(6)], n_bits=3)
    bridge_rows = [r for r in rows if r["event"] == "bridge"]
    played_bridges = [t for kind, t in voice.events if kind == "start" and t.startswith("b")]
    assert [r["text"] for r in bridge_rows] == played_bridges and len(bridge_rows) >= 2
    for r in bridge_rows:
        assert r["bit"] in (1, 2) and r["audio_end_at"] > r["audio_start_at"] and r["tts_seconds"] >= 0
    # a bridge row sits between the previous bit's last gap and its own bit's first joke_played
    events = [(r["event"], r.get("bit")) for r in rows if r["event"] in ("bridge", "joke_played", "gap")]
    for r in bridge_rows:
        at = rows.index(r)
        assert rows[at - 1]["event"] in ("gap", "bridge", "bit_end", "bit_decision")
        assert next(x for x in rows[at:] if x["event"] == "joke_played")["bit"] == r["bit"]
    assert events  # the log is non-empty


def test_a_slow_dripping_reply_is_cut_off_at_the_attempt_timeout():
    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(20):
                time.sleep(0.1)  # every chunk arrives in time, so no httpx phase timeout ever fires
                yield b" "

    c = llm.Client("http://fake", "k", "m", timeout=0.5, transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=Drip())))
    start = time.monotonic()
    with pytest.raises(llm.LLMError, match="deadline"):
        c.chat("s", "u")
    assert time.monotonic() - start < 1.0  # about 0.6 s; read to the end it would take 2 s


def test_peak_requests_per_minute_counts_the_busiest_window():
    from sim import evaluate_live as ev
    assert ev.peak_per_minute([0, 10, 59, 61, 200]) == 3 and ev.peak_per_minute([]) == 0


def test_bank_reason_is_read_from_the_bank_item_format():
    from sim import evaluate_live as ev
    assert ev.bank_reason("hand-checked bank: every candidate rejected (bank-v1); rules: ok") == "every candidate rejected"
    why = "generation failed: GenerationError: x: y (z)"
    assert ev.bank_reason(f"hand-checked bank: {why} (bank-v1); rules: ok") == why


def test_bank_plays_counts_the_jokes_that_came_from_the_bank():
    from sim import evaluate_live as ev
    rows = [{"event": "candidates", "bit": 0, "joke": 0, "items": [
                {"reason": "ok"}, {"reason": "hand-checked bank: voice failed (bank-v1); rules: ok"}]},
            {"event": "candidates", "bit": 0, "joke": 1, "items": [{"reason": "ok"}]},
            {"event": "joke_played", "bit": 0, "joke": 0, "chosen": 1}, {"event": "joke_played", "bit": 0, "joke": 1, "chosen": 0}]
    assert ev.bank_plays(rows) == ["voice failed"]


def test_bridge_lines_cover_the_worst_wait_for_a_bit():
    # A bit prepares after the previous one ends: generation, then the filter, back to back.
    from app import show
    worst = generate.GENERATOR_TIMING[1] + generate.FILTER_TIMING[1]
    assert show.MAX_BRIDGES * 5.0 >= worst  # about 5 s per bridge line


# --- the local writer (D-107, D-109) ---

OLLAMA_ENV = ("JOKE_WRITER_PROVIDER", "JOKE_WRITER_MODEL", "OLLAMA_URL", "LLM_MODEL")


def test_the_ollama_writer_runs_with_thinking_off_the_schema_enforced_and_no_key(monkeypatch):
    for name in OLLAMA_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JOKE_WRITER_PROVIDER", "ollama")
    c = llm.from_env(role="joke_writer")
    assert (c.base_url, c.api_key, c.model) == ("http://127.0.0.1:11434/v1", "", "bonsai-2:4b")
    assert c.extra_body == llm.OLLAMA_NO_THINK and c.json_schema is True and c.local is True
    monkeypatch.setenv("OLLAMA_URL", "http://box:1/v1")
    monkeypatch.setenv("JOKE_WRITER_MODEL", "qwen3:4b-instruct")
    c = llm.from_env(role="joke_writer")
    assert (c.base_url, c.model) == ("http://box:1/v1", "qwen3:4b-instruct")
    monkeypatch.setenv("JOKE_WRITER_MODEL", "claude-haiku-5-5")
    with pytest.raises(RuntimeError, match="Claude model but the provider is ollama"):
        llm.from_env(role="joke_writer")
    assert llm.Client("http://g", "k", "m").local is False  # Gemini and Groq are never warmed


def test_the_shows_writer_is_the_client_exp_24_measured(monkeypatch):
    """The show must run bonsai the way Exp 24 timed it: same endpoint, timeout, retries, deadline and request fields."""
    from sim import compare_generators as cg
    for name in OLLAMA_ENV + ("BONSAI_OLLAMA_TAG", "SAFETY_CHECK_PROVIDER", "SAFETY_CHECK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JOKE_WRITER_PROVIDER", "ollama")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://g/v1")  # the filter's client is built too; it is not under test
    monkeypatch.setenv("SAFETY_CHECK_MODEL", "flash-lite")
    show = generate.llm_stack("sim")[0].client
    measured = cg.make_client(cg.MODELS["bonsai-2-4b"])
    fields = ("base_url", "api_key", "model", "timeout", "retries", "deadline_s", "extra_body", "json_schema")
    assert {f: getattr(show, f) for f in fields} == {f: getattr(measured, f) for f in fields}


def local_writer(*script):
    transport, seen = server(*script)
    c = llm.Client("http://local/v1", "", "bonsai-2:4b", 10.0, transport=transport, retries=2, deadline_s=12.0,
                   local=True, extra_body=llm.OLLAMA_NO_THINK, json_schema=True)
    return generate.ThreadGenerator(c), seen


def ps(*models, context=8192):
    """A fake `/api/ps`: (name, size, size_vram) per loaded model."""
    asked = []

    def http(url):
        asked.append(url)
        return {"models": [{"name": n, "model": n, "size": s, "size_vram": v, "context_length": context}
                           for n, s, v in models]}
    return http, asked


ON_GPU = ps(("bonsai-2:4b", 1_940_000_000, 1_940_000_000))[0]


def test_warm_up_loads_a_local_writer_once_without_touching_the_shows_client():
    gen, seen = local_writer((200, completion('{"ready": true}')))
    http, asked = ps(("bonsai-2:4b", 1_940_000_000, 1_940_000_000))
    assert isinstance(generate.warm_up(gen, http), float) and len(seen) == 1
    assert asked == ["http://local/api/ps"]
    assert seen[0]["model"] == "bonsai-2:4b" and seen[0]["reasoning_effort"] == "none"
    assert gen.client.attempts == 0 and gen.client.timeout == 10.0  # a copy carried the long timeout
    assert generate.warm_up(generate.ThreadGenerator(llm.Client("http://g", "k", "m"))) is None
    assert generate.warm_up(BankGenerator()) is None


def test_a_writer_that_cannot_load_is_an_error_that_says_how_to_fix_it():
    gen, seen = local_writer(httpx.ConnectError("refused"))
    with pytest.raises(llm.LLMError, match="ollama serve.*ollama pull bonsai-2:4b"):
        generate.warm_up(gen, ON_GPU)
    assert len(seen) == 1  # no retries: a down server is not a busy one


@pytest.mark.parametrize("loaded, says", [
    ((("gemma4:12b", 7_000_000_000, 6_000_000_000), ("bonsai-2:4b", 1_940_000_000, 1_000_000_000)),
     "only 52% of it is on the GPU; also loaded: gemma4:12b"),  # 2026-11-25: gemma held the GPU, every bit timed out
    ((("gemma4:12b", 7_000_000_000, 7_000_000_000),), "Ollama does not list it; also loaded: gemma4:12b"),
    ((("bonsai-2:4b", 0, 0),), "Ollama does not list it"),
    ((("bonsai-2:4b", 1_000, 2_000),), "Ollama does not list it"),  # more in VRAM than its size: a bad report
])
def test_a_writer_not_all_on_the_gpu_stops_the_show_and_names_what_holds_it(loaded, says):
    gen, _ = local_writer((200, completion('{"ready": true}')))
    with pytest.raises(llm.LLMError, match=re.escape(says)):
        generate.warm_up(gen, ps(*loaded)[0])


def test_an_unreadable_gpu_report_counts_as_not_on_the_gpu():
    gen, _ = local_writer((200, completion('{"ready": true}')))

    def down(url):
        raise OSError("refused")
    with pytest.raises(llm.LLMError, match="does not list it"):
        generate.warm_up(gen, down)


def test_the_shipped_sample_env_writes_with_bonsai_on_ollama(monkeypatch):
    sample = pathlib.Path(__file__).resolve().parents[1] / "backend" / "sample.env"
    values = dict(line.split("=", 1) for line in sample.read_text(encoding="utf-8").splitlines()
                  if line and not line.startswith("#"))
    for name in OLLAMA_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("JOKE_WRITER_PROVIDER", "JOKE_WRITER_MODEL"):
        monkeypatch.setenv(name, values[name])
    c = llm.from_env(role="joke_writer")
    assert c.local and c.model == "bonsai-2:4b"


def test_a_writer_named_without_a_tag_is_found_as_ollama_lists_it():
    gen, _ = local_writer((200, completion('{"ready": true}')))
    gen.client.model = "bonsai-2"  # Ollama answers to it and lists it as bonsai-2:latest
    assert isinstance(generate.warm_up(gen, ps(("bonsai-2:latest", 10, 10))[0]), float)


def test_a_writer_loaded_at_another_context_than_exp_24_measured_stops_the_show():
    from sim import compare_generators as cg
    assert generate.WRITER_CONTEXT == cg.GATE_CONTEXT
    gen, _ = local_writer((200, completion('{"ready": true}')))
    with pytest.raises(llm.LLMError, match="context 4096, not 8192.*OLLAMA_CONTEXT_LENGTH=8192"):
        generate.warm_up(gen, ps(("bonsai-2:4b", 10, 10), context=4096)[0])
