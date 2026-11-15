"""The generator, the retries and the show's render-ahead, bridge and bank fallback. A fake HTTP server and fake voice;
no real API or Kokoro. The timing numbers live in notebooks/decisions-live.ipynb (Exp 20)."""
import asyncio
import json
import pathlib
import sys
import threading
import time

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import generate, llm, safety  # noqa: E402
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
    return json.dumps({"premises": ["p"], "threads": [[f"{tag} line {j}.{i}" for j in range(n_jokes)] for i in range(k)]})


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


# --- the generator ---

def test_one_call_per_bit_serves_all_five_jokes_with_three_candidates_each():
    c, seen = client((200, completion(reply())))
    g = generate.GeminiGenerator(c)
    batches = [g.generate("cynic", 0, j, CTX) for j in range(5)]
    assert len(seen) == 1
    assert [len(b.texts) for b in batches] == [3] * 5 and batches[2].texts[1] == "x line 2.1"
    assert seen[0]["temperature"] == generate.TEMPERATURE and seen[0]["response_format"] == {"type": "json_object"}
    assert batches[0].prompt.startswith(seen[0]["messages"][0]["content"])  # the log holds what the model saw


def test_a_second_bit_is_a_second_call_and_remembers_the_first_premise():
    c, seen = client((200, completion(reply())))
    g = generate.GeminiGenerator(c)
    g.generate("cynic", 0, 0, CTX)
    g.generate("optimist", 1, 0, CTX)
    assert len(seen) == 2 and "p" in json.loads(seen[1]["messages"][1]["content"])["premises_already_told"]


@pytest.mark.parametrize("bad", [JOKE_MAJOR,  # the old shape is rejected, not guessed at
                                 json.dumps({"threads": ["Line one.", "Line two.", "Line three."]}),  # string rows
                                 json.dumps({"threads": "xyz"}), json.dumps({"threads": [[1] * 5] * 3}),
                                 "not json", reply(n_jokes=4), reply(n_jokes=6), reply(k=0), reply(k=2), reply(k=4),
                                 json.dumps({"threads": [["a"] * 5, ["b"] * 5, ["", "b", "b", "b", "b"]]}),
                                 json.dumps([1, 2])])
def test_a_malformed_reply_raises_for_every_joke_without_a_second_call(bad):
    c, seen = client((200, completion(bad)))
    g = generate.GeminiGenerator(c)
    for joke in range(5):
        with pytest.raises(generate.GenerationError):
            g.generate("cynic", 0, joke, CTX)
    assert len(seen) == 1


def test_a_timeout_raises_for_every_joke_without_a_second_call():
    c, seen = client(httpx.ConnectTimeout("down"))
    g = generate.GeminiGenerator(c)
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
    premise, lines = generate.parse(fence + "json" + chr(10) + reply() + chr(10) + fence)
    assert premise == "p" and [len(r) for r in lines] == [3] * 5
    assert lines[4] == ["x line 4.0", "x line 4.1", "x line 4.2"] and lines[0][2] == "x line 0.2"


def test_the_prompt_holds_the_identity_the_rules_and_teens_only_for_teens():
    c, seen = client((200, completion(reply())))
    g = generate.GeminiGenerator(c)
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


# --- the show around it ---

class FakeVoice:
    def __init__(self, delay=0.0):
        self.delay, self.events, self.lock, self.times = delay, [], threading.Lock(), {}

    def synthesize(self, text):
        with self.lock:
            self.events.append(("synth", text))
        time.sleep(self.delay)
        return to_wav(np.zeros(240))


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
    def __init__(self, tag="ok", fail=False, delay=0.0):
        self.generator_id, self.tag, self.fail, self.delay, self.calls = f"gen-{tag}", tag, fail, delay, 0

    def generate(self, mode, bit, joke, context):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            raise llm.LLMError("http 503", 503)
        return generate.Batch("p", f"prompt {bit}", [f"{self.tag} {bit}.{joke}.{i}" for i in range(3)])


def show(tmp_path, gen, flt, voice=None, fallback=None, bridges=None, n_bits=3):
    voice = voice or FakeVoice()
    log = SessionLog(tmp_path)
    asyncio.run(run_show(load_director([], seed=1), "sim", CTX, generator=gen, safety=flt,
                         selector=UniformSelector(0), tts=voice, player=FakePlayer(voice), source=SimRoom(CTX, 0),
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
    assert {r["generator_id"] for r in cands} == {bank.generator_id}
    assert all("generation failed: LLMError: http 503" in i["reason"] for r in cands for i in r["items"])


def test_all_candidates_rejected_falls_back_to_the_bank_for_that_joke_only(tmp_path):
    flt = Filter(rule=lambda t: not t.startswith("ok 0.2"))
    rows, _ = show(tmp_path, Gen(), flt, fallback=BankGenerator())
    by_joke = {(r["bit"], r["joke"]): r for r in rows if r["event"] == "candidates"}
    assert [r["generator_id"] for r in rows if r["event"] == "candidates" and r["bit"] == 0 and r["joke"] == 2][-2:] == [
        "gen-ok", "bank-v1"]
    assert by_joke[(0, 2)]["items"][0]["reason"].startswith("bank fallback (every candidate rejected)")
    assert all(r["generator_id"] == "gen-ok" for k, r in by_joke.items() if k != (0, 2))
    assert len(played(rows)) == 15


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
               for r in rows if r["event"] == "candidates")
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


def test_thread_picks_are_uniform_over_the_eligible_threads_and_probabilities_sum_to_one():
    from app.show import _pick_thread
    batches = [generate.Batch("p", "x", [f"line {j}.{i}" for i in range(3)]) for j in range(5)]
    passing = [[0, 2]] * 4 + [[0, 1, 2]]  # threads 0 and 2 pass everywhere, thread 1 misses four jokes
    selector = UniformSelector(3)
    picks = [_pick_thread(batches, passing, selector) for _ in range(400)]
    assert {t for t, _ in picks} == {0, 2} and {p for _, p in picks} == {0.5}
    assert 140 <= sum(t == 0 for t, _ in picks) <= 260
    assert _pick_thread(batches, [[]] * 5, selector) is None  # nothing passed anywhere


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
    lines = [f"b{i}" for i in range(8)]
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


def test_only_one_thread_synthesizes_at_a_time(tmp_path):
    class Counting(FakeVoice):
        active = peak = 0

        def synthesize(self, text):
            with self.lock:
                Counting.active += 1
                Counting.peak = max(Counting.peak, Counting.active)
            time.sleep(0.02)
            with self.lock:
                Counting.active -= 1
            return to_wav(np.zeros(240))
    show(tmp_path, Gen(), Filter(), voice=Counting(), bridges=["b1", "b2", "b3"], n_bits=3)
    assert Counting.peak == 1


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
    monkeypatch.setenv("LLM_BASE_URL", "http://x")
    monkeypatch.setenv("LLM_MODEL", "m")
    gen, filt, fallback, bridges = main.make_stack("sim")
    assert isinstance(gen, generate.GeminiGenerator) and isinstance(fallback, BankGenerator) and bridges
    assert gen.client.retries == filt.client.retries == 2


# --- Exp 20's bookkeeping ---

ROWS = [{"event": "joke_played", "text": "a", "bit": 0, "joke": 0}, {"event": "gap", "at": 12.0},
        {"event": "joke_played", "text": "b", "bit": 0, "joke": 1}, {"event": "gap", "at": 20.0},
        {"event": "joke_played", "text": "c", "bit": 1, "joke": 0}]


def test_silences_measure_from_the_later_of_audio_end_and_hold_end():
    from sim import evaluate_live as ev
    plays = [{"text": "a", "start": 10.0, "end": 11.0},   # first line: no silence before it
             {"text": "b", "start": 12.5, "end": 13.0},   # hold ended 12.0, audio 11.0: 0.5 s
             {"text": generate.BRIDGES[0], "start": 20.25, "end": 24.0},  # hold ended 20.0
             {"text": "c", "start": 25.0, "end": 26.0}]   # after the bridge, which ended 24.0
    out = ev.silences(ROWS, plays)
    assert out["within"] == [pytest.approx(0.5)] and out["between"] == [pytest.approx(0.25), pytest.approx(1.0)]


def test_a_hold_whose_timestamp_equals_the_next_start_is_not_counted_as_silence():
    from sim import evaluate_live as ev
    # A coarse clock stamps the gap row and the next play with the same instant: the gap is 0, not the hold length.
    plays = [{"text": "a", "start": 10.0, "end": 11.0}, {"text": "b", "start": 12.0, "end": 13.0},
             {"text": "c", "start": 20.0, "end": 21.0}]
    out = ev.silences(ROWS, plays)
    assert out["within"] == [0.0] and out["between"] == [0.0]
    # and a clock that runs the other way round (the next play stamped before the row) cannot make it negative
    plays[1]["start"] = 11.9
    assert ev.silences(ROWS, plays)["within"] == [0.0]


def test_peak_requests_per_minute_counts_the_busiest_window():
    from sim import evaluate_live as ev
    assert ev.peak_per_minute([0, 10, 59, 61, 200]) == 3 and ev.peak_per_minute([]) == 0
