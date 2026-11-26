"""Where a bit's wait goes (D-113): the `prep` row's stage timings and the first joke's bridge overhang.
Stage tests run on a fake clock that each fake stage advances, so a slow stage must show up in its own field."""
import asyncio
import json
import pathlib
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.log import SessionLog, read_session  # noqa: E402
from app.rooms import SimRoom  # noqa: E402
from sim import cache  # noqa: E402
from sim import stage_timing as st  # noqa: E402
from app.show import Batch, UniformSelector, run_show  # noqa: E402
from app.tts import SerialTTS, to_wav  # noqa: E402
from tests.test_cached_bits import pool  # noqa: E402
from tests.test_generate import CTX as REAL_CTX  # noqa: E402
from tests.test_generate import FAST, FakePlayer as RealPlayer, FakeVoice, Filter, Gen  # noqa: E402
from tests.test_headless_show import CTX, FakeClock, FakePlayer, director  # noqa: E402

SIM_RESULT = {"started_at", "generate_s", "safety_s", "bank_s", "render_wait_s", "render_queue_s", "render_s", "ready_at",
              "bank_fallbacks", "other_s", "first_render_failed"}


class SlowGenerator:
    generator_id = "slow"
    speculative_safe = True  # no state a guessed prep could disturb

    def __init__(self, clock, per_call):
        self.clock, self.per_call = clock, per_call

    def generate(self, mode, bit, joke, context):
        self.clock.t += self.per_call  # runs on a worker thread, as the real one does; the loop is awaiting it
        return Batch(f"p{bit}-{joke}", "prompt", [f"{mode} line {bit}.{joke}.{i}" for i in range(3)])


class SlowSafety:
    def __init__(self, clock, prefetch_s, check_s):
        self.clock, self.prefetch_s, self.check_s = clock, prefetch_s, check_s

    def prefetch(self, texts, age):
        self.clock.t += self.prefetch_s

    def check(self, text, age):
        self.clock.t += self.check_s
        return SimpleNamespace(ok=True, reason=None)


class SlowVoice:
    """A serial voice that says how long each call queued, as SerialTTS does."""

    def __init__(self, clock, synth_s, queued_s=None):
        self.clock, self.synth_s, self.queued_s, self.local = clock, synth_s, queued_s, threading.local()
        if queued_s is not None:
            self.last_queue_s = lambda: self.local.queued

    def synthesize(self, text):
        self.local.queued = self.queued_s
        numeric = isinstance(self.queued_s, (int, float)) and not isinstance(self.queued_s, bool) and self.queued_s > 0
        self.clock.t += self.synth_s + (self.queued_s if numeric else 0.0)
        return to_wav(np.zeros(2400))


def show(tmp_path, gen=None, safety=None, voice=None, n_bits=2, jokes=1, **kw):
    """Each stage is a function of the fake clock, so the stage can advance it."""
    clock = FakeClock()
    log = SessionLog(tmp_path)
    asyncio.run(run_show(director(), "sim", CTX, generator=(gen or (lambda c: SlowGenerator(c, 0.0)))(clock),
                         safety=(safety or (lambda c: SlowSafety(c, 0, 0)))(clock), selector=UniformSelector(0),
                         tts=(voice or (lambda c: SlowVoice(c, 0.0)))(clock), player=FakePlayer(clock),
                         source=SimRoom(CTX, 0), clock=clock, log=log, n_bits=n_bits, jokes_per_bit=jokes, **kw))
    return read_session(log.path)


def preps(rows):
    return [r for r in rows if r["event"] == "prep"]


def sums(r):
    return r["generate_s"] + r["safety_s"] + r["bank_s"] + r["other_s"] + r["render_wait_s"]


def test_every_bit_logs_one_prep_row_with_consistent_non_negative_stages(tmp_path):
    got = preps(show(tmp_path, gen=lambda c: SlowGenerator(c, 0.7), n_bits=3, jokes=2))
    assert [r["bit"] for r in got] == [0, 1, 2]
    for r in got:
        assert SIM_RESULT <= set(r)
        for key in ("generate_s", "safety_s", "bank_s", "other_s", "render_wait_s"):
            assert r[key] >= 0
        assert r["started_at"] <= r["ready_at"]
        assert sums(r) == pytest.approx(r["ready_at"] - r["started_at"])


def test_a_slow_stage_shows_up_in_its_own_field_only(tmp_path):
    (r,) = preps(show(tmp_path, gen=lambda c: SlowGenerator(c, 3.0), safety=lambda c: SlowSafety(c, 2.0, 0.5),
                      voice=lambda c: SlowVoice(c, 1.1, queued_s=0.4), n_bits=1))
    assert r["generate_s"] == pytest.approx(3.0)
    assert r["safety_s"] == pytest.approx(2.0 + 3 * 0.5)  # one prefetch, then a check per candidate
    assert r["render_wait_s"] == pytest.approx(1.5) and r["render_queue_s"] == pytest.approx(0.4)
    assert r["render_s"] == pytest.approx(1.1)
    assert r["bank_s"] == 0 and r["bank_fallbacks"] == 0
    assert r["ready_at"] - r["started_at"] == pytest.approx(3.0 + 3.5 + 1.5)


def test_generation_time_is_summed_over_every_joke_of_the_bit(tmp_path):
    (r,) = preps(show(tmp_path, gen=lambda c: SlowGenerator(c, 1.0), n_bits=1, jokes=4))
    assert r["generate_s"] == pytest.approx(4.0)


def test_a_voice_that_cannot_split_the_render_logs_the_total_and_null_for_the_split(tmp_path):
    (r,) = preps(show(tmp_path, voice=lambda c: SlowVoice(c, 0.9, queued_s=None), n_bits=1))
    assert r["render_wait_s"] == pytest.approx(0.9) and r["render_queue_s"] is None and r["render_s"] is None


def test_a_bit_with_nothing_to_say_waits_zero_for_a_render(tmp_path):
    class Reject(SlowSafety):
        def check(self, text, age):
            return SimpleNamespace(ok=False, reason="no")
    (r,) = preps(show(tmp_path, safety=lambda c: Reject(c, 0, 0), n_bits=1))
    assert r["render_wait_s"] == 0.0 and r["bank_fallbacks"] == 0 and r["render_queue_s"] is None


def test_bank_fallbacks_are_counted_and_timed(tmp_path):
    class Bank:
        generator_id = "bank"

        def __init__(self, clock):
            self.clock = clock

        def generate(self, mode, bit, joke, context):
            self.clock.t += 0.25
            return Batch("b", "bank", [f"bank {joke}"])

    class RejectsGenerated(SlowSafety):
        def check(self, text, age):
            return SimpleNamespace(ok=False, reason="no")

        def check_rules(self, text, age):
            return SimpleNamespace(ok=True, reason="ok")

    clock = FakeClock()
    log = SessionLog(tmp_path)
    asyncio.run(run_show(director(), "sim", CTX, generator=SlowGenerator(clock, 0.0), safety=RejectsGenerated(clock, 0, 0),
                         selector=UniformSelector(0), tts=SlowVoice(clock, 0.0), player=FakePlayer(clock),
                         source=SimRoom(CTX, 0), clock=clock, log=log, n_bits=1, jokes_per_bit=3, fallback=Bank(clock)))
    (r,) = preps(read_session(log.path))
    assert r["bank_fallbacks"] == 3 and r["bank_s"] == pytest.approx(0.75)
    assert sums(r) == pytest.approx(r["ready_at"] - r["started_at"])


def test_time_no_stage_covers_is_recorded_as_other_s_so_the_row_still_adds_up(tmp_path):
    clock = FakeClock()

    class SlowPick(UniformSelector):
        def pick(self, passing_texts):
            clock.t += 0.4  # picking a thread is in no stage's timer
            return super().pick(passing_texts)

    log = SessionLog(tmp_path)
    asyncio.run(run_show(director(), "sim", CTX, generator=SlowGenerator(clock, 1.0), safety=SlowSafety(clock, 0, 0),
                         selector=SlowPick(0), tts=SlowVoice(clock, 0.0), player=FakePlayer(clock), source=SimRoom(CTX, 0),
                         clock=clock, log=log, n_bits=1, jokes_per_bit=1))
    (r,) = preps(read_session(log.path))
    assert r["other_s"] > 0.3  # the gap is recorded, not lost
    assert r["generate_s"] == pytest.approx(1.0) and r["safety_s"] == 0
    assert sums(r) == pytest.approx(r["ready_at"] - r["started_at"])


def test_a_first_render_that_failed_is_flagged_and_a_good_one_is_not(tmp_path):
    class Dead(SlowVoice):
        def synthesize(self, text):
            raise ValueError("no audio")

    (bad,) = preps(show(tmp_path, voice=lambda c: Dead(c, 0.0), n_bits=1))
    assert bad["first_render_failed"] is True
    (good,) = preps(show(tmp_path, n_bits=1))
    assert good["first_render_failed"] is False


@pytest.mark.parametrize("queued", [-0.5, "x", True])
def test_a_queue_time_the_row_would_reject_is_dropped_so_the_show_never_fails(tmp_path, queued):
    (r,) = preps(show(tmp_path, voice=lambda c: SlowVoice(c, 0.2, queued_s=queued), n_bits=1))
    assert r["render_queue_s"] is None and r["render_s"] is None and r["render_wait_s"] >= 0


def test_serial_tts_reports_how_long_a_call_queued_apart_from_synthesis():
    class Engine:
        def synthesize(self, text):
            time.sleep(0.15)
            return b"x"
    tts, queued = SerialTTS(Engine()), {}

    def call(name):
        tts.synthesize(name)
        queued[name] = tts.last_queue_s()

    first = threading.Thread(target=call, args=("a",))
    first.start()
    time.sleep(0.03)  # a holds the lock for 0.15 s
    call("b")
    first.join()
    assert queued["a"] < 0.05 and queued["b"] > 0.08
    assert tts.last_queue_s() == queued["b"]  # per thread: this thread's last call, not a's


# --- the first line's wait after the bit was ready, on the real clock ---


def test_the_first_line_logs_how_long_a_bridge_made_the_room_wait_after_the_bit_was_ready(tmp_path):
    from app.director import load_director
    from app.rooms import RealClock
    voice, log = FakeVoice(), SessionLog(tmp_path)
    asyncio.run(run_show(load_director([], seed=1), "sim", REAL_CTX, generator=Gen(delay=0.1), safety=Filter(),
                         selector=UniformSelector(0), tts=voice, player=RealPlayer(voice, seconds=0.4),
                         source=SimRoom(REAL_CTX, 0), clock=RealClock(), log=log, n_bits=3, config=FAST,
                         bridge_lines=["b1", "b2", "b3", "b4"], predict=False))  # the end-of-bit path: the bridge fills the wait
    rows = read_session(log.path)
    ready = {r["bit"]: r["ready_at"] for r in preps(rows)}
    played = [r for r in rows if r["event"] == "joke_played"]
    first = {}
    for r in played:
        first.setdefault(r["bit"], r)
    carrying = [r for r in played if "bridge_overhang_s" in r]
    assert [r["bit"] for r in carrying] == [0, 1, 2] and all(r is first[r["bit"]] for r in carrying)  # once per bit
    for r in carrying:
        assert r["bridge_overhang_s"] == pytest.approx(max(0.0, r["audio_start_at"] - ready[r["bit"]]))
    # bit 0 plays no bridge, so it starts as soon as it is ready; a bridge line (0.4 s) is still going when the bit is
    # ready (about 0.5 s of generation), so the later bits wait for it
    assert first[0]["bridge_overhang_s"] < 0.1
    assert max(r["bridge_overhang_s"] for r in carrying[1:]) > 0.1
    # the overhang is what the last bridge line left: the first joke starts when the last bridge line ends
    for bit in (1, 2):
        last_bridge_end = max(r["audio_end_at"] for r in rows if r["event"] == "bridge" and r["bit"] == bit)
        assert first[bit]["audio_start_at"] >= last_bridge_end - 1e-3


# --- the analysis, on synthetic logs with known numbers ---

JOKE_TEXT = "SECRET JOKE TEXT"
LLM = "gen-v4:bonsai"
BASE = {"generate_s": 1.0, "safety_s": 0.5, "bank_s": 0.0, "other_s": 0.0, "render_queue_s": 0.0, "render_s": 0.5}


def bit(overhang=0.5, bridges=1, failed=False, **stages):
    return ({**BASE, **stages}, overhang, bridges, failed)


def write_show(directory, sid, bits, at=1000.0, generator=LLM, with_prep=True, pauses=(), room="live"):
    """bits: from `bit(...)`. A bit starts at the previous bit_end (`at` for bit 0); its wait is its stages plus its
    overhang, so the rows agree with each other. `pauses`: (start, end) seconds after `at`."""
    rows = [{"event": "show_start", "at": at, "room": room, "context": {"venue": "bar", "energy": "normal", "age": "mixed"},
             "policy": {"name": "linucb", "epsilon": 0.1, "alpha": 1.0, "prior": "uniform"}, "generator_id": generator,
             "selector_id": "u", "git_sha": "x", "operator": "o", "audio": {"consent": False, "path": None}}]
    for a, b in pauses:
        rows += [{"event": "pause", "at": at + a, "paused_at": at + a}, {"event": "resume", "at": at + b, "resumed_at": at + b}]
    t = at
    for i, (stages, overhang, bridges, failed) in enumerate(bits):
        rows += [{"event": "bridge", "at": t + 1, "bit": i, "text": JOKE_TEXT, "tts_seconds": 1.0,
                  "audio_start_at": t + 1, "audio_end_at": t + 2} for _ in range(bridges)]
        unsplit = stages.get("render_unsplit_s")
        render_wait = unsplit if unsplit is not None else stages["render_queue_s"] + stages["render_s"]
        ready = t + stages["generate_s"] + stages["safety_s"] + stages["bank_s"] + stages["other_s"] + render_wait
        if with_prep:
            rows.append({"event": "prep", "at": ready, "bit": i, "started_at": t, "ready_at": ready,
                         "generate_s": stages["generate_s"], "safety_s": stages["safety_s"], "bank_s": stages["bank_s"],
                         "other_s": stages["other_s"], "render_wait_s": render_wait,
                         "render_queue_s": None if unsplit is not None else stages["render_queue_s"],
                         "render_s": None if unsplit is not None else stages["render_s"], "bank_fallbacks": 0,
                         "first_render_failed": failed})
        start = ready + overhang
        rows.append({"event": "joke_played", "at": start, "bit": i, "joke": 0, "chosen": 0, "pick_prob": 0.3, "text": JOKE_TEXT,
                     "tts_seconds": 0.1, "audio_seconds": 1.0, "audio_start_at": start, "audio_end_at": start + 1,
                     **({"bridge_overhang_s": overhang} if with_prep else {})})
        t = start + 20
        rows.append({"event": "bit_end", "at": t, "bit": i, "n_labels": 5, "reward_version": 1, "reward": 0.5})
    rows.append({"event": "show_end", "at": t, "status": "finished"})
    with (directory / f"{sid}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"schema": 2, "session": sid, **r}) + chr(10))


def dominated(stage, n=9, size=12.0):
    """Bit 0 (not counted) then n bits where `stage` is `size` s and every other stage is small."""
    def one():
        return bit(overhang=size if stage == "bridge_overhang_s" else 0.5, **({} if stage == "bridge_overhang_s" else {stage: size}))
    return [one() for _ in range(n + 1)]


def two_shows(tmp_path, bits_a, bits_b=None, **kw):
    write_show(tmp_path, "aaaaaaaaaaaa", bits_a, **kw)
    write_show(tmp_path, "bbbbbbbbbbbb", bits_b or bits_a, at=5000.0, **kw)
    return st.run(None, tmp_path)


def clear(path):
    for f in path.glob("*.jsonl"):
        f.unlink()


def test_a_bit_waits_from_the_previous_bit_end_and_the_stats_are_nearest_rank(tmp_path):
    write_show(tmp_path, "aaaaaaaaaaaa", [bit(0.0, 1), bit(1.5, 2, generate_s=2.0), bit(0.0, 0, generate_s=3.0), bit(0.0, 0)])
    (res,) = st.run(None, tmp_path)["shows"]
    assert res["bits"] == 4 and res["bits_counted"] == 3  # bit 0 does not count
    # waits of bits 1 to 3: 4.5, 4.0 and 2.0 (generate + safety + render + overhang)
    assert res["wait_s"]["p50"] == 4.0 and res["wait_s"]["p95"] == 4.5 and res["wait_s"]["n"] == 3
    assert res["bridges"]["p50"] == 0 and res["bridge_overhang_s"]["p95"] == 1.5 and res["generate_s"]["p50"] == 2.0


@pytest.mark.parametrize("stage,outcome,fix", [
    ("bridge_overhang_s", "fix", "(a)"), ("render_queue_s", "fix", "(b)"), ("safety_s", "fix", "(c)"),
    ("generate_s", "no_candidate_fix", None), ("render_s", "no_candidate_fix", None),
])
def test_the_biggest_stage_picks_its_fix_and_the_writer_or_voice_has_none(tmp_path, stage, outcome, fix):
    d = two_shows(tmp_path, dominated(stage))["decision"]
    assert d["outcome"] == outcome and d["stage"] == stage and d["ci"][0] > 0
    assert (fix is None and d["fix"] is None) or d["fix"].startswith(fix)


def test_a_stage_that_is_big_on_some_bits_and_zero_on_most_is_not_missed_by_a_median(tmp_path):
    # queue 8 s on 3 bits of 8: its median is 0, its total is 24 s of 56. Overhang is a steady 2 s (total 16).
    bits = [bit()] + [bit(2.0, render_queue_s=8.0)] * 3 + [bit(2.0)] * 5
    res = two_shows(tmp_path, bits)
    assert res["pooled"]["render_queue_s"]["p50"] == 0.0 and res["pooled"]["bridge_overhang_s"]["p50"] == 2.0  # a median picks overhang
    assert res["pooled"]["shares"]["render_queue_s"] == pytest.approx(24 / 56)
    assert res["decision"]["stage"] == "render_queue_s" and res["decision"]["outcome"] == "fix"


def test_shares_are_totals_over_the_total_wait_and_add_to_one(tmp_path):
    shares = two_shows(tmp_path, dominated("safety_s"))["pooled"]["shares"]
    assert sum(shares.values()) == pytest.approx(1.0)  # a bit starts at the previous bit end, so no time is left over
    assert shares["safety_s"] == pytest.approx(12.0 / (1.0 + 12.0 + 0.5 + 0.5))


def test_shows_that_disagree_give_an_interval_across_zero_and_no_decision(tmp_path):
    d = two_shows(tmp_path, dominated("safety_s"), dominated("bridge_overhang_s"))["decision"]
    assert d["outcome"] == "inconclusive" and d["ci"][0] < 0 < d["ci"][1]


def test_a_small_share_or_too_little_data_decides_nothing(tmp_path):
    flat = [bit(overhang=2.0, generate_s=2.0, safety_s=2.0, render_queue_s=2.0, render_s=2.0)] * 10
    assert two_shows(tmp_path, flat)["decision"]["outcome"] == "no_one_stage"  # no stage reaches a quarter of the wait
    clear(tmp_path)
    assert two_shows(tmp_path, dominated("safety_s", n=7))["decision"]["outcome"] == "not_enough_data"  # 14 counted bits < 16
    clear(tmp_path)
    write_show(tmp_path, "aaaaaaaaaaaa", dominated("safety_s", n=20))
    assert st.run(None, tmp_path)["decision"]["outcome"] == "not_enough_data"  # one show: no between-show interval


def test_bank_shows_are_listed_but_never_counted(tmp_path):
    write_show(tmp_path, "aaaaaaaaaaaa", dominated("safety_s"))
    write_show(tmp_path, "bbbbbbbbbbbb", dominated("generate_s"), at=5000.0, generator="bank-v1")
    res = st.run(None, tmp_path)
    assert {s["session"]: s["included"] for s in res["shows"]} == {"aaaaaaaaaaaa": True, "bbbbbbbbbbbb": False}
    assert {s["session"]: s["generator_id"] for s in res["shows"]}["bbbbbbbbbbbb"] == "bank-v1"
    assert res["pooled"]["shows_counted"] == 1 and res["decision"]["outcome"] == "not_enough_data"
    assert "(not counted)" in st.table(res) and "bank-v1" in st.table(res)


def test_bit_zero_is_reported_apart_and_does_not_decide(tmp_path):
    bits = dominated("safety_s")
    bits[0] = bit(generate_s=30.0)  # a cold model
    res = two_shows(tmp_path, bits)
    assert res["bit0"]["bits"] == 2 and res["bit0"]["generate_s"]["p50"] == 30.0
    assert res["pooled"]["generate_s"]["p95"] == 1.0 and res["decision"]["stage"] == "safety_s"


def test_a_bit_whose_wait_holds_a_pause_or_a_failed_first_render_is_dropped_and_counted(tmp_path):
    bits = dominated("safety_s")
    bits[2] = bit(failed=True)
    # bit 3 waits from 90.5 s to 104.5 s after the show starts; the pause (95 to 97 s) is inside it
    write_show(tmp_path, "aaaaaaaaaaaa", bits, pauses=[(95.0, 97.0)])
    (res,) = st.run(None, tmp_path)["shows"]
    assert res["dropped_render_failed"] == 1 and res["dropped_paused"] == 1 and res["bits_counted"] == 7


def test_a_voice_that_cannot_split_the_render_counts_all_of_it_as_render_not_nothing(tmp_path):
    bits = [bit()] + [bit(render_queue_s=0.0, render_s=0.0, render_unsplit_s=9.0)] * 9
    res = two_shows(tmp_path, bits)
    assert res["pooled"]["render_queue_s"]["n"] == 0 and res["pooled"]["render_s"]["p50"] == 9.0
    assert res["decision"]["stage"] == "render_s" and res["decision"]["outcome"] == "no_candidate_fix"


def test_a_log_from_before_prep_is_listed_but_does_not_count(tmp_path):
    write_show(tmp_path, "aaaaaaaaaaaa", dominated("safety_s", n=20), with_prep=False)
    res = st.run(None, tmp_path)
    s = res["shows"][0]
    assert s["bits"] == 21 and s["bits_with_prep"] == 0 and s["safety_s"]["p50"] is None and not s["included"]
    assert res["pooled"]["bits_counted"] == 0 and res["decision"]["outcome"] == "not_enough_data"


def test_named_sessions_only_and_the_record_holds_seconds_and_ids_never_text(tmp_path, monkeypatch):
    logs, metrics = tmp_path / "logs", tmp_path / "metrics"
    logs.mkdir()
    monkeypatch.setattr(cache, "METRICS", metrics)
    write_show(logs, "aaaaaaaaaaaa", dominated("safety_s"))
    write_show(logs, "bbbbbbbbbbbb", dominated("safety_s"), at=5000.0)
    res = st.record(["aaaaaaaaaaaa"], logs)
    assert [s["session"] for s in res["shows"]] == ["aaaaaaaaaaaa"] and res["pooled"]["bits_counted"] == 9
    saved = (metrics / "stage_timing.json").read_text(encoding="utf-8")
    assert JOKE_TEXT not in saved and "aaaaaaaaaaaa" in saved
    assert st.stored() == json.loads(saved)["results"]
    assert "Biggest stage" in st.verdict(res) and "| aaaaaaaaaaaa" in st.table(res)


def test_a_bit_prepared_on_a_guess_is_marked_and_kept_out_of_the_stage_shares_but_not_the_wait_columns(tmp_path):
    for seed in range(8):  # a right guess is likely, not certain, in any one show
        clock = FakeClock()
        log = SessionLog(tmp_path)
        asyncio.run(run_show(director(seed=seed), "sim", CTX, generator=SlowGenerator(clock, 0.0), safety=SlowSafety(clock, 0, 0),
                             selector=UniformSelector(0), tts=SlowVoice(clock, 0.0), player=FakePlayer(clock),
                             source=SimRoom(CTX, seed), clock=clock, log=log, n_bits=6, jokes_per_bit=5))
        rows = read_session(log.path)
        spec = {r["bit"] for r in preps(rows) if r.get("speculative")}
        if spec:
            break
    guessed = {r["bit"] for r in rows if r["event"] == "prediction"}
    assert spec and spec <= guessed  # a right guess marks its prep row; a wrong one leaves no speculative row
    bits = st.bits_of(rows)
    assert {b["bit"] for b in bits if b["speculative"]} == spec
    assert not any(b["speculative"] for b in st.counted(bits))
    report = st.show_report("s", rows)
    assert report["speculative_bits"] == len(spec)
    assert report["wait_s"]["n"] == len([b for b in bits if b["bit"] >= 1])  # the wait columns still hold them


def test_the_report_counts_guesses_hits_by_whether_the_copy_learned_and_wasted_prefetches():
    def row(hit, learned, adopted, prefetched):
        return {"event": "prediction", "hit": hit, "learned": learned, "adopted": adopted, "spec_prefetched": prefetched}
    rows = [row(True, True, True, True), row(False, True, False, True), row(True, False, False, False), row(False, False, False, False)]
    got = st.guess_stats(rows)
    assert got == {"n": 4, "hits": 2, "adopted": 1, "learned_n": 2, "learned_hits": 1, "wasted_prefetch": 1}
    assert st.guess_stats([])["n"] == 0
