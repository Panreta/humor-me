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
              "bank_fallbacks"}


class SlowGenerator:
    generator_id = "slow"

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
        self.clock.t += self.synth_s + (self.queued_s or 0.0)
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
    return r["generate_s"] + r["safety_s"] + r["bank_s"] + r["render_wait_s"]


def test_every_bit_logs_one_prep_row_with_consistent_non_negative_stages(tmp_path):
    got = preps(show(tmp_path, gen=lambda c: SlowGenerator(c, 0.7), n_bits=3, jokes=2))
    assert [r["bit"] for r in got] == [0, 1, 2]
    for r in got:
        assert SIM_RESULT <= set(r)
        for key in ("generate_s", "safety_s", "bank_s", "render_wait_s"):
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
                         bridge_lines=["b1", "b2", "b3", "b4"]))
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


def write_show(directory, sid, bits, at=1000.0, room="live", with_prep=True):
    """bits: (wait_s, stages dict, overhang_s, bridges). Each bit starts at the previous bit_end (`at` for bit 0)."""
    rows, t = [{"event": "show_start", "at": at, "room": room, "context": {"venue": "bar", "energy": "normal", "age": "mixed"},
                "policy": {"name": "linucb", "epsilon": 0.1, "alpha": 1.0, "prior": "uniform"}, "generator_id": "g",
                "selector_id": "u", "git_sha": "x", "operator": "o", "audio": {"consent": False, "path": None}}], at
    for i, (wait, stages, overhang, bridges) in enumerate(bits):
        for _ in range(bridges):
            rows.append({"event": "bridge", "at": t + 1, "bit": i, "text": JOKE_TEXT, "tts_seconds": 1.0,
                         "audio_start_at": t + 1, "audio_end_at": t + 2})
        ready = t + sum(stages.values())
        if with_prep:
            rows.append({"event": "prep", "at": ready, "bit": i, "started_at": t, "ready_at": ready,
                         "generate_s": stages.get("generate_s", 0.0), "safety_s": stages.get("safety_s", 0.0),
                         "bank_s": stages.get("bank_s", 0.0), "render_wait_s": stages.get("render_queue_s", 0.0) + stages.get("render_s", 0.0),
                         "render_queue_s": stages.get("render_queue_s", 0.0), "render_s": stages.get("render_s", 0.0), "bank_fallbacks": 0})
        extra = {"bridge_overhang_s": overhang} if with_prep else {}
        rows.append({"event": "joke_played", "at": t + wait, "bit": i, "joke": 0, "chosen": 0, "pick_prob": 0.3, "text": JOKE_TEXT,
                     "tts_seconds": 0.1, "audio_seconds": 1.0, "audio_start_at": t + wait, "audio_end_at": t + wait + 1, **extra})
        t = t + wait + 20
        rows.append({"event": "bit_end", "at": t, "bit": i, "n_labels": 5, "reward_version": 1, "reward": 0.5})
    rows.append({"event": "show_end", "at": t, "status": "finished"})
    with (directory / f"{sid}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"schema": 2, "session": sid, **r}) + chr(10))


def dominated_by(stage, n=8, size=6.0):
    stages = {"generate_s": 2.0, "safety_s": 1.0, "bank_s": 0.0, "render_queue_s": 0.5, "render_s": 0.5}
    overhang = 0.5
    if stage == "bridge_overhang_s":
        overhang = size
    else:
        stages[stage] = size
    return [(sum(stages.values()) + overhang, dict(stages), overhang, 2)] * n


def test_a_bit_waits_from_the_previous_bit_end_and_the_stats_are_nearest_rank(tmp_path):
    write_show(tmp_path, "aaaaaaaaaaaa", [(10.0, {"generate_s": 4.0}, 0.0, 1), (5.0, {"generate_s": 2.0}, 1.5, 2), (8.0, {}, 0.0, 0)])
    (res,) = [st.run(None, tmp_path)["shows"][0]]
    assert res["bits"] == 3 and res["wait_s"]["p50"] == 8.0 and res["wait_s"]["p95"] == 10.0
    assert res["bridges"]["p50"] == 1 and res["bridge_overhang_s"]["p50"] == 0.0 and res["bridge_overhang_s"]["p95"] == 1.5
    assert res["generate_s"]["n"] == 3 and res["generate_s"]["p50"] == 2.0


@pytest.mark.parametrize("stage,outcome,fix", [
    ("bridge_overhang_s", "fix", "(a)"), ("render_queue_s", "fix", "(b)"), ("safety_s", "fix", "(c)"),
    ("generate_s", "no_candidate_fix", None), ("render_s", "no_candidate_fix", None),
])
def test_the_biggest_stage_picks_its_fix_and_the_writer_or_voice_has_none(tmp_path, stage, outcome, fix):
    write_show(tmp_path, "aaaaaaaaaaaa", dominated_by(stage, n=8, size=12.0))
    write_show(tmp_path, "bbbbbbbbbbbb", dominated_by(stage, n=8, size=12.0), at=5000.0)
    d = st.run(None, tmp_path)["decision"]
    assert d["outcome"] == outcome and d["stage"] == stage
    assert (fix is None and d["fix"] is None) or d["fix"].startswith(fix)


def test_a_tie_a_small_share_and_too_few_bits_decide_nothing(tmp_path):
    two = lambda bits: (write_show(tmp_path, "aaaaaaaaaaaa", bits), write_show(tmp_path, "bbbbbbbbbbbb", bits, at=5000.0))
    # overhang and safety are both 6 s of about 10.5 s: within 1.1x of each other
    two([(11.0, {"generate_s": 1.0, "safety_s": 6.0, "bank_s": 0.0, "render_queue_s": 0.0, "render_s": 0.0}, 6.0, 1)] * 8)
    assert st.run(None, tmp_path)["decision"]["outcome"] == "no_one_stage"
    for f in tmp_path.glob("*.jsonl"):
        f.unlink()
    two([(10.0, {"generate_s": 2.0, "safety_s": 2.0, "bank_s": 0.0, "render_queue_s": 2.0, "render_s": 2.0}, 2.0, 1)] * 8)
    assert st.run(None, tmp_path)["decision"]["outcome"] == "no_one_stage"  # no stage reaches a quarter of the wait
    for f in tmp_path.glob("*.jsonl"):
        f.unlink()
    write_show(tmp_path, "aaaaaaaaaaaa", dominated_by("safety_s", n=15, size=12.0))
    assert st.run(None, tmp_path)["decision"]["outcome"] == "not_enough_data"  # 15 < 16


def test_a_log_from_before_prep_is_listed_but_does_not_count(tmp_path):
    write_show(tmp_path, "aaaaaaaaaaaa", dominated_by("safety_s", n=20, size=12.0), with_prep=False)
    res = st.run(None, tmp_path)
    assert res["shows"][0]["bits"] == 20 and res["shows"][0]["bits_with_prep"] == 0 and res["shows"][0]["safety_s"]["p50"] is None
    assert res["pooled"]["bits_with_prep"] == 0 and res["decision"]["outcome"] == "not_enough_data"


def test_named_sessions_only_and_the_record_holds_seconds_and_ids_never_text(tmp_path, monkeypatch):
    logs, metrics = tmp_path / "logs", tmp_path / "metrics"
    logs.mkdir()
    monkeypatch.setattr(cache, "METRICS", metrics)
    write_show(logs, "aaaaaaaaaaaa", dominated_by("safety_s", n=8, size=12.0))
    write_show(logs, "bbbbbbbbbbbb", dominated_by("safety_s", n=8, size=12.0), at=5000.0)
    res = st.record(["aaaaaaaaaaaa"], logs)
    assert [s["session"] for s in res["shows"]] == ["aaaaaaaaaaaa"] and res["pooled"]["bits_with_prep"] == 8
    saved = (metrics / "stage_timing.json").read_text(encoding="utf-8")
    assert JOKE_TEXT not in saved and "aaaaaaaaaaaa" in saved
    assert st.stored() == json.loads(saved)["results"]
    assert "Biggest stage" in st.verdict(res) and "| aaaaaaaaaaaa" in st.table(res)
