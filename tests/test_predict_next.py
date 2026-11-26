"""Guessing the next bit's mode while the current bit's last jokes play (D-115). The load-bearing claim: the show,
its director and its rewards are the same with the guess on or off; only the wait changes. Real clock where
concurrency is the point."""
import asyncio
import json
import pathlib
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import generate  # noqa: E402
from app.log import SessionLog, read_session, validate  # noqa: E402
from app.predict import predict_mode  # noqa: E402
from app.rooms import RealClock, SimRoom  # noqa: E402
from app.show import GUESS_AFTER_JOKE, UniformSelector, run_show  # noqa: E402
from sim.env import AudienceEnv, Context  # noqa: E402
from tests.test_generate import CTX, FAST, FakePlayer, FakeVoice, Filter, Gen, completion, parallel_client, thread_reply  # noqa: E402
from tests.test_headless_show import Scripted, director, same  # noqa: E402

MODES = AudienceEnv().modes


class SpyLog(SessionLog):
    def __init__(self, directory, events):
        super().__init__(directory)
        self.events = events

    def write(self, event, **fields):
        self.events.append((event, fields.get("bit")))
        super().write(event, **fields)


class SpyGen(Gen):
    """Notes each call with its mode, how a stateful generator would be asked, and the premises told so far."""

    def __init__(self, events, **kw):
        super().__init__(**kw)
        self.events, self.told, self.told_at, self.cache = events, [], {}, {}

    def generate(self, mode, bit, joke, context):
        self.events.append(("generate", bit, mode))
        self.told_at.setdefault((bit, "real"), list(self.told))
        return super().generate(mode, bit, joke, context)

    def generate_speculative(self, mode, bit, joke, context):
        self.events.append(("speculate", bit, mode))
        self.cache[(bit, mode)] = True
        self.told_at.setdefault((bit, "spec"), list(self.told))
        return super().generate(mode, bit, joke, context)

    def adopt(self, bit, mode):
        self.events.append(("adopt", bit, mode))

    def discard(self, bit, mode):
        self.events.append(("discard", bit, mode))
        self.cache.pop((bit, mode), None)

    def played_thread(self, bit, thread):
        self.events.append(("told", bit))
        self.told.append(f"premise {bit}")


def play(tmp_path, n_bits=3, gen=None, voice=None, seconds=0.02, seed=0, d=None, predict=True, bridges=None,
         events=None, source=None, room_seed=0):
    events = [] if events is None else events
    voice = voice or FakeVoice()
    log = SpyLog(tmp_path, events)
    d = d or director(seed=3)
    asyncio.run(run_show(d, "sim", CTX, generator=gen or SpyGen(events), safety=Filter(), selector=UniformSelector(seed),
                         tts=voice, player=FakePlayer(voice, seconds=seconds), source=source or SimRoom(CTX, room_seed),
                         clock=RealClock(), log=log, n_bits=n_bits, config=FAST, bridge_lines=bridges, predict=predict,
                         explore_seed=1))
    return read_session(log.path), events, voice, d


def rows_of(rows, event):
    return [r for r in rows if r["event"] == event]


def core(rows):
    """What the show did, minus time stamps and the rows this feature adds: the golden comparison."""
    skip = {"prediction", "bridge", "prep", "gap"}
    out = []
    for r in rows:
        if r["event"] in skip or r["event"] in ("show_start", "show_end"):
            continue
        r = {k: v for k, v in r.items() if k not in ("at", "session", "audio_start_at", "audio_end_at", "tts_seconds",
                                                     "latency_s", "bridge_overhang_s")}
        out.append(r)
    return out


def test_guess_after_joke_three_is_the_default():
    assert GUESS_AFTER_JOKE == 3


def test_the_show_is_identical_with_the_guess_on_or_off(tmp_path):
    for seed in range(3):
        on, _, _, d_on = play(tmp_path, n_bits=5, seed=seed, d=director(seed=seed), room_seed=seed)
        off, _, _, d_off = play(tmp_path, n_bits=5, seed=seed, d=director(seed=seed), room_seed=seed, predict=False)
        assert core(on) == core(off)  # decisions (scores, probs, actions), candidates, picks, reactions, rewards
        assert same(d_on, d_off)
        assert d_on.rng.bit_generator.state == d_off.rng.bit_generator.state  # the exploration draws are the same
        assert rows_of(on, "prediction") and not rows_of(off, "prediction")


def test_a_guess_starts_after_joke_two_reacts_and_before_the_bit_ends(tmp_path):
    _, ev, _, _ = play(tmp_path, n_bits=2)
    first = next(i for i, e in enumerate(ev) if e[0] == "speculate")
    reactions = [i for i, e in enumerate(ev) if e == ("reaction", 0)]
    assert reactions[2] < first < reactions[3] or reactions[2] < first < ev.index(("bit_end", 0))
    assert first < ev.index(("bit_end", 0)) and ev[first][1] == 1


def test_the_guess_is_made_on_a_copy_and_leaves_the_director_untouched():
    d = director(seed=5)
    obs = {"t": 1, "n_bits": 8, "context": Context(**CTX), "last_mode": 2, "last_reward": 0.3}
    a0, b0, state = d.inner.A.copy(), d.inner.b.copy(), d.rng.bit_generator.state
    for learn in (True, False):
        mode = predict_mode(d.inner, obs, 3, [3, 2, 1], n_bits=8, learn=learn)
        assert 0 <= mode < len(MODES)
    assert np.array_equal(a0, d.inner.A) and np.array_equal(b0, d.inner.b) and state == d.rng.bit_generator.state


def test_the_guess_learns_from_the_partial_reward_only_with_enough_labels():
    d = director(seed=5)
    obs = {"t": 1, "n_bits": 8, "context": Context(**CTX), "last_mode": 2, "last_reward": None}
    # two labels is too thin to teach the copy (MIN_LABELS), so learning changes nothing
    assert predict_mode(d.inner, obs, 0, [3, 3, None], n_bits=8, learn=True) == predict_mode(
        d.inner, obs, 0, [3, 3, None], n_bits=8, learn=False)


def test_a_hit_adopts_the_prep_and_a_miss_discards_it(tmp_path):
    seen = set()
    for seed in range(8):
        rows, ev, _, _ = play(tmp_path, n_bits=4, seed=seed, d=director(seed=seed), room_seed=seed)
        for p in rows_of(rows, "prediction"):
            bit = p["bit"]
            if p["hit"]:
                assert ("adopt", bit, p["decided"]) in ev and ("told", bit) in ev
                assert ("discard", bit, p["predicted"]) not in ev
            else:
                assert ("discard", bit, p["predicted"]) in ev and ("adopt", bit, p["decided"]) not in ev
                assert ev.count(("generate", bit, p["decided"])) >= 1  # prepared again for the real mode
            seen.add(p["hit"])
    assert seen == {True, False}  # both paths ran


def test_a_missed_guess_leaves_no_row_a_hit_marks_its_prep_row_speculative(tmp_path):
    for seed in range(8):
        rows, _, _, _ = play(tmp_path, n_bits=4, seed=seed, d=director(seed=seed), room_seed=seed)
        preps = {}
        for r in rows_of(rows, "prep"):
            preps.setdefault(r["bit"], []).append(r)
        assert all(len(v) == 1 for v in preps.values())  # exactly one prep row per bit, so none from a thrown-away guess
        for p in rows_of(rows, "prediction"):
            assert preps[p["bit"]][0].get("speculative", False) is p["hit"]
        assert not any(r.get("speculative") for b in (0,) for r in preps[b])
        for r in rows_of(rows, "prediction"):
            assert r["started_at"] > 0 and r["hit"] == (r["predicted"] == r["decided"])


def test_the_last_bit_is_never_guessed(tmp_path):
    rows, ev, _, _ = play(tmp_path, n_bits=3)
    assert sorted(p["bit"] for p in rows_of(rows, "prediction")) == [1, 2]
    assert {e[1] for e in ev if e[0] == "speculate"} == {1, 2}


def test_no_guess_without_one_when_the_bit_is_short_or_the_setting_is_off(tmp_path):
    rows, _, _, _ = play(tmp_path, predict=False)
    assert not rows_of(rows, "prediction")


def test_a_stateful_generator_that_cannot_keep_a_guess_apart_is_not_guessed_for(tmp_path):
    class Stateful(Gen):
        def played_thread(self, bit, thread):
            pass
    rows, _, _, _ = play(tmp_path, gen=Stateful())
    assert not rows_of(rows, "prediction")


def wait_s(rows, bit):
    end = next(r for r in rows if r["event"] == "bit_end" and r["bit"] == bit - 1)["at"]
    return next(r for r in rows if r["event"] == "joke_played" and r["bit"] == bit)["audio_start_at"] - end


def test_a_hit_closes_the_gap_and_a_miss_falls_back_to_the_filler_path(tmp_path):
    waits = {True: [], False: []}
    for seed in range(4):
        rows, _, _, _ = play(tmp_path, n_bits=3, seed=seed, d=director(seed=seed), room_seed=seed,
                             gen=Gen(delay=0.08), seconds=0.4, bridges=["b1", "b2", "b3"])
        off, _, _, _ = play(tmp_path, n_bits=3, seed=seed, d=director(seed=seed), room_seed=seed, predict=False,
                            gen=Gen(delay=0.08), seconds=0.4, bridges=["b1", "b2", "b3"])
        for p in rows_of(rows, "prediction"):
            waits[p["hit"]].append((wait_s(rows, p["bit"]), wait_s(off, p["bit"])))
            fillers = [r for r in rows if r["event"] == "bridge" and r["bit"] == p["bit"]]
            if p["hit"]:
                assert not fillers
    assert waits[True] and waits[False]
    assert all(on < 0.25 and off > on + 0.25 for on, off in waits[True])  # prep (0.4 s) hides behind jokes 4 and 5
    assert all(abs(on - off) < 0.35 for on, off in waits[False])  # a miss costs about what no guess costs


class OrderVoice(FakeVoice):
    """Not serial on purpose: an early render would overlap the current bit's, which a serial voice hides as a queue."""

    def __init__(self, delay):
        super().__init__(delay)
        self.spans = {}

    def synthesize(self, text):
        start = time.perf_counter()
        wav = super().synthesize(text)
        self.spans[text] = (start, time.perf_counter())
        return wav


def test_the_guessed_bits_first_render_waits_for_the_current_bits_remaining_renders(tmp_path):
    voice = OrderVoice(0.15)
    play(tmp_path, n_bits=2, voice=voice, seconds=0.3, gen=Gen())  # instant prep: the worst case for queuing
    now = [s for t, s in voice.spans.items() if t.startswith("ok 0.")]
    nxt = [s for t, s in voice.spans.items() if t.startswith("ok 1.")]
    assert len(now) == 5 and nxt
    assert min(s for s, _ in nxt) >= max(e for _, e in now) - 1e-3


def test_an_abort_cancels_the_guess_and_writes_nothing_after_show_end(tmp_path):
    import threading
    release, started = threading.Event(), threading.Event()
    events = []

    class Blocking(SpyGen):
        def generate_speculative(self, mode, bit, joke, context):
            started.set()
            release.wait(5)
            return super().generate_speculative(mode, bit, joke, context)

    async def main():
        voice, log = FakeVoice(), SpyLog(tmp_path, events)
        task = asyncio.ensure_future(run_show(
            director(), "sim", CTX, generator=Blocking(events), safety=Filter(), selector=UniformSelector(0), tts=voice,
            player=FakePlayer(voice, seconds=0.2), source=SimRoom(CTX, 0), clock=RealClock(), log=log, n_bits=3,
            config=FAST))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.sleep(0.2)  # a guess that was not cancelled would write its row now
        return log

    rows = read_session(asyncio.run(main()).path)
    assert rows[-1]["event"] == "show_end" and rows[-1]["status"] == "aborted"
    assert [r["bit"] for r in rows if r["event"] == "prep"] == [0] and not rows_of(rows, "prediction")


def test_the_premise_told_to_the_next_bit_is_the_one_just_played_hit_or_miss(tmp_path):
    for seed in range(6):
        events = []
        gen = SpyGen(events)
        rows, _, _, _ = play(tmp_path, n_bits=3, seed=seed, d=director(seed=seed), room_seed=seed, gen=gen, events=events)
        # whichever prep ran for the bit (guessed or real), it saw exactly the premises of the bits already played
        for bit in (1, 2):
            seen = gen.told_at.get((bit, "spec"))
            assert seen == [f"premise {b}" for b in range(bit)]
        assert gen.told == ["premise 0", "premise 1", "premise 2"]  # one premise per bit played, none for a discarded guess


# --- the real generator: a guess is kept apart, keyed by the mode it was written for ---


def test_a_guessed_thread_never_answers_the_real_request_and_adds_no_premise():
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abcdef"))
    g = generate.ThreadGenerator(c)
    spec = g.generate_speculative("cynic", 1, 0, CTX)
    assert g._bits == {} and g._told == [] and (1, "cynic") in g._spec
    real = g.generate("observer", 1, 0, CTX)  # a different mode: fetched fresh, not the guess
    assert "observer" in real.prompt_id and "cynic" in spec.prompt_id
    assert c.attempts == 6 and g._told == []
    g.discard(1, "cynic")
    assert g._spec == {}
    g.played_thread(1, 0)
    assert len(g._told) == 1  # only the adopted/real bit's premise


def test_adopting_a_guess_moves_it_over_and_tells_its_premise():
    c, seen, _ = parallel_client(*(thread_reply(tag=t) for t in "abc"))
    g = generate.ThreadGenerator(c)
    g.generate_speculative("cynic", 1, 0, CTX)
    g.adopt(1, "cynic")
    assert 1 in g._bits and g._spec == {}
    again = g.generate("cynic", 1, 1, CTX)
    assert c.attempts == 3  # no second call: the adopted bit answers
    g.played_thread(1, 0)
    assert len(g._told) == 1 and again.texts


# --- the log ---


def row(**fields):
    base = dict(schema=2, session="s", at=1.0, event="prediction", bit=1, predicted="cynic", decided="cynic", hit=True,
                started_at=10.0)
    return {**base, **fields}


@pytest.mark.parametrize("bad", [dict(hit="yes"), dict(hit=False), dict(predicted=3), dict(started_at="x"),
                                 dict(decided="observer", hit=True)])
def test_the_log_refuses_a_malformed_prediction_row(bad):
    with pytest.raises(ValueError):
        validate(row(**bad))


def test_the_log_accepts_a_good_prediction_row_and_a_speculative_prep_row():
    validate(row())
    validate(row(decided="observer", hit=False))
    prep = dict(schema=2, session="s", at=1.0, event="prep", bit=1, started_at=1.0, generate_s=1.0, safety_s=1.0,
                render_wait_s=1.0, ready_at=4.0, bank_fallbacks=0)
    validate({**prep, "speculative": True})
    with pytest.raises(ValueError):
        validate({**prep, "speculative": "yes"})


def test_teaching_the_copy_the_partial_reward_changes_the_guess_somewhere():
    d = director(seed=5)
    differs = 0
    for action in range(len(MODES)):
        for levels in ([3, 3, 3], [0, 0, 0]):
            obs = {"t": 1, "n_bits": 8, "context": Context(**CTX), "last_mode": 0, "last_reward": 0.5}
            differs += predict_mode(d.inner, obs, action, levels, n_bits=8, learn=True) != predict_mode(
                d.inner, obs, action, levels, n_bits=8, learn=False)
    assert differs >= 3
