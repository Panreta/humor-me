"""The show loop: `run_show` runs one show against whatever voice, room and clock it is given."""
import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.log import SessionLog
from app.reward import VERSION, reward_v1
from app.rooms import REACTION
from app.tts import wav_seconds
from sim.env import AudienceEnv, Context

_log = logging.getLogger(__name__)


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...


@dataclass(frozen=True)
class Batch:
    prompt_id: str
    prompt: str  # exactly what the model saw, so a log replays the generation
    texts: list[str]


class Generator(Protocol):
    generator_id: str

    def generate(self, mode: str, bit: int, joke: int, context: dict) -> Batch: ...


class SafetyFilter(Protocol):
    def check(self, text: str, age: str): ...  # object with .ok and .reason
    # Optionally `prefetch(texts, age)`: the show calls it once per bit, ahead of the per-line `check`s.


class Selector(Protocol):
    selector_id: str

    def pick(self, passing_texts: list[str]) -> tuple[int, float]: ...


class Player(Protocol):
    async def play(self, wav: bytes, text: str) -> None: ...  # returns when the audio has finished


class Clock(Protocol):
    def now(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...


class RewardSource(Protocol):
    source: str  # button | detector | sim

    def begin_bit(self, action: int) -> None: ...
    async def reaction(self, bit: int, joke: int, timeout: float) -> tuple[int | None, str | None, float]: ...

    async def hold(self, bit: int, joke: int, seconds: float, clock: Clock) -> tuple[int | None, str | None, float] | None:
        """Wait out the hold. Returns a corrected (level, key, latency_s) if the operator changed their mind, else None."""
        ...


class UniformSelector:
    selector_id = "uniform"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def pick(self, passing_texts):
        return int(self.rng.integers(len(passing_texts))), 1 / len(passing_texts)


class _SerialVoice:
    """Kokoro's pipeline is not promised to be thread-safe, and the show synthesizes from several threads at once."""

    def __init__(self, tts: TTS):
        self.tts, self.lock = tts, threading.Lock()

    def synthesize(self, text: str) -> bytes:
        with self.lock:
            return self.tts.synthesize(text)


def _judge(check, text: str, age: str) -> dict:
    try:
        verdict = check(text, age)
        ok, reason = bool(verdict.ok), verdict.reason
    except Exception as e:  # a filter that cannot decide must not let the joke through
        ok, reason = False, f"filter error: {type(e).__name__}: {e}"
    return {"text": text, "verdict": "pass" if ok else "reject", "reason": reason}


@dataclass
class _Plan:
    """One joke decided before it plays: what was picked, and the voice render started ahead of it."""
    batch: Batch
    chosen: int
    prob: float
    render: asyncio.Task | None = None


async def _render(tts, text: str, clock):
    asked = clock.now()
    wav = await asyncio.to_thread(tts.synthesize, text)  # keeps the event loop free
    return wav, clock.now() - asked


async def _prepare(bit, mode, context, *, generator, fallback, safety, selector, tts, clock, log, jokes_per_bit, spawn):
    """Everything a bit needs before its first line: candidates, the safety verdicts and the first render.
    A joke whose generation failed, or whose candidates were all rejected, falls back to the hand-checked bank."""
    age = context["age"]
    batches: list[Batch | None] = []
    reasons: list[str] = []
    for joke in range(jokes_per_bit):
        try:
            batches.append(await asyncio.to_thread(generator.generate, mode, bit, joke, context))
            reasons.append("")
        except Exception as e:  # a bad reply or a dead provider must not stop the show
            batches.append(None)
            reasons.append(f"generation failed: {type(e).__name__}: {e}")
            if not fallback:  # nothing will play for this joke, so the log must still say why (invariant 6)
                _log.warning("bit %s joke %s: %s; no fallback, skipping", bit, joke, reasons[-1])
                log.write("candidates", bit=bit, joke=joke, prompt_id=f"{generator.generator_id}:failed", prompt="",
                          generator_id=generator.generator_id,
                          items=[{"text": "", "verdict": "reject", "reason": reasons[-1]}])
    if prefetch := getattr(safety, "prefetch", None):  # the LLM layer judges the whole bit in one call
        await asyncio.to_thread(prefetch, [t for b in batches if b for t in b.texts], age)
    plans: list[_Plan | None] = []
    for joke, batch in enumerate(batches):
        plan = None
        if batch:
            items = [_judge(safety.check, t, age) for t in batch.texts]
            log.write("candidates", bit=bit, joke=joke, prompt_id=batch.prompt_id, prompt=batch.prompt,
                      generator_id=generator.generator_id, items=items)
            passing = [i for i, item in enumerate(items) if item["verdict"] == "pass"]
            if passing:
                pick, prob = selector.pick([batch.texts[i] for i in passing])
                plan = _Plan(batch, passing[pick], prob)
            else:
                reasons[joke] = "every candidate rejected"
        if plan is None and fallback:
            _log.warning("bit %s joke %s: %s; using %s", bit, joke, reasons[joke], fallback.generator_id)
            batch = await asyncio.to_thread(fallback.generate, mode, bit, joke, context)
            # The bank is hand-checked, so only the rules run on it: no API call while the provider is failing.
            rules = getattr(safety, "check_rules", safety.check)
            items = [_judge(rules, t, age) for t in batch.texts]
            for item in items:
                item["reason"] = f"bank fallback ({reasons[joke]}); rules: {item['reason']}"
            log.write("candidates", bit=bit, joke=joke, prompt_id=batch.prompt_id, prompt=batch.prompt,
                      generator_id=fallback.generator_id, items=items)
            passing = [i for i, item in enumerate(items) if item["verdict"] == "pass"]
            if passing:
                pick, prob = selector.pick([batch.texts[i] for i in passing])
                plan = _Plan(batch, passing[pick], prob)
        plans.append(plan)
    _start_render(plans, 0, tts, clock, spawn)
    if first := next((p for p in plans if p), None):
        await first.render  # the bit is ready only when its first line is audio, or the bridge ends into a render wait
    return plans


def _start_render(plans, joke, tts, clock, spawn) -> None:
    """Render the next playable line from `joke` on, so its audio is ready when the current one ends."""
    plan = next((p for p in plans[joke:] if p), None)
    if plan and plan.render is None:
        plan.render = spawn(_render(tts, plan.batch.texts[plan.chosen], clock))


async def _joke(bit, joke, plans, *, tts, player, source, clock, log, config, spawn):
    """One joke, end to end. Returns the level, or None when nothing was played or the room did not answer."""
    plan = plans[joke]
    if plan is None:
        return None  # unlabeled: no reaction row, so replay sees the same thing
    _start_render(plans, joke, tts, clock, spawn)
    _start_render(plans, joke + 1, tts, clock, spawn)
    wav, tts_seconds = await plan.render
    text = plan.batch.texts[plan.chosen]
    start = clock.now()
    await player.play(wav, text)
    end = clock.now()
    log.write("joke_played", bit=bit, joke=joke, chosen=plan.chosen, pick_prob=plan.prob, text=text,
              tts_seconds=tts_seconds, audio_seconds=wav_seconds(wav), audio_start_at=start, audio_end_at=end)
    level, key, latency = await source.reaction(bit, joke, config.timeout_s)
    log.write("reaction", bit=bit, joke=joke, source=source.source, level=level, key=key, latency_s=latency)
    before = clock.now()
    while True:
        hold = config.holds[level]
        fix = await source.hold(bit, joke, hold, clock)
        if fix is None:
            break
        level, key, latency = fix  # last press wins; the hold restarts at the new level's length
        log.write("reaction", bit=bit, joke=joke, source=source.source, level=level, key=key, latency_s=latency)
    log.write("gap", bit=bit, joke=joke, planned_hold_s=hold, actual_gap_s=clock.now() - before)
    return level


MAX_BRIDGES = 3  # a bit that is still loading after three bridge lines is a provider problem, not a pause


async def _bridge(prep: asyncio.Task, rendered: list, player, used: list) -> None:
    """Fill the wait for the next bit with in-character lines, so the room never hears dead air. Uses whatever has
    rendered so far and never blocks on the rest; a line is not repeated while an unused one remains."""
    for _ in range(MAX_BRIDGES):
        if prep.done() or not rendered:
            return
        unused = [b for b in rendered if b[0] not in used]
        text, wav = unused[0] if unused else rendered[len(used) % len(rendered)]
        used.append(text)
        await player.play(wav, text)


async def run_show(director, room: str, context: dict, *, generator: Generator, safety: SafetyFilter,
                   selector: Selector, tts: TTS, player: Player, source: RewardSource, clock: Clock, log: SessionLog,
                   explore_seed: int | None = None, operator: str = "sim", git_sha: str = "unknown",
                   audio: dict | None = None, n_bits: int = 8, jokes_per_bit: int = 5, config=REACTION,
                   fallback: Generator | None = None, bridge_lines: list[str] | None = None) -> None:
    """One show. `director` is an EpsilonSoft over LinUCB, already warm: the caller replays the career into it
    (app.director.load_director) so this loop never decides which rooms count."""
    modes = AudienceEnv().modes
    inner = director.inner
    tasks: list[asyncio.Task] = []

    def spawn(coro) -> asyncio.Task:
        tasks.append(asyncio.ensure_future(coro))
        return tasks[-1]

    tts = _SerialVoice(tts)
    rendered: list[tuple[str, bytes]] = []  # bridge lines that are ready; a failed one is just not here
    bridges_used: list[str] = []

    async def render_bridges():
        for text in bridge_lines:
            try:
                rendered.append((text, await asyncio.to_thread(tts.synthesize, text)))
            except Exception:  # filler must never stop the show
                _log.warning("bridge line did not render: %r", text, exc_info=True)
    log.write("show_start", room=room, context=context,
              policy={"name": "linucb", "epsilon": director.eps, "alpha": inner.alpha, "prior": "uniform",
                      "seed": explore_seed},
              generator_id=generator.generator_id, selector_id=selector.selector_id, git_sha=git_sha,
              operator=operator, audio=audio or {"consent": False, "path": None})
    try:
        last_action = last_reward = None
        for bit in range(n_bits):
            obs = {"t": bit, "n_bits": n_bits, "context": Context(**context), "last_mode": last_action,
                   "last_reward": last_reward}
            action, greedy, probs = director.decide(obs)
            scores = inner.scores(obs).sum(axis=1)  # prediction + exploration bonus
            log.write("bit_decision", bit=bit,
                      obs={"t": bit, "n_bits": n_bits, "context": context,
                           "last_mode": None if last_action is None else modes[last_action]},
                      scores={m: float(x) for m, x in zip(modes, scores)}, greedy=modes[greedy],
                      action=modes[action], probs={m: float(p) for m, p in zip(modes, probs)})
            source.begin_bit(action)
            prep = spawn(_prepare(bit, modes[action], context, generator=generator, fallback=fallback, safety=safety,
                                  selector=selector, tts=tts, clock=clock, log=log, jokes_per_bit=jokes_per_bit,
                                  spawn=spawn))
            if bridge_lines and bit > 0:
                await _bridge(prep, rendered, player, bridges_used)
            plans = await prep
            if bridge_lines and bit == 0:
                spawn(render_bridges())  # after bit 0's first line, so the voice is not busy when the show opens
            levels = [await _joke(bit, joke, plans, tts=tts, player=player, source=source, clock=clock, log=log,
                                  config=config, spawn=spawn) for joke in range(jokes_per_bit)]
            reward = reward_v1(levels)
            log.write("bit_end", bit=bit, n_labels=sum(lv is not None for lv in levels), reward_version=VERSION,
                      reward=reward)
            if reward is not None:
                director.update(obs, action, reward)
            last_action, last_reward = action, reward
    except BaseException:  # includes cancellation: a closed tab must still leave a readable log
        log.write("show_end", status="aborted")
        raise
    finally:
        for task in tasks:
            task.cancel()  # a render nobody will play, or a bit nobody will hear
    log.write("show_end", status="finished")
