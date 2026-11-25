"""Cached bits: the best-received bits of earlier live shows, kept as filler for the next show's silences
(docs/agent/decisions-detail.md, D-88). A cached line is never evaluated: no reaction window, no reward, no director
update. It was judged for its own show's audience, so `screen` re-checks it for this one before it can be said."""
from dataclasses import dataclass

from app.director import joke_levels
from app.reward import reward_v1


@dataclass(frozen=True)
class CachedLine:
    joke: int  # the line's index in its bit, for the log's `source`
    text: str


@dataclass(frozen=True)
class CachedBit:
    session: str
    bit: int
    mode: str
    reward: float  # recomputed from the reaction rows, never read from `bit_end.reward`
    lines: tuple[CachedLine, ...]


def popular_bits(sessions: list[list[dict]], per_mode: int = 2, min_labels: int = 3) -> dict[str, list[CachedBit]]:
    """The top `per_mode` bits of each mode, from live shows only: a simulated reward comes from made-up
    `audience.yaml` values, so it cannot say what was popular. Ranked by reward, ties to the later show. A bit with
    fewer than `min_labels` labels, no reward or no played line is skipped, and a bit whose lines repeat a better
    one's is dropped."""
    found: list[tuple[float, float, int, CachedBit]] = []
    for rows in sessions:
        start = next((r for r in rows if r["event"] == "show_start"), None)
        if start is None or start["room"] != "live":
            continue
        actions = {r["bit"]: r["action"] for r in rows if r["event"] == "bit_decision"}
        for end in (r for r in rows if r["event"] == "bit_end"):
            bit = end["bit"]
            levels = joke_levels(rows, bit)
            reward = reward_v1(levels)
            lines = tuple(CachedLine(r["joke"], r["text"]) for r in rows if r["event"] == "joke_played" and r["bit"] == bit)
            if bit not in actions or reward is None or not lines or sum(lv is not None for lv in levels) < min_labels:
                continue
            found.append((reward, start["at"], bit, CachedBit(start["session"], bit, actions[bit], reward, lines)))
    found.sort(key=lambda f: f[:3], reverse=True)
    out: dict[str, list[CachedBit]] = {}
    seen: set[tuple[str, ...]] = set()
    for *_, cb in found:
        texts = tuple(line.text for line in cb.lines)
        if texts in seen or len(out.setdefault(cb.mode, [])) >= per_mode:
            continue
        seen.add(texts)
        out[cb.mode].append(cb)
    return {mode: bits for mode, bits in out.items() if bits}


def screen(cached: dict[str, list[CachedBit]], check, age: str) -> list[CachedBit]:
    """Every bit with only the lines that pass `check` (the rules layer, no API call) for this show's `age`, best reward
    first; a bit with no passing line is dropped. A filter that cannot decide fails the line, as in the show."""
    def passes(text: str) -> bool:
        try:
            return bool(check(text, age).ok)
        except Exception:
            return False

    kept = [CachedBit(b.session, b.bit, b.mode, b.reward, tuple(line for line in b.lines if passes(line.text)))
            for bits in cached.values() for b in bits]
    return sorted((b for b in kept if b.lines), key=lambda b: -b.reward)
