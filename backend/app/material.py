"""Material for a bit (D-116): a topic from the topic list and two facts from the character sheet, drawn so the writer
has something other than the robot's hardware to build on. Data lives in `app/material/*.yaml`.

Used means PLAYED: a topic or fact is spent only when its bit's thread was told (`ShowMaterial.played`), mirroring
`ThreadGenerator.played_thread`. A draw is a pure function of what has been played, so a bit written on a guess that
is thrown away (D-115) spends nothing, and drawing twice for one bit gives one answer."""
import os
import pathlib
import random
import secrets

import yaml

from app.cache import normalise
from app.safety import STANDARD_AGES

DATA = pathlib.Path(__file__).resolve().parent / "material"
SHEET_ID = "character-v1"
LIST_ID = "topics-v1"
FACTS_PER_BIT = 2
CHARACTER_EVERY = 3  # bits 1, 4, 7 take a topic from the robot's own life (`from: character`), the rest an everyday one
CHARACTER, EVERYDAY = "character", "everyday"


def _load(name: str, key: str) -> list:
    with (DATA / f"{name}.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)[key]


def load_sheet(sheet_id: str = SHEET_ID) -> list[str]:
    return list(_load(sheet_id, "facts"))


def load_topics(list_id: str = LIST_ID) -> list[tuple[str, bool, str]]:
    """(topic, adult, from) in file order; `from` is "character" or "everyday"."""
    return [(t["topic"], bool(t.get("adult", False)), t.get("from", EVERYDAY)) for t in _load(list_id, "topics")]


def used_topics(sessions: list[list[dict]]) -> dict[str, float]:
    """Every topic played in any logged show, any room, sim or live, with the time it was last used (the `material`
    row's `at`). Mirrors `cache.heard_texts`: on this laptop the listener is always the same person."""
    last: dict[str, float] = {}
    for rows in sessions:
        for r in rows:
            if r["event"] == "material":
                key = normalise(r["topic"])
                last[key] = max(last.get(key, float("-inf")), r["at"])
    return last


class ShowMaterial:
    """One per show. `seed` orders the topics and facts and is logged in each `material` row, so the show replays."""

    def __init__(self, seed: int | None = None, history: dict[str, float] | None = None,
                 facts: list[str] | None = None, topics: list[tuple[str, bool, str]] | None = None):
        self.seed = secrets.randbits(32) if seed is None else seed
        rng = random.Random(self.seed)
        self.facts = list(load_sheet() if facts is None else facts)
        self.topics = list(load_topics() if topics is None else topics)
        rng.shuffle(self.facts)
        rng.shuffle(self.topics)
        self.history = dict(history or {})
        self.sheet_id, self.list_id = SHEET_ID, LIST_ID
        self._topic_plays: list[str] = []  # topics played this show, in order
        self._fact_uses: dict[str, int] = {}
        self._played_bits: set[int] = set()

    def set_history(self, history: dict[str, float]) -> None:
        self.history = dict(history)

    def draw(self, bit: int, age: str) -> dict:
        """{topic, from, facts, sheet_id, list_id, recycled}. A bit takes from one pool by its number (`CHARACTER_EVERY`):
        the first topic in the seeded order that no earlier show and no earlier bit of this show used; if that pool has
        none, the other pool's; when every topic has been used, the least recently used of the bit's own pool (this
        show's plays count as the latest) and `recycled`. An adult topic is skipped in a room outside STANDARD_AGES.
        The facts are the least used this show, ties broken by the seeded order."""
        want = CHARACTER if bit % CHARACTER_EVERY == 1 else EVERYDAY
        eligible = [(t, src) for t, adult, src in self.topics if not adult or age in STANDARD_AGES]
        fresh = [(t, src) for t, src in eligible if normalise(t) not in self.history and t not in self._topic_plays]
        recycled = not fresh
        if recycled:
            pool = [x for x in eligible if x[1] == want] or eligible

            def recency(item):
                i, (t, _) = item
                if t in self._topic_plays:
                    return (1, self._topic_plays.index(t), i)
                return (0, self.history[normalise(t)], i)
            topic, source = min(enumerate(pool), key=recency)[1]
        else:
            topic, source = next((x for x in fresh if x[1] == want), fresh[0])
        order = {f: i for i, f in enumerate(self.facts)}
        facts = sorted(self.facts, key=lambda f: (self._fact_uses.get(f, 0), order[f]))[:FACTS_PER_BIT]
        return {"topic": topic, "from": source, "facts": facts, "sheet_id": self.sheet_id, "list_id": self.list_id,
                "recycled": recycled}

    def played(self, bit: int, draw: dict) -> None:
        """The bit's thread was told: its topic and facts are spent. A second call for the same bit does nothing."""
        if bit in self._played_bits:
            return
        self._played_bits.add(bit)
        self._topic_plays.append(draw["topic"])
        for f in draw["facts"]:
            self._fact_uses[f] = self._fact_uses.get(f, 0) + 1


def enabled(environ=None) -> bool:
    """The MATERIAL setting: off unless 1, true, yes or on."""
    value = (os.environ if environ is None else environ).get("MATERIAL", "0")
    return value.strip().lower() in ("1", "true", "yes", "on")
