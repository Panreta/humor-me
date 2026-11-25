"""Pauses on the show clock, and which jokes one spoiled (D-92). The live loop and the replay both call `overlaps`,
so a joke's label is dropped from the reward by the same rule in both. No FastAPI imports."""
import math

Interval = tuple[float, float]


def intervals(rows: list[dict]) -> list[Interval]:
    """Pause intervals from a show's `pause` and `resume` rows, in log order. A pause with no resume (the show ended
    while paused) runs to infinity: nothing plays after it. A repeated pause or a stray resume is ignored; the
    server never writes one."""
    out: list[Interval] = []
    opened = None
    for r in rows:
        if r["event"] == "pause" and opened is None:
            opened = r["paused_at"]
        elif r["event"] == "resume" and opened is not None:
            out.append((opened, r["resumed_at"]))
            opened = None
    if opened is not None:
        out.append((opened, math.inf))
    return out


def overlaps(start: float, end: float, pauses: list[Interval]) -> bool:
    """Whether a pause shares time with the audio span [start, end]. Touching at one instant is not sharing: a pause
    that starts when the line ends, or ends when it starts, spoiled nothing."""
    return any(p < end and q > start for p, q in pauses)


def paused_jokes(rows: list[dict], bit: int) -> set[int]:
    """Jokes of `bit` whose playback overlapped a pause. A pause in the reaction window or between lines drops none."""
    pauses = intervals(rows)
    if not pauses:
        return set()
    return {r["joke"] for r in rows if r["event"] == "joke_played" and r["bit"] == bit
            and overlaps(r["audio_start_at"], r["audio_end_at"], pauses)}


class Pauses:
    """The live show's own pause state, kept beside the log rows the server writes, so `_joke` can apply `overlaps`
    the moment a line ends."""

    def __init__(self) -> None:
        self._closed: list[Interval] = []
        self._since: float | None = None

    @property
    def paused(self) -> bool:
        return self._since is not None

    def pause(self, now: float) -> None:
        self._since = now

    def resume(self, now: float) -> None:
        self._closed.append((self._since, now))
        self._since = None

    def intervals(self) -> list[Interval]:
        return self._closed + ([(self._since, math.inf)] if self._since is not None else [])
