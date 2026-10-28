"""Simulated audience: a gym-style env where each step the policy picks the persona mode for the
next **bit** and the room answers every joke in it with a laugh level 0-3.

Decisions are per bit, not per joke: a comedian commits to one premise and attitude for a bit
and varies between bits (decisions.md 2026-10-27). Within a bit the mode is fixed.

Latent per joke: mu(mode, context, joke time, bit history) + N(0, noise); level = thresholds
cleared. Bit reward = mean level / 3, so samplers work on [0, 1] while logs keep the 0-3 levels
that live laugh detection and transcript labels use.

No gymnasium dependency: the API mirrors it (reset/step) but stays plain numpy so thousands
of episodes run in seconds.
"""
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import yaml

PRIORS = pathlib.Path(__file__).with_name("priors.yaml")
VENUES = ("bar", "company_event", "showcase")
ENERGIES = ("low", "normal", "high")
AGES = ("teens", "young_adults", "mixed")
MAX_LEVEL = 3


@dataclass(frozen=True)
class Context:
    venue: str
    energy: str
    age: str


def load_priors(path: pathlib.Path = PRIORS) -> dict:
    return yaml.safe_load(path.read_text())


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class AudienceEnv:
    def __init__(self, priors: dict | None = None, n_bits: int | None = None,
                 jokes_per_bit: int | None = None, shift_at: int | None = None,
                 shift: dict | None = None):
        p = priors or load_priors()
        self.modes: tuple[str, ...] = tuple(p["modes"])
        self.p = p
        d = p["dynamics"]
        self.thresholds = np.array(d["thresholds"], dtype=float)
        self.noise = float(d["noise"])
        self.n_bits = n_bits or int(d["n_bits"])
        self.jokes_per_bit = jokes_per_bit or int(d["jokes_per_bit"])
        self.shift_at = shift_at  # in bits
        self._shift = self._vec(shift if shift is not None else p.get("shift", {}))
        self.rng = np.random.default_rng()
        self.context: Context | None = None
        self.t = 0  # bits played
        self.history: list[int] = []

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def n_jokes(self) -> int:
        return self.n_bits * self.jokes_per_bit

    def _vec(self, offsets: dict) -> np.ndarray:
        unknown = set(offsets) - set(self.modes)
        if unknown:
            raise ValueError(f"unknown modes in priors: {sorted(unknown)}")
        return np.array([offsets.get(m, 0.0) for m in self.modes], dtype=float)

    def reset(self, context: Context | None = None, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.context = context or Context(
            str(self.rng.choice(VENUES)), str(self.rng.choice(ENERGIES)), str(self.rng.choice(AGES)))
        c, p = self.context, self.p
        # Context part of mu is fixed for the episode; time/history parts are added per joke.
        self._static = (self._vec(p["base"]) + self._vec(p["venue"][c.venue])
                        + self._vec(p["age"][c.age]) + p["energy"][c.energy])
        self.t = 0
        self.history = []
        return self._obs(None, None)

    def mu(self, joke_offset: int = 0) -> np.ndarray:
        """Latent mean per mode for joke `joke_offset` of the next bit."""
        d = self.p["dynamics"]
        joke_t = self.t * self.jokes_per_bit + joke_offset
        mu = self._static + d["novelty"] * math.exp(-joke_t / d["novelty_tau"])
        recent = self.history[-d["fatigue_window"]:]
        mu = mu - d["fatigue"] * np.bincount(recent, minlength=self.n_modes)
        if self.shift_at is not None and self.t >= self.shift_at:
            mu = mu + self._shift
        return mu

    def expected_rewards(self) -> np.ndarray:
        """E[mean level]/3 per mode for the next bit: the oracle that regret is measured against."""
        z = self.noise
        total = np.zeros(self.n_modes)
        for j in range(self.jokes_per_bit):
            total += [sum(_phi((m - th) / z) for th in self.thresholds) for m in self.mu(j)]
        return total / self.jokes_per_bit / MAX_LEVEL

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        if self.context is None:
            raise RuntimeError("call reset() first")
        if self.t >= self.n_bits:
            raise RuntimeError("episode is over; call reset()")
        if not 0 <= action < self.n_modes:
            raise ValueError(f"action {action} not in [0, {self.n_modes})")
        expected = self.expected_rewards()
        mus = np.array([self.mu(j)[action] for j in range(self.jokes_per_bit)])
        latent = mus + self.rng.normal(0, self.noise, self.jokes_per_bit)
        levels = (latent[:, None] > self.thresholds).sum(1).astype(int).tolist()
        reward = sum(levels) / len(levels) / MAX_LEVEL
        info = {"levels": levels, "expected": expected, "regret": float(expected.max() - expected[action])}
        self.history.append(action)
        self.t += 1
        return self._obs(action, reward), reward, self.t >= self.n_bits, info

    def _obs(self, last_mode: int | None, last_reward: float | None) -> dict:
        return {"t": self.t, "n_bits": self.n_bits, "context": self.context,
                "last_mode": last_mode, "last_reward": last_reward}
