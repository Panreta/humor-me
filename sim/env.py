"""Simulated audience: a gym-style env where each step the policy picks a persona mode and the
room answers with a laugh level 0-3.

Latent per joke: mu(mode, context, t, history) + N(0, noise); level = thresholds cleared.
Reward = level / 3, so samplers work on [0, 1] while logs keep the 0-3 level that live laugh
detection and transcript labels use.

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
    def __init__(self, priors: dict | None = None, n_jokes: int | None = None,
                 shift_at: int | None = None, shift: dict | None = None):
        p = priors or load_priors()
        self.modes: tuple[str, ...] = tuple(p["modes"])
        self.p = p
        d = p["dynamics"]
        self.thresholds = np.array(d["thresholds"], dtype=float)
        self.noise = float(d["noise"])
        self.n_jokes = n_jokes or int(d["n_jokes"])
        self.shift_at = shift_at
        self._shift = self._vec(shift if shift is not None else p.get("shift", {}))
        self.rng = np.random.default_rng()
        self.context: Context | None = None
        self.t = 0
        self.history: list[int] = []

    @property
    def n_modes(self) -> int:
        return len(self.modes)

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
        # Context part of mu is fixed for the episode; time/history parts are added per step.
        self._static = (self._vec(p["base"]) + self._vec(p["venue"][c.venue])
                        + self._vec(p["age"][c.age]) + p["energy"][c.energy])
        self.t = 0
        self.history = []
        return self._obs(None, None)

    def mu(self) -> np.ndarray:
        """Latent mean per mode for the next joke, given the current time and history."""
        d = self.p["dynamics"]
        mu = self._static + d["novelty"] * math.exp(-self.t / d["novelty_tau"])
        recent = self.history[-d["fatigue_window"]:]
        mu = mu - d["fatigue"] * np.bincount(recent, minlength=self.n_modes)
        if self.shift_at is not None and self.t >= self.shift_at:
            mu = mu + self._shift
        return mu

    def expected_rewards(self) -> np.ndarray:
        """E[level]/3 per mode for the next joke: the oracle that regret is measured against."""
        z = self.noise
        return np.array([sum(_phi((m - th) / z) for th in self.thresholds) for m in self.mu()]) / MAX_LEVEL

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        if self.context is None:
            raise RuntimeError("call reset() first")
        if self.t >= self.n_jokes:
            raise RuntimeError("episode is over; call reset()")
        if not 0 <= action < self.n_modes:
            raise ValueError(f"action {action} not in [0, {self.n_modes})")
        expected = self.expected_rewards()
        latent = self.mu()[action] + self.rng.normal(0, self.noise)
        level = int((latent > self.thresholds).sum())
        reward = level / MAX_LEVEL
        info = {"level": level, "expected": expected, "regret": float(expected.max() - expected[action])}
        self.history.append(action)
        self.t += 1
        return self._obs(action, level), reward, self.t >= self.n_jokes, info

    def _obs(self, last_mode: int | None, last_level: int | None) -> dict:
        return {"t": self.t, "n_jokes": self.n_jokes, "context": self.context,
                "last_mode": last_mode, "last_level": last_level}
