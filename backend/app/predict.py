"""Guess the next bit's mode before the current bit ends (D-115), so its prep can start while the last jokes play.

The guess is the mode the director would pick if the bit ended now: a throwaway copy of the model, taught the reactions
so far, asked greedily. It is a copy, so the real director's matrices and random draws are untouched and the real
decision at the bit's end is exactly what it would have been without a guess. The sim's Exp 31 uses this same function."""
import copy
from os import environ

from app.reward import reward_v1


def predict_mode(inner, obs: dict, action: int, levels: list, *, n_bits: int, learn: bool = True) -> int:
    """`inner` is the director's LinUCB, `obs`/`action` the bit being played, `levels` its reactions so far.
    `learn` teaches the copy the partial reward first (only with at least MIN_LABELS labels); False asks the model as it is."""
    partial = reward_v1(levels)
    model = copy.deepcopy(inner)
    if learn and partial is not None:
        model.update(obs, action, partial)
    nxt = {"t": obs["t"] + 1, "n_bits": n_bits, "context": obs["context"], "last_mode": action, "last_reward": partial}
    return int(model.act(nxt))


def enabled(env=environ, default: bool = True) -> bool:
    """The PREDICT_NEXT_BIT setting: on unless 0, false, no or off (D-115)."""
    value = env.get("PREDICT_NEXT_BIT")
    return default if value is None else value.strip().lower() not in ("0", "false", "no", "off")
