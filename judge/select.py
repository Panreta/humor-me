"""Which pairs to compare: a random warm start, every pair, or the adaptive (LUCB) next pair.

Pairs come back as (a, b) with no position meaning; rank() decides which is shown first.
"""
import numpy as np


def warm_start_pairs(n: int, rounds: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """`rounds` random perfect matchings; with odd n, one random joke sits out each round."""
    pairs = []
    for _ in range(rounds):
        perm = rng.permutation(n)[: n - n % 2]
        pairs += [(int(perm[i]), int(perm[i + 1])) for i in range(0, len(perm), 2)]
    return pairs


def all_pairs(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Every unordered pair once, as (i, j) with i < j, in shuffled order."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [pairs[k] for k in rng.permutation(len(pairs))]
