"""Joke bank: shape, uniqueness and a whole show that never raises. Needs no FastAPI."""
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.bank import BankGenerator, load_bank  # noqa: E402

MODES = yaml.safe_load((ROOT / "sim" / "audience.yaml").read_text(encoding="utf-8"))["modes"]
N_BITS, JOKES_PER_BIT = 8, 5


def test_counts_and_mode_names():
    bank = load_bank()
    assert sorted(bank) == sorted(MODES)
    for mode, bits in bank.items():
        assert len(bits) == 4, mode
        for b in bits:
            assert b["premise"]
            assert len(b["jokes"]) == JOKES_PER_BIT, (mode, b["premise"])


def test_no_duplicate_lines_or_premises():
    bank = load_bank()
    lines = [j.strip().lower() for bits in bank.values() for b in bits for j in b["jokes"]]
    assert len(lines) == 120 and len(set(lines)) == 120
    premises = [b["premise"] for bits in bank.values() for b in bits]
    assert len(set(premises)) == len(premises)


def test_lines_are_single_speakable_lines():
    for bits in load_bank().values():
        for b in bits:
            for j in b["jokes"]:
                assert "\n" not in j and 20 <= len(j) <= 140, j


@pytest.mark.parametrize("mode", MODES)
def test_fixed_mode_show_never_raises_and_reuses_deterministically(mode):
    def play():
        gen = BankGenerator()
        return [gen.candidates(mode, bit, joke) for bit in range(N_BITS) for joke in range(JOKES_PER_BIT)]

    first = play()
    assert all(len(c) >= 1 and all(isinstance(t, str) and t for t in c) for c in first)
    assert first == play()
    # 8 bits on 4 bank bits: every bank bit is used exactly twice, none is starved
    gen = BankGenerator()
    for bit in range(N_BITS):
        gen.candidates(mode, bit, 0)
    assert gen._used[mode] == [2, 2, 2, 2]


def test_bit_stays_on_one_premise_and_mixed_show_runs():
    gen = BankGenerator()
    assert gen.generator_id == "bank-v1"
    for bit in range(N_BITS):
        mode = MODES[bit % len(MODES)]
        lines = [gen.candidates(mode, bit, j)[0] for j in range(JOKES_PER_BIT)]
        bank_bit = next(b for b in gen.bank[mode] if lines[0] in b["jokes"])
        assert lines == bank_bit["jokes"]


def test_unknown_mode_raises():
    with pytest.raises(KeyError):
        BankGenerator().candidates("grumpy", 0, 0)


def test_out_of_range_joke_index_raises():
    gen = BankGenerator()
    with pytest.raises(IndexError):
        gen.candidates("neurotic", 0, JOKES_PER_BIT)
