"""D-105: the ballot and the judge must not be able to tell qwen's bits by their punctuation."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sim.blind import PLAIN, plain  # noqa: E402

TELLS = "’‘“”—–…"  # the characters only qwen wrote in the Exp 24 run


def test_no_tell_survives_and_plain_text_is_untouched():
    shown = plain("I’m “fine” — really… it’s 3–5 bugs. ‘Next’")
    assert not set(shown) & set(TELLS)
    assert shown == "I'm \"fine\" - really... it's 3-5 bugs. 'Next'"
    assert plain("I'm fine - really... it's 3-5 bugs.") == "I'm fine - really... it's 3-5 bugs."  # already plain: same


def test_an_em_dash_with_or_without_spaces_reads_the_same():
    assert plain("wait—what") == plain("wait — what") == "wait - what"


def test_every_tell_has_a_plain_form():
    assert set(TELLS) <= {chr(c) for c in PLAIN}
