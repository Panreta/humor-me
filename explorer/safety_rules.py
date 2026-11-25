"""The live show's rule-based safety layer (backend/app/safety.py check_rules), for offline data.

Training lines are checked under the "teens" rules, the strictest the show has, so the joke model never learns
a line the show would refuse in any room (CLAUDE.md invariant 1). Rules only: the show's second layer is an
LLM call per bit, which belongs to the live loop, not to a batch job. The rules fail closed, like the show.
Needs httpx (backend/app/llm.py imports it), which is in explorer/requirements-cluster.txt.
"""
import pathlib
import sys

AGE = "teens"
_BACKEND = pathlib.Path(__file__).resolve().parent.parent / "backend"


def _check_rules():
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    from app.safety import check_rules
    return check_rules


def verdict(text: str, age: str = AGE) -> tuple[bool, str]:
    """(ok, reason) from the show's rules."""
    v = _check_rules()(text, age)
    return v.ok, v.reason


def is_safe(text: str, age: str = AGE) -> bool:
    return verdict(text, age)[0]
