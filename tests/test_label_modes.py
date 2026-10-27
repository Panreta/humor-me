"""Parser for local-LLM label output; no model needed."""
from explorer.label_modes import PROMPT, SAFETY, parse


def test_parses_json_inside_chatter():
    d = parse('Sure! {"mode": "Cynic", "mode2": "observer", "sexual": true}')
    assert d["mode"] == "cynic" and d["mode2"] == "observer"
    assert d["safety"]["sexual"] is True and d["safety"]["slur"] is False


def test_unknown_mode_is_a_failure_not_a_guess():
    assert parse('{"mode": "dad joke"}') is None
    assert parse("no json here") is None


def test_mode2_equal_to_mode_is_dropped():
    assert parse('{"mode": "absurdist", "mode2": "absurdist"}')["mode2"] is None


def test_prompt_lists_every_safety_key():
    assert all(f'"{k}"' in PROMPT for k in SAFETY)
    assert "{joke}" in PROMPT
