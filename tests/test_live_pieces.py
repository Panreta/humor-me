"""Protocol checks, the button source's windows, the session loader and the bank-only filter. No FastAPI needed."""
import asyncio
import json
import logging
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.bank import load_bank  # noqa: E402
from app.live import BankOnlySafety, ButtonSource, git_sha, load_sessions  # noqa: E402
from app.protocol import ProtocolError, parse  # noqa: E402

CTX = {"venue": "bar", "energy": "normal", "age": "mixed"}


def frame(obj):
    return {"text": json.dumps(obj)}


def test_valid_messages_parse():
    assert parse(frame({"type": "played"})) == {"type": "played"}
    assert parse(frame({"type": "reaction", "bit": 1, "joke": 2, "key": "Space"}))["key"] == "Space"
    start = parse(frame({"type": "start", "context": CTX, "room": "live", "operator": "me", "extra": 1}))
    assert start == {"type": "start", "context": CTX, "room": "live", "operator": "me"}


@pytest.mark.parametrize("msg", [
    {"type": "reaction", "bit": True, "joke": 0, "key": "W"},
    {"type": "reaction", "bit": -1, "joke": 0, "key": "W"},
    {"type": "reaction", "bit": 0, "joke": 0, "key": "w"},
    {"type": "start", "context": {**CTX, "age": "old"}, "room": "sim", "operator": "me"},
    {"type": "start", "context": CTX, "room": "stage", "operator": "me"},
    {"type": "start", "context": CTX, "room": "sim", "operator": " "},
    {"type": "start", "room": "sim", "operator": "me"},
    {"type": "other"}, [], 3,
])
def test_bad_messages_are_rejected(msg):
    with pytest.raises(ProtocolError):
        parse(frame(msg))


def test_non_text_and_non_json_frames_are_rejected():
    for f in ({"bytes": b"x"}, {"text": "{"}):
        with pytest.raises(ProtocolError):
            parse(f)


class Clock:
    t = 0.0

    def now(self):
        return self.t


def test_a_press_with_no_open_window_is_noted_not_queued(caplog):
    source = ButtonSource(None, Clock())
    with caplog.at_level(logging.INFO, logger="app.live"):
        source.press(0, 0, "W")
    assert source._presses.empty() and "ignored press" in caplog.text


def test_the_window_closes_at_the_end_of_the_hold():
    async def scenario():
        sent = []

        async def send(m):
            sent.append(m)

        source = ButtonSource(send, Clock())
        task = asyncio.create_task(source.reaction(0, 0, 1))
        await asyncio.sleep(0)
        source.press(0, 0, "K")
        assert await task == (2, "K", 0.0)
        assert await source.hold(0, 0, 0.01, Clock()) is None
        source.press(0, 0, "W")  # too late
        return sent, source._presses.empty()

    sent, empty = asyncio.run(scenario())
    assert [m["state"] for m in sent] == ["open", "closed"] and empty


def test_loader_skips_demo_logs_but_not_corrupt_v1_logs(tmp_path):
    good = {"schema": 1, "session": "s", "at": 1.0, "event": "rating", "value": 5}
    (tmp_path / "a.jsonl").write_text(json.dumps({"schema": 0, "session": "d", "at": 1.0, "event": "line"}) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(good) + "\n")
    assert load_sessions(tmp_path) == [[good]]
    (tmp_path / "c.jsonl").write_text("\n".join([json.dumps(good), "{torn", json.dumps(good)]) + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        load_sessions(tmp_path)


def test_bank_only_filter_passes_bank_text_and_rejects_the_rest():
    bank = load_bank()
    text = bank["neurotic"][0]["jokes"][0]["text"]
    safety = BankOnlySafety()
    assert safety.check(text, "teens").ok
    verdict = safety.check(text + " ", "teens")
    assert not verdict.ok and verdict.reason


def test_git_sha_is_a_hash_or_unknown():
    assert git_sha() == "unknown" or len(git_sha()) == 40
