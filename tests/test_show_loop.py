"""The show over the websocket, with a fake voice; no Kokoro, audio device or browser needed."""
import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.log import read_session  # noqa: E402
from app.rooms import ReactionConfig  # noqa: E402
from app.tts import to_wav, wav_seconds  # noqa: E402
from sim.env import AudienceEnv  # noqa: E402

main = importlib.import_module("app.main")
CTX = {"venue": "bar", "energy": "normal", "age": "mixed"}
FAST = ReactionConfig(holds={3: 0.003, 2: 0.002, 1: 0.001, 0: 0.001, None: 0.0}, timeout_s=0.5)
# Written out here, not imported, so a wrong table in app code fails these tests.
LEVEL_OF = {"W": 3, "K": 2, "Space": 1, "D": 0}


class FakeTTS:
    def synthesize(self, text):
        return to_wav(np.zeros(2400))


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_tts", FakeTTS)
    monkeypatch.setattr(main, "sessions_dir", tmp_path)
    monkeypatch.setattr(main, "reaction_config", FAST)
    return tmp_path


def start(room, operator="t"):
    return {"type": "start", "context": CTX, "room": room, "operator": operator}


def pump(ws, on_open=None, on_closed=None, stop=lambda m: m["type"] == "show_end"):
    """Play the browser: acknowledge every audio blob, hand each window event to a callback, return what arrived."""
    seen = []
    while True:
        frame = ws.receive()
        if frame.get("bytes") is not None:
            assert frame["bytes"].startswith(b"RIFF")
            ws.send_json({"type": "played"})
            continue
        msg = json.loads(frame["text"])
        seen.append(msg)
        if msg["type"] == "reaction_window":
            callback = on_open if msg["state"] == "open" else on_closed
            if callback:
                callback(msg)
        if stop(msg):
            return seen


def only_log(tmp_path):
    (path,) = tmp_path.glob("*.jsonl")
    return read_session(path)


def reactions(rows):
    return [r for r in rows if r["event"] == "reaction"]


def press(ws, msg, key):
    ws.send_json({"type": "reaction", "bit": msg["bit"], "joke": msg["joke"], "key": key})


def test_wav_round_trip_reports_its_length():
    assert abs(wav_seconds(to_wav(np.zeros(12_000))) - 0.5) < 1e-9


def test_loud_audio_is_clipped_rather_than_wrapped_around():
    wav = to_wav(np.array([2.0, -2.0]))
    assert int.from_bytes(wav[-4:-2], "little", signed=True) == 32767
    assert int.from_bytes(wav[-2:], "little", signed=True) == -32767


def test_health_still_works():
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}


def test_a_whole_simulated_show_logs_40_reactions(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        seen = pump(ws)
    rows = only_log(server)
    assert seen[-1] == {"type": "show_end", "status": "finished"}
    assert len(reactions(rows)) == 40 and {r["source"] for r in reactions(rows)} == {"sim"}
    assert rows[0]["room"] == "sim" and rows[0]["generator_id"] == "bank-v1" and rows[0]["operator"] == "t"
    assert isinstance(rows[0]["policy"]["seed"], int) and rows[0]["git_sha"]
    assert rows[-1]["status"] == "finished"
    keys = [m["key"] for m in seen if m["type"] == "sim_reaction"]
    assert keys == [r["key"] for r in reactions(rows)] and len(keys) == 40
    first = next(r for r in rows if r["event"] == "candidates")
    assert first["prompt_id"].startswith("bank-v1:") and first["prompt"] == "bank-v1.yaml"
    assert first["items"][0]["verdict"] == "pass"


def test_a_live_show_answered_with_keys_logs_the_frozen_levels(server):
    order = ["W", "K", "Space", "D"]
    answered = []

    def answer(msg):
        key = order[len(answered) % 4]
        answered.append((msg["bit"], msg["joke"]))
        press(ws, msg, key)

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        seen = pump(ws, on_open=answer)
    rows = only_log(server)
    got = reactions(rows)
    assert seen[-1]["status"] == "finished" and len(got) == 40
    assert [(r["key"], r["level"]) for r in got] == [(order[i % 4], LEVEL_OF[order[i % 4]]) for i in range(40)]
    assert {r["source"] for r in got} == {"button"} and all(r["latency_s"] >= 0 for r in got)
    assert answered == [(b, j) for b in range(8) for j in range(5)]
    assert [r["planned_hold_s"] for r in rows if r["event"] == "gap"][:4] == [0.003, 0.002, 0.001, 0.001]


def test_the_last_press_wins_during_the_hold_and_the_hold_restarts(server, monkeypatch):
    monkeypatch.setattr(main, "reaction_config", ReactionConfig(holds={3: 0.02, 1: 0.5, 2: 0.02, 0: 0.02}, timeout_s=2))
    fired = []

    def two_presses(msg):
        if not fired:
            fired.append(msg)
            press(ws, msg, "W")
            press(ws, msg, "Space")

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        # Stop at the first line after the window: joke 0's gap row is written by then.
        pump(ws, on_open=two_presses, stop=lambda m: m["type"] == "line" and bool(fired))
    rows = only_log(server)
    first = [r for r in reactions(rows) if r["joke"] == 0 and r["bit"] == 0]
    assert [(r["key"], r["level"]) for r in first] == [("W", 3), ("Space", 1)]
    (gap,) = [r for r in rows if r["event"] == "gap" and r["bit"] == 0 and r["joke"] == 0]
    assert gap["planned_hold_s"] == 0.5 and gap["actual_gap_s"] >= 0.45  # restarted at Space's hold, not W's


def test_a_late_or_misdirected_press_is_not_a_reaction_and_changes_no_reward(server):
    sent = []

    def wrong_then_right(msg):
        ws.send_json({"type": "reaction", "bit": msg["bit"], "joke": (msg["joke"] + 1) % 5, "key": "D"})
        press(ws, msg, "K")

    def late(msg):
        press(ws, msg, "D")  # the window has just closed
        sent.append(msg)

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        pump(ws, on_open=wrong_then_right, on_closed=late)
    rows = only_log(server)
    assert len(sent) == 40
    assert [r["level"] for r in reactions(rows)] == [2] * 40
    assert [r["reward"] for r in rows if r["event"] == "bit_end"] == pytest.approx([2 / 3] * 8)


def test_a_timeout_writes_a_null_reaction(server):
    seen_open = []

    def ignore(msg):
        seen_open.append(msg)

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        pump(ws, on_open=ignore, stop=lambda m: m["type"] == "reaction_window" and m["state"] == "closed")
        pump(ws, on_open=ignore, stop=lambda m: m["type"] == "line")
    rows = only_log(server)
    first = reactions(rows)[0]
    assert (first["source"], first["level"], first["key"]) == ("button", None, None)
    assert [r["planned_hold_s"] for r in rows if r["event"] == "gap"][0] == 0.0


def test_a_malformed_message_gets_an_error_and_the_show_goes_on(server):
    junk = [lambda: ws.send_text("not json"), lambda: ws.send_json({"type": "nope"}),
            lambda: ws.send_json({"type": "reaction", "bit": "0", "joke": 0, "key": "W"}),
            lambda: ws.send_json({"type": "reaction", "bit": 0, "joke": 0, "key": "X"}),
            lambda: ws.send_json({"type": "reaction", "bit": 0, "joke": 0, "key": ["W"]}),
            lambda: ws.send_json({"type": "reaction", "bit": 0, "joke": 0, "key": {"W": 1}}),
            lambda: ws.send_json([1, 2]), lambda: ws.send_bytes(b"\x00")]
    errors = []
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "room": "live"})
        errors.append(ws.receive_json()["type"])
        ws.send_json(start("live"))
        opened = pump(ws, stop=lambda m: m["type"] == "reaction_window")[-1]
        for send in junk:
            send()
            errors.append(ws.receive_json()["type"])
        ws.send_json(start("live"))
        errors.append(ws.receive_json()["type"])  # a second show may not start beside the first
        press(ws, opened, "K")
        pump(ws, stop=lambda m: m["type"] == "line")
    assert errors == ["error"] * 10
    assert len(reactions(only_log(server))) == 1


def test_a_disconnect_writes_show_end_aborted(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        pump(ws, stop=lambda m: m["type"] == "reaction_window")
    rows = only_log(server)
    assert rows[-1]["event"] == "show_end" and rows[-1]["status"] == "aborted"


def test_a_corrupt_session_log_fails_the_start_visibly(server):
    good = {"schema": 1, "session": "s", "at": 1.0, "event": "rating", "value": 5}
    (server / "bad.jsonl").write_text("\n".join([json.dumps(good), "{torn", json.dumps(good)]) + "\n")
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        first, second = ws.receive_json(), ws.receive_json()
    assert first["type"] == "error" and "corrupt" in first["message"]
    assert second == {"type": "show_end", "status": "aborted"}
    assert len(list(server.glob("*.jsonl"))) == 1  # no new show started


def test_the_mode_never_reaches_the_client(server):
    fields = {"line": {"type", "text"}, "reaction_window": {"type", "state", "bit", "joke"},
              "sim_reaction": {"type", "bit", "joke", "key"}, "show_end": {"type", "status"}}
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        seen = pump(ws)
    assert {m["type"] for m in seen} == set(fields)
    for m in seen:
        assert set(m) == fields[m["type"]]
        rest = json.dumps({k: v for k, v in m.items() if k != "text"})
        assert not [mode for mode in AudienceEnv().modes if mode in rest]


def test_the_selector_and_the_director_get_different_seeds(server, monkeypatch):
    counter = iter(range(1000, 2000))
    monkeypatch.setattr(main.secrets, "randbits", lambda n: next(counter))
    seeds = {}
    real_director, real_selector = main.load_director, main.UniformSelector

    def spy_director(sessions, seed=None):
        seeds["director"] = seed
        return real_director(sessions, seed=seed)

    def spy_selector(seed=None):
        seeds["selector"] = seed
        return real_selector(seed)

    monkeypatch.setattr(main, "load_director", spy_director)
    monkeypatch.setattr(main, "UniformSelector", spy_selector)
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        pump(ws, stop=lambda m: m["type"] == "line")
    rows = only_log(server)
    assert seeds["director"] != seeds["selector"] and rows[0]["policy"]["seed"] == seeds["director"]
