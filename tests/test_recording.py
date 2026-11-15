"""Consent-gated raw audio: the file exists only after a consented first frame and stays readable if the connection
drops; the logged start time is real. No microphone or browser needed."""
import importlib
import json
import pathlib
import subprocess
import sys
import wave

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import llm  # noqa: E402
from app.log import SessionLog, read_session  # noqa: E402
from app.protocol import ProtocolError, parse  # noqa: E402
from app.recording import RATE, Recorder  # noqa: E402
from sim import evaluate_clock as ev  # noqa: E402
from test_show_loop import CTX, FAST, FakeTTS, pump  # noqa: E402

main = importlib.import_module("app.main")
CHUNK = np.arange(4096, dtype="<i2").tobytes()
POLICY = {"name": "x", "epsilon": 0, "alpha": 1, "prior": "u"}


def start(consent, room="sim"):
    return {"type": "start", "context": CTX, "room": room, "operator": "t", "consent": consent}


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_tts", FakeTTS)
    monkeypatch.setattr(main, "sessions_dir", tmp_path)
    monkeypatch.setattr(main, "reaction_config", FAST)
    return tmp_path


def until(ws, kind):
    while True:
        frame = ws.receive()
        if frame.get("bytes") is not None:
            ws.send_json({"type": "played"})
            continue
        msg = json.loads(frame["text"])
        if msg["type"] == kind:
            return msg


def frames(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()


def test_no_file_exists_before_a_frame_and_the_wav_is_readable_without_close(tmp_path):
    rec = Recorder(tmp_path / "a.wav", clock=lambda: 100.0)
    assert not (tmp_path / "a.wav").exists()
    rec.write(CHUNK)
    rec.write(CHUNK)  # never closed: a crash here must leave this file
    assert frames(tmp_path / "a.wav") == (8192, RATE, 1, 2)
    assert (tmp_path / "a.wav").read_bytes()[44:48] == CHUNK[:4]
    assert rec.start_at == pytest.approx(100.0 - 4096 / RATE)


@pytest.mark.parametrize("bad", [b"", b"\x00", b"\x00" * (64 * 1024 + 2)])
def test_malformed_frames_are_refused_and_write_nothing(tmp_path, bad):
    rec = Recorder(tmp_path / "a.wav")
    with pytest.raises(ValueError):
        rec.write(bad)
    assert not (tmp_path / "a.wav").exists()


def test_a_closed_recorder_takes_no_more_audio(tmp_path):
    rec = Recorder(tmp_path / "a.wav")
    rec.write(CHUNK)
    rec.close()
    with pytest.raises(ValueError):
        rec.write(CHUNK)
    assert frames(tmp_path / "a.wav")[0] == 4096


def test_a_binary_frame_is_never_a_control_message():
    with pytest.raises(ProtocolError):
        parse({"bytes": json.dumps({"type": "played"}).encode(), "text": None})


def test_audio_without_consent_is_refused_and_no_file_appears(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_bytes(CHUNK)  # before any show
        assert "refused" in until(ws, "error")["message"]
        ws.send_json(start(False))
        ws.send_bytes(CHUNK)  # during a show that was never agreed
        assert "refused" in until(ws, "error")["message"]
        pump(ws)
    assert list(server.glob("*.wav")) == []
    assert read_session(next(server.glob("*.jsonl")))[0]["audio"] == {"consent": False, "path": None}


def test_consented_show_records_and_logs_when_the_audio_starts(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True))
        until(ws, "record")
        ws.send_bytes(CHUNK)
        ws.send_bytes(CHUNK)
        pump(ws)
    (wav,) = server.glob("*.wav")
    rows = read_session(next(server.glob("*.jsonl")))
    audio = rows[0]["audio"]
    assert audio["consent"] is True and audio["path"] == wav.name and wav.stem == rows[0]["session"]
    # sample 0 predates the first frame's arrival, which predates show_start
    assert audio["start_at"] <= rows[0]["at"] and rows[0]["at"] - audio["start_at"] < 10
    assert frames(wav) == (8192, RATE, 1, 2)


def test_a_dropped_connection_leaves_a_readable_wav_with_every_sample(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True, room="live"))  # a live room waits for keys, so the show is still running
        until(ws, "record")
        for _ in range(3):
            ws.send_bytes(CHUNK)
        until(ws, "line")
    (wav,) = server.glob("*.wav")
    assert frames(wav)[0] == 3 * 4096
    assert read_session(next(server.glob("*.jsonl")))[-1]["status"] == "aborted"


def test_a_microphone_that_failed_runs_the_show_unrecorded(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True))
        until(ws, "record")
        ws.send_json({"type": "audio_failed"})
        pump(ws)
    assert list(server.glob("*.wav")) == []
    assert read_session(next(server.glob("*.jsonl")))[0]["audio"] == {"consent": True, "path": None, "start_at": None}


def test_a_recording_must_have_a_start_time_in_the_log(tmp_path):
    log = SessionLog(tmp_path)
    fields = dict(room="sim", context=CTX, policy=POLICY, generator_id="g", selector_id="s", git_sha="x", operator="t")
    with pytest.raises(ValueError, match="start_at"):
        log.write("show_start", audio={"consent": True, "path": "a.wav"}, **fields)
    log.write("show_start", audio={"consent": True, "path": "a.wav", "start_at": 1.5}, **fields)


@pytest.mark.parametrize("name", ["abc123.wav", "abc123.jsonl"])
def test_sessions_are_gitignored(name):
    path = ROOT / "data" / "sessions" / name
    assert subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT).returncode == 0


def test_llm_without_its_settings_says_what_to_set(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(RuntimeError, match=r"LLM_BASE_URL.*--env-file"):
        llm.from_env()


def clicked_show(tmp_path, true_times, start_at, rate=16000, seconds=60):
    audio = np.random.default_rng(0).normal(0, 0.002, rate * seconds)
    for t in true_times:
        i = int((t - start_at) * rate)
        audio[i : i + 40] = 0.8
    rec = Recorder(tmp_path / "s.wav", clock=lambda: start_at + 4096 / RATE)
    pcm = (audio * 32767).astype("<i2").tobytes()
    for i in range(0, len(pcm), 8192):
        rec.write(pcm[i : i + 8192])
    log = SessionLog(tmp_path)
    log.write("show_start", room="live", context=CTX, policy=POLICY, generator_id="g", selector_id="s",
              git_sha="x", operator="t", audio={"consent": True, "path": "s.wav", "start_at": start_at})
    return log.path, tmp_path / "s.wav"


def test_click_offsets_are_recovered_at_known_times(tmp_path):
    start_at = 1_000_000.0
    true = [start_at + 3 + 5 * k for k in range(10)]
    log_path, wav = clicked_show(tmp_path, true, start_at)
    # the logged times are 30 ms earlier than the clicks really sound
    (tmp_path / "clicks.json").write_text(json.dumps([t - 0.03 for t in true]))
    out = ev.run(str(log_path), str(wav), str(tmp_path / "clicks.json"))
    assert out["n_matched"] == 10 and out["n_missed"] == 0
    assert out["median_offset_s"] == pytest.approx(0.03, abs=0.002)
    assert out["p95_abs_offset_s"] == pytest.approx(0.03, abs=0.002)
    assert out["pass_p95"] and out["pass_drift"] and abs(out["drift_over_show_s"]) < 0.002


def test_a_drifting_clock_fails_the_drift_bar_and_a_late_one_the_p95_bar(tmp_path):
    start_at = 1_000_000.0
    true = [start_at + 3 + 5 * k for k in range(10)]
    log_path, wav = clicked_show(tmp_path, true, start_at)
    drifting = [t - 0.2 * (t - true[0]) / (true[-1] - true[0]) for t in true]  # 200 ms across the show
    (tmp_path / "c.json").write_text(json.dumps(drifting))
    out = ev.run(str(log_path), str(wav), str(tmp_path / "c.json"))
    assert out["drift_over_show_s"] == pytest.approx(0.2, abs=0.005)
    assert not out["pass_drift"] and not out["pass_p95"]
