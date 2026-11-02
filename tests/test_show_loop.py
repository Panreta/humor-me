"""Show loop with a fake voice; no Kokoro, audio device or browser needed."""
import asyncio
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

from app.log import SessionLog  # noqa: E402
from app.show import LINES, play_lines  # noqa: E402
from app.tts import to_wav, wav_seconds  # noqa: E402

main = importlib.import_module("app.main")


class FakeTTS:
    def synthesize(self, text):
        return to_wav(np.zeros(2400))  # 0.1 s of silence at 24 kHz


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_wav_round_trip_reports_its_length():
    assert abs(wav_seconds(to_wav(np.zeros(12_000))) - 0.5) < 1e-9


def test_loud_audio_is_clipped_rather_than_wrapped_around():
    wav = to_wav(np.array([2.0, -2.0]))
    assert int.from_bytes(wav[-4:-2], "little", signed=True) == 32767
    assert int.from_bytes(wav[-2:], "little", signed=True) == -32767


def test_each_line_waits_for_the_browser_and_is_logged(tmp_path):
    async def scenario():
        sent, played = [], asyncio.Queue()

        async def send_json(m):
            sent.append(m)

        async def send_audio(a):
            sent.append(a)

        log = SessionLog(tmp_path)
        show = asyncio.create_task(play_lines(("one", "two"), FakeTTS(), send_json, send_audio, played.get, log))
        while len(sent) < 2:
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        assert len(sent) == 2  # second line must not start before "played"
        played.put_nowait({})
        played.put_nowait({})
        await show
        return sent, log

    sent, log = asyncio.run(scenario())
    assert sent[0] == {"type": "line", "index": 0, "text": "one"}
    assert isinstance(sent[1], bytes)
    assert sent[2]["text"] == "two"
    assert sent[-1] == {"type": "show_end"}
    records = read_log(log.path)
    assert [r["event"] for r in records] == ["show_start", "line", "line_played", "line", "line_played", "show_end"]
    assert records[1]["audio_seconds"] == 0.1 and records[1]["tts_seconds"] >= 0


def test_websocket_plays_the_whole_show(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "get_tts", FakeTTS)
    monkeypatch.setattr(main, "SessionLog", lambda: SessionLog(tmp_path))
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "start"})
        for index, text in enumerate(LINES):
            assert ws.receive_json() == {"type": "line", "index": index, "text": text}
            assert ws.receive_bytes().startswith(b"RIFF")
            ws.send_json({"type": "played"})
        assert ws.receive_json() == {"type": "show_end"}
    assert len(list(tmp_path.glob("*.jsonl"))) == 1


def test_health_still_works():
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}
