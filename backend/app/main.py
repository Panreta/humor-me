import asyncio
import contextlib
import logging
import os
import secrets
import types

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.bank import BankGenerator
from app.director import load_director
from app.generate import llm_stack
from app.live import AnnouncedSim, BankOnlySafety, ButtonSource, WebSocketPlayer, git_sha, load_sessions
from app.log import SESSIONS, SessionLog
from app.protocol import ProtocolError, parse
from app.recording import Recorder
from app.rooms import REACTION, RealClock
from app.show import UniformSelector, run_show

log = logging.getLogger(__name__)
app = FastAPI(title="humor-me")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
tts_engine = None  # loading Kokoro takes seconds, so the first show pays for it, not server start
sessions_dir = SESSIONS
reaction_config = REACTION
HEALTH_MIN = 0.99
AUDIO_WAIT_S = 15.0  # the operator may be answering the browser's microphone prompt


def get_tts():
    global tts_engine
    if tts_engine is None:
        from app.tts import KokoroTTS, SerialTTS

        tts_engine = SerialTTS(KokoroTTS())
    return tts_engine


def make_stack(room: str):
    """(generator, safety, fallback, bridge lines). `GENERATOR=llm` brings the real model and filter; otherwise the
    bank alone, which needs no key or quota (and is what the tests run)."""
    if os.environ.get("GENERATOR", "bank") == "llm":
        return llm_stack(room)
    return BankGenerator(), BankOnlySafety(), None, None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _release(ws: WebSocket, live: types.SimpleNamespace, reason: str | None = None) -> None:
    """End the recording, once: later frames are dropped without comment, and the browser is told to stop its mic."""
    recorder, live.recorder = live.recorder, None
    if recorder is None:
        return
    live.late = True
    recorder.close()
    if reason:  # the browser's notice is transient; the server log keeps why a consented show went unrecorded
        log.warning("recording ended: %s", reason)
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "record_stop", "reason": reason})


async def _open_recorder(ws: WebSocket, live: types.SimpleNamespace, log: SessionLog) -> dict:
    """The show waits for the first audio frame so `start_at` is known before `show_start` is written."""
    recorder = live.recorder = Recorder(log.path.with_suffix(".wav"))
    await ws.send_json({"type": "record"})
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(recorder.started.wait(), AUDIO_WAIT_S)
    if recorder.start_at is None:
        await _release(ws, live, recorder.reason or "no microphone audio arrived")
        return {"consent": True, "path": None, "start_at": None}
    return {"consent": True, "path": recorder.path.name, "start_at": recorder.start_at}


def _note_health(live: types.SimpleNamespace, msg: dict) -> None:
    """The browser's own count of audio produced against wall clock: diagnostics only, not in the session log."""
    live.health = msg
    if msg["elapsed_s"] > 0 and msg["produced_s"] / msg["elapsed_s"] < HEALTH_MIN:
        log.warning("browser produced %.1f s of audio in %.1f s (%.0f%%), %s silent quanta, device rate %s Hz",
                    msg["produced_s"], msg["elapsed_s"], 100 * msg["produced_s"] / msg["elapsed_s"],
                    msg["zero_quanta"], msg["rate"])


async def _audio_frame(ws: WebSocket, live: types.SimpleNamespace, data: bytes) -> None:
    recorder = live.recorder
    if recorder is None:
        if not live.late:  # a frame in flight when the recording ended is not an error
            await ws.send_json({"type": "error", "message": "audio refused: no recording was agreed for this show"})
        return
    try:
        await asyncio.to_thread(recorder.write, data)  # disk writes must not stall the show's clock
        if recorder.start_at is not None:
            recorder.started.set()  # here, not in the thread: asyncio.Event is not thread-safe
    except ValueError as e:
        await ws.send_json({"type": "error", "message": str(e)})
    except OSError as e:  # a full disk or a file locked by a sync tool must not end the show
        log.warning("recording stopped: %s", e)
        if live.recorder is recorder:
            await _release(ws, live, "the recording stopped: the disk refused the audio")


async def _show(ws: WebSocket, msg: dict, played: asyncio.Queue, live: types.SimpleNamespace) -> None:
    seed = secrets.randbits(32)  # the director's seed, logged as policy.seed; selector and sim draw their own
    try:
        sessions, sha, tts = await asyncio.gather(asyncio.to_thread(load_sessions, sessions_dir),
                                                  asyncio.to_thread(git_sha), asyncio.to_thread(get_tts))
        director = load_director(sessions, seed=seed)
        clock = RealClock()
        show_log = SessionLog(sessions_dir)
        audio = await _open_recorder(ws, live, show_log) if msg["consent"] else None
        if msg["room"] == "live":
            source = live.source = ButtonSource(ws.send_json, clock)
        else:
            source = AnnouncedSim(ws.send_json, msg["context"], secrets.randbits(32))
        generator, safety, fallback, bridge_lines = make_stack(msg["room"])
        await run_show(director, msg["room"], msg["context"], generator=generator, safety=safety,
                       selector=UniformSelector(secrets.randbits(32)), tts=tts,
                       player=WebSocketPlayer(ws.send_json, ws.send_bytes, played), source=source,
                       clock=clock, log=show_log, explore_seed=seed, operator=msg["operator"],
                       audio=audio, git_sha=sha, config=reaction_config, fallback=fallback, bridge_lines=bridge_lines)
        status = "finished"
    except Exception as e:  # cancellation is not caught: a closed socket has nobody to tell
        log.exception("show failed")
        status = "aborted"
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": f"show failed: {e}"})
    finally:
        live.source = None
        await _release(ws, live)
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "show_end", "status": status})


@app.websocket("/ws")
async def session(ws: WebSocket) -> None:
    await ws.accept()
    played: asyncio.Queue = asyncio.Queue()
    live = types.SimpleNamespace(source=None, recorder=None, late=False, health=None)  # late: audio was asked for and ended
    show = None
    try:
        while True:  # keeps running while the show does, so keys and "played" are heard
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                break
            if frame.get("bytes") is not None:  # binary is only ever audio; every control message is a text frame
                await _audio_frame(ws, live, frame["bytes"])
                continue
            try:
                msg = parse(frame)
                if msg["type"] == "start" and show is not None and not show.done():
                    raise ProtocolError("a show is already running")
            except ProtocolError as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue
            if msg["type"] == "played":
                played.put_nowait(msg)
            elif msg["type"] == "audio_health":
                _note_health(live, msg)
            elif msg["type"] == "audio_failed":
                if live.recorder is not None:
                    live.recorder.abandon(msg["reason"])
            elif msg["type"] == "reaction":
                if live.source is None:
                    log.info("ignored press %s: no live show", msg["key"])
                else:
                    live.source.press(msg["bit"], msg["joke"], msg["key"])
            else:
                live.late = False
                show = asyncio.create_task(_show(ws, msg, played, live))
    except WebSocketDisconnect:
        pass
    finally:
        if live.recorder is not None:
            live.recorder.close()
        if show is not None:
            show.cancel()  # run_show writes show_end: aborted on the way out
            await asyncio.gather(show, return_exceptions=True)
