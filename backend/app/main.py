import asyncio
import contextlib
import logging
import secrets
import types

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.bank import BankGenerator
from app.director import load_director
from app.live import AnnouncedSim, BankOnlySafety, ButtonSource, Voice, WebSocketPlayer, git_sha, load_sessions
from app.log import SESSIONS, SessionLog
from app.protocol import ProtocolError, parse
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


def get_tts():
    global tts_engine
    if tts_engine is None:
        from app.tts import KokoroTTS

        tts_engine = KokoroTTS()
    return tts_engine


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


async def _show(ws: WebSocket, msg: dict, played: asyncio.Queue, live: types.SimpleNamespace) -> None:
    seed = secrets.randbits(32)  # the director's seed, logged as policy.seed; selector and sim draw their own
    try:
        sessions, sha, tts = await asyncio.gather(asyncio.to_thread(load_sessions, sessions_dir),
                                                  asyncio.to_thread(git_sha), asyncio.to_thread(get_tts))
        director = load_director(sessions, seed=seed)
        clock = RealClock()
        if msg["room"] == "live":
            source = live.source = ButtonSource(ws.send_json, clock)
        else:
            source = AnnouncedSim(ws.send_json, msg["context"], secrets.randbits(32))
        voice = Voice(tts)
        await run_show(director, msg["room"], msg["context"], generator=BankGenerator(), safety=BankOnlySafety(),
                       selector=UniformSelector(secrets.randbits(32)), tts=voice,
                       player=WebSocketPlayer(voice, ws.send_json, ws.send_bytes, played), source=source,
                       clock=clock, log=SessionLog(sessions_dir), explore_seed=seed, operator=msg["operator"],
                       git_sha=sha, config=reaction_config)
        status = "finished"
    except Exception as e:  # cancellation is not caught: a closed socket has nobody to tell
        log.exception("show failed")
        status = "aborted"
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": f"show failed: {e}"})
    finally:
        live.source = None
    with contextlib.suppress(Exception):
        await ws.send_json({"type": "show_end", "status": status})


@app.websocket("/ws")
async def session(ws: WebSocket) -> None:
    await ws.accept()
    played: asyncio.Queue = asyncio.Queue()
    live = types.SimpleNamespace(source=None)
    show = None
    try:
        while True:  # keeps running while the show does, so keys and "played" are heard
            frame = await ws.receive()
            if frame["type"] == "websocket.disconnect":
                break
            try:
                msg = parse(frame)
                if msg["type"] == "start" and show is not None and not show.done():
                    raise ProtocolError("a show is already running")
            except ProtocolError as e:
                await ws.send_json({"type": "error", "message": str(e)})
                continue
            if msg["type"] == "played":
                played.put_nowait(msg)
            elif msg["type"] == "reaction":
                if live.source is None:
                    log.info("ignored press %s: no live show", msg["key"])
                else:
                    live.source.press(msg["bit"], msg["joke"], msg["key"])
            else:
                show = asyncio.create_task(_show(ws, msg, played, live))
    except WebSocketDisconnect:
        pass
    finally:
        if show is not None:
            show.cancel()  # run_show writes show_end: aborted on the way out
            await asyncio.gather(show, return_exceptions=True)
