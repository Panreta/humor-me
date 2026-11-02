import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.log import SessionLog
from app.show import LINES, play_lines

app = FastAPI(title="humor-me")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
tts_engine = None  # loading Kokoro takes seconds, so the first show pays for it, not server start


def get_tts():
    global tts_engine
    if tts_engine is None:
        from app.tts import KokoroTTS

        tts_engine = KokoroTTS()
    return tts_engine


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def session(ws: WebSocket) -> None:
    await ws.accept()
    played: asyncio.Queue = asyncio.Queue()
    show = None
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "played":
                played.put_nowait(msg)
            elif msg.get("type") == "start" and (show is None or show.done()):
                # receive_json has to keep running to hear "played", so the show runs beside it.
                tts = await asyncio.to_thread(get_tts)
                show = asyncio.create_task(
                    play_lines(LINES, tts, ws.send_json, ws.send_bytes, played.get, SessionLog()))
    except WebSocketDisconnect:
        if show is not None:
            show.cancel()
