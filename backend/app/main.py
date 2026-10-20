from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="humor-me")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def session(ws: WebSocket) -> None:
    # Placeholder for the show loop: UI sends operator events (context, laugh button),
    # backend will stream joke text + audio chunks back.
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            await ws.send_json({"type": "echo", "payload": msg})
    except WebSocketDisconnect:
        pass
