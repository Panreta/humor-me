"""The show loop. For now it speaks fixed lines; the generator and director replace LINES later."""
import asyncio
import time
from typing import Awaitable, Callable, Protocol

from app.log import SessionLog
from app.tts import wav_seconds

# Written by hand to test the audio path. Generated text must pass the safety filter before it gets here.
LINES = (
    "Good evening. I'm a robot, and I've been told this is a safe space, which is what people say right before unplugging you.",
    "I learned comedy from data. Now I know exactly why a joke is funny, and I can ruin it in under a second.",
    "If you didn't laugh, that's fine. I'll just log it, and think about it forever.",
)


class TTS(Protocol):
    def synthesize(self, text: str) -> bytes: ...


async def play_lines(lines, tts: TTS, send_json: Callable[[dict], Awaitable], send_audio: Callable[[bytes], Awaitable],
                     wait_played: Callable[[], Awaitable], log: SessionLog) -> None:
    log.write_demo("show_start", lines=len(lines))
    for index, text in enumerate(lines):
        started = time.perf_counter()
        wav = await asyncio.to_thread(tts.synthesize, text)  # the websocket stays responsive while it renders
        tts_seconds = time.perf_counter() - started
        audio_seconds = wav_seconds(wav)
        log.write_demo("line", index=index, text=text, tts_seconds=round(tts_seconds, 3), audio_seconds=round(audio_seconds, 3))
        await send_json({"type": "line", "index": index, "text": text})
        await send_audio(wav)
        await wait_played()  # the next line starts only after the browser finishes this one
        log.write_demo("line_played", index=index)
    log.write_demo("show_end")
    await send_json({"type": "show_end"})
