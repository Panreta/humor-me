"""Raw room audio from the operator's browser, written straight to a WAV beside the session log. Only runs when the
operator confirmed the room was told and agreed (docs/plans/live-loop.md, "Raw audio"). No FastAPI imports."""
import asyncio
import pathlib
import struct
import time

RATE = 16_000  # mono 16-bit little-endian PCM; the browser captures at this rate, nothing is resampled here
MAX_FRAME = 64 * 1024
_HEADER = 44


class Recorder:
    """The file does not exist until the first frame arrives, so a refused or failed recording leaves nothing behind.
    The header's sizes are patched after every frame: a dropped connection or a crash leaves a WAV any reader opens,
    cut at the last whole frame."""

    def __init__(self, path: pathlib.Path, clock=time.time):
        self.path, self.clock = pathlib.Path(path), clock
        self.start_at: float | None = None  # wall clock (the log's `at` timebase) of the first sample
        self.started = asyncio.Event()  # set by the first frame or by `abandon`
        self.samples = 0
        self._file = None
        self._closed = False

    def write(self, pcm: bytes) -> None:
        if self._closed:
            raise ValueError("recording has ended")
        if not pcm or len(pcm) % 2 or len(pcm) > MAX_FRAME:
            raise ValueError("audio frames must be 16-bit PCM, non-empty and under 64 KiB")
        if self._file is None:
            # A frame is sent after its samples were captured, so the first sample is one frame older than its arrival.
            self.start_at = self.clock() - len(pcm) / 2 / RATE
            self._file = self.path.open("wb")
            self._file.write(struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 0, b"WAVE", b"fmt ", 16, 1, 1, RATE,
                                         RATE * 2, 2, 16, b"data", 0))
        self._file.write(pcm)
        self.samples += len(pcm) // 2
        end = self._file.tell()
        self._file.seek(4)
        self._file.write(struct.pack("<I", end - 8))
        self._file.seek(_HEADER - 4)
        self._file.write(struct.pack("<I", end - _HEADER))
        self._file.seek(end)
        self._file.flush()
        self.started.set()

    def abandon(self) -> None:
        """The browser could not open the microphone: release whoever waits for the first frame."""
        self.started.set()

    def close(self) -> None:
        self._closed = True
        if self._file is not None:
            self._file.close()
            self._file = None
        self.started.set()
