"""Text to speech for the show loop. Kokoro runs locally, so the demo needs no network or quota."""
import io
import wave

import numpy as np

SAMPLE_RATE = 24_000  # Kokoro's fixed output rate
VOICE = "am_michael"


def to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """16-bit mono WAV, which the browser's decodeAudioData handles without extra libraries."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_seconds(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / w.getframerate()


class KokoroTTS:
    def __init__(self, voice: str = VOICE, device: str = "cpu"):
        from kokoro import KPipeline  # slow import that pulls in torch: only when really speaking

        # CPU leaves the 8 GB GPU free for the joke model.
        self.pipeline = KPipeline(lang_code="a", device=device)
        self.voice = voice

    def synthesize(self, text: str) -> bytes:
        parts = [np.asarray(r.audio) for r in self.pipeline(text, voice=self.voice) if r.audio is not None]
        if not parts:
            raise ValueError(f"no audio produced for: {text!r}")
        return to_wav(np.concatenate(parts))
