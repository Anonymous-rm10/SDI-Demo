import io
import wave
from collections import deque

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
FRAME_BYTES = FRAME_SAMPLES * 2


def pcm16(samples):
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return stream.getvalue()


def read_wav(data: bytes, max_seconds=30):
    with wave.open(io.BytesIO(data), "rb") as wav:
        rate = wav.getframerate()
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError("Expected mono PCM16 WAV")
        if rate < 8000 or rate > 48000 or wav.getnframes() > rate * max_seconds:
            raise ValueError("Invalid sample rate or audio too long")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if not audio.size:
        raise ValueError("Empty audio")
    return audio, rate


class EnergyVAD:
    """20 ms RMS frames, pre-roll, silence endpoint and bounded utterances."""

    def __init__(self, settings):
        self.cfg = settings
        self.pending = b""
        self.preroll = deque(maxlen=(settings.vad_preroll_ms + 19) // 20)
        self.frames = []
        self.voiced_ms = self.silence_ms = 0

    def feed(self, data: bytes):
        events = []
        self.pending += data
        while len(self.pending) >= FRAME_BYTES:
            frame, self.pending = self.pending[:FRAME_BYTES], self.pending[FRAME_BYTES:]
            audio = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768
            voiced = float(np.sqrt(np.mean(audio * audio))) >= self.cfg.vad_threshold
            if not self.frames:
                if not voiced:
                    self.preroll.append(frame)
                    continue
                self.frames = list(self.preroll)
                self.preroll.clear()
                events.append(("start", None))
            self.frames.append(frame)
            self.voiced_ms += 20 if voiced else 0
            self.silence_ms = 0 if voiced else self.silence_ms + 20
            if (self.silence_ms >= self.cfg.vad_silence_ms or
                    len(self.frames) * 20 >= self.cfg.vad_max_utterance_ms):
                events.append(("end", self.flush()))
        return events

    def flush(self):
        # Sub-frame remainder is kept until the next frame; at most 19 ms.
        output = b"".join(self.frames) if self.voiced_ms >= self.cfg.vad_min_speech_ms else None
        self.frames = []
        self.voiced_ms = self.silence_ms = 0
        self.preroll.clear()
        return output
