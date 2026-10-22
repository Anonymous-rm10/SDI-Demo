"""Model workers run in separate Python environments; no shared Transformers import."""
import asyncio
import concurrent.futures
import logging
import os
import threading
import wave
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .audio import pcm16, read_wav
from .config import load_env
from .paths import project_path, use_source

log = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 16000 * 2 * 30 + 4096


async def limited_body(request, limit):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(413, "Request too large")
    return bytes(data)


def asr_app(mock=False):
    load_env()
    model = None
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        nonlocal model
        if not mock:
            import torch
            use_source("ASR_REPO")
            from qwen_asr import Qwen3ASRModel
            path = project_path("ASR_MODEL")
            if not path.is_dir():
                raise RuntimeError(f"ASR_MODEL directory not found: {path}")
            dtype = os.getenv("ASR_DTYPE", "bfloat16")
            if dtype not in {"bfloat16", "float16", "float32"}:
                raise ValueError("ASR_DTYPE must be bfloat16, float16 or float32")
            model = Qwen3ASRModel.from_pretrained(
                str(path), dtype=getattr(torch, dtype), device_map=os.getenv("ASR_DEVICE", "cuda:0"),
                attn_implementation="sdpa", max_inference_batch_size=1,
                max_new_tokens=int(os.getenv("ASR_MAX_NEW_TOKENS", "256")),
            )
        yield

    app = FastAPI(title="Qwen3-ASR worker", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ok": mock or model is not None, "backend": "mock" if mock else "qwen3-asr"}

    @app.post("/transcribe")
    async def transcribe(request: Request):
        data = await limited_body(request, MAX_AUDIO_BYTES)
        try:
            samples, rate = read_wav(data)
        except (ValueError, EOFError, wave.Error) as exc:
            raise HTTPException(400, str(exc)) from exc

        def infer():
            with lock:
                if mock:
                    return {"text": "Hello, please introduce yourself.", "language": "English"}
                result = model.transcribe(audio=(samples, rate),
                                          language=os.getenv("ASR_LANGUAGE") or None)[0]
                return {"text": result.text, "language": result.language}

        return await asyncio.to_thread(infer)

    return app


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


def tts_app(mock=False):
    load_env()
    model = None
    rate = 24000
    lock = threading.Lock()
    prompt_wav = project_path("TTS_PROMPT_WAV")
    prompt_text = os.getenv("TTS_PROMPT_TEXT") or "You are a helpful assistant.<|endofprompt|>\u5e0c\u671b\u4f60\u4ee5\u540e\u80fd\u591f\u505a\u7684\u6bd4\u6211\u8fd8\u597d\u5466\u3002"

    @asynccontextmanager
    async def lifespan(app):
        nonlocal model, rate
        if not mock:
            use_source("TTS_REPO")
            from cosyvoice.cli.cosyvoice import AutoModel
            path = project_path("TTS_MODEL")
            if not path.is_dir() or not prompt_wav.is_file():
                raise RuntimeError("TTS_MODEL / TTS_PROMPT_WAV not found")
            model = AutoModel(model_dir=str(path), load_trt=False, load_vllm=False,
                              fp16=os.getenv("TTS_FP16", "false").lower() in {"true", "1"})
            rate = int(model.sample_rate)
            # Cache only in memory. Do not modify the model's spk2info.pt.
            model.add_zero_shot_spk(prompt_text, str(prompt_wav), "voice-agent")
        yield

    app = FastAPI(title="CosyVoice worker", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ok": mock or model is not None, "backend": "mock" if mock else "cosyvoice",
                "sample_rate": rate}

    @app.post("/synthesize")
    async def synthesize(body: SynthesisRequest):
        if not body.text.strip():
            raise HTTPException(422, "Text is blank")
        # Producer retains the model lock until an in-flight GPU call finishes.
        # Cancellation stops queueing, then closes the model iterator safely.
        async def stream():
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue(maxsize=4)
            stop = threading.Event()

            def put(value):
                future = asyncio.run_coroutine_threadsafe(queue.put(value), loop)
                while not stop.is_set():
                    try:
                        future.result(timeout=0.1)
                        return True
                    except concurrent.futures.TimeoutError:
                        pass
                future.cancel()
                return False

            def produce():
                acquired = False
                iterator = None
                try:
                    while not stop.is_set():
                        if lock.acquire(timeout=0.1):
                            acquired = True
                            break
                    if not acquired or stop.is_set():
                        return
                    if mock:
                        import numpy as np
                        iterator = iter([pcm16(0.08 * np.sin(2 * np.pi * 440 * np.arange(rate // 5) / rate))])
                    else:
                        iterator = model.inference_zero_shot(
                            body.text, "", "", zero_shot_spk_id="voice-agent", stream=True,
                            speed=float(os.getenv("TTS_SPEED", "1")),
                        )
                    for chunk in iterator:
                        if stop.is_set():
                            break
                        data = chunk if mock else pcm16(chunk["tts_speech"].detach().float().cpu().numpy().reshape(-1))
                        # Bound WS messages and keep cancellation responsive.
                        for pos in range(0, len(data), 8192):
                            if not put(data[pos:pos + 8192]):
                                return
                except Exception as exc:
                    log.exception("TTS inference failed")
                    if not stop.is_set():
                        put(exc)
                finally:
                    try:
                        if iterator is not None and hasattr(iterator, "close"):
                            iterator.close()
                    finally:
                        if acquired:
                            lock.release()
                    if not stop.is_set():
                        put(None)

            thread = threading.Thread(target=produce, daemon=True, name="tts-producer")
            thread.start()
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    if isinstance(item, Exception):
                        # Abort the HTTP stream; the gateway reports the failure.
                        raise RuntimeError("TTS inference failed") from item
                    yield item
            finally:
                stop.set()

        return StreamingResponse(stream(), media_type="application/octet-stream", headers={
            "X-Sample-Rate": str(rate), "X-Audio-Format": "pcm_s16le", "X-Channels": "1",
        })

    return app
