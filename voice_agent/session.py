import asyncio
import base64
import logging
from dataclasses import asdict

from .audio import EnergyVAD
from .backends import SentenceBuffer

log = logging.getLogger(__name__)


class Session:
    def __init__(self, ws, backends, settings):
        self.ws, self.backends, self.cfg = ws, backends, settings
        self.vad = EnergyVAD(settings)
        self.history = []
        self.generation = 0
        self.reply_task = None
        self.playing_turn = None
        self.current_text = ""
        self.send_lock = asyncio.Lock()
        self.inputs = asyncio.Queue(maxsize=2)

    async def send(self, event, **fields):
        async with self.send_lock:
            await self.ws.send_json({"type": event, **fields})

    @property
    def speaking(self):
        return self.playing_turn is not None or (self.reply_task is not None and not self.reply_task.done())

    async def cancel(self, reason="manual"):
        old = self.generation
        self.generation += 1
        self.playing_turn = None
        if self.reply_task is not None:
            self.reply_task.cancel()
            await asyncio.gather(self.reply_task, return_exceptions=True)
            self.reply_task = None
        await self.send("turn.cancelled", turn_id=old, reason=reason)

    async def enqueue(self, item):
        try:
            self.inputs.put_nowait(item)
        except asyncio.QueueFull:
            await self.send("error", message="Input queue is full. Please try again shortly.")

    async def handle_inputs(self):
        while True:
            item = await self.inputs.get()
            try:
                # A manual interrupt/reset invalidates any ASR already in flight.
                generation = self.generation
                text = item if isinstance(item, str) else await self.backends.transcribe(item)
                if generation != self.generation:
                    continue
                text = text.strip()
                if not text:
                    continue
                await self.send("transcript", text=text)
                busy = self.speaking
                decision = await self.backends.decide(text, self.history, busy, self.current_text)
                if generation != self.generation:
                    continue
                await self.send("decision", **asdict(decision))
                if busy and not decision.interrupt:
                    await self.send("input.ignored", reason="keep_speaking")
                    continue
                if not decision.respond and not decision.interrupt:
                    await self.send("input.ignored", reason="no_reply_needed")
                    continue
                await self.cancel("barge_in" if busy else "new_turn")
                if not decision.respond:
                    continue
                self.history.append({"role": "user", "content": text})
                starts = [i for i, message in enumerate(self.history) if message["role"] == "user"]
                if len(starts) > self.cfg.max_history_turns:
                    self.history = self.history[starts[-self.cfg.max_history_turns]:]
                self.current_text = ""
                self.reply_task = asyncio.create_task(self.respond(self.generation))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Input processing failed")
                await self.send("error", message="Speech recognition or turn detection failed. Check the service logs.")
            finally:
                self.inputs.task_done()

    async def respond(self, generation):
        queue = asyncio.Queue(maxsize=4)
        spoken = []
        producer = None
        try:
            await self.send("turn.start", turn_id=generation)

            async def produce():
                buffer = SentenceBuffer()
                messages = [{"role": "system", "content": self.cfg.system_prompt}] + list(self.history)
                async for delta in self.backends.llm(messages):
                    if generation != self.generation:
                        return
                    self.current_text += delta
                    await self.send("text.delta", turn_id=generation, text=delta)
                    for sentence in buffer.feed(delta):
                        await queue.put(sentence)
                tail = buffer.flush()
                if tail:
                    await queue.put(tail)
                await queue.put(None)

            producer = asyncio.create_task(produce())
            # Wait on producer errors as well as queue input, so failures never
            # leave the TTS consumer blocked waiting for a sentinel.
            while True:
                get_task = asyncio.create_task(queue.get())
                try:
                    done, _ = await asyncio.wait({get_task, producer}, return_when=asyncio.FIRST_COMPLETED)
                    if producer in done and producer.exception() is not None:
                        raise producer.exception()
                    sentence = await get_task
                finally:
                    if not get_task.done():
                        get_task.cancel()
                    await asyncio.gather(get_task, return_exceptions=True)
                if sentence is None:
                    break
                first = True
                async for rate, pcm in self.backends.synthesize(sentence):
                    if generation != self.generation:
                        return
                    if first:
                        spoken.append(sentence)
                        first = False
                    await self.send("audio.chunk", turn_id=generation, sample_rate=rate,
                                    audio=base64.b64encode(pcm).decode("ascii"))
                if first:
                    raise ValueError("TTS returned no audio")
            await producer
            if not self.current_text.strip():
                raise ValueError("LLM returned empty text")
            await self.send("turn.done", turn_id=generation)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Reply failed")
            await self.send("error", turn_id=generation,
                            message="Reply generation failed. Check the LLM configuration and TTS service logs.")
            await self.send("turn.failed", turn_id=generation)
        finally:
            if producer is not None:
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            if spoken:
                self.history.append({"role": "assistant", "content": "".join(spoken)})

    async def run(self):
        worker = asyncio.create_task(self.handle_inputs())
        await self.send("ready", sample_rate=16000, format="pcm_s16le", mock=self.cfg.mock)
        try:
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    if len(data) > 32000 or len(data) % 2:
                        await self.send("error", message="Audio must use PCM16, with no more than one second per message.")
                        continue
                    for kind, pcm in self.vad.feed(data):
                        await self.send("vad." + kind)
                        if pcm:
                            await self.enqueue(pcm)
                    continue
                import json
                try:
                    event = json.loads(message.get("text", ""))
                    if not isinstance(event, dict):
                        raise ValueError()
                except (ValueError, TypeError):
                    await self.send("error", message="Invalid JSON message.")
                    continue
                kind = event.get("type")
                if kind == "text":
                    text = event.get("text")
                    if isinstance(text, str) and 0 < len(text.strip()) <= 2000:
                        await self.enqueue(text.strip())
                    else:
                        await self.send("error", message="Text must contain 1–2000 characters.")
                elif kind == "audio.commit":
                    pcm = self.vad.flush()
                    if pcm:
                        await self.enqueue(pcm)
                elif kind in {"interrupt", "reset"}:
                    await self.cancel(kind)
                    self.vad = EnergyVAD(self.cfg)
                    while not self.inputs.empty():
                        self.inputs.get_nowait()
                        self.inputs.task_done()
                    if kind == "reset":
                        self.history.clear()
                        self.current_text = ""
                elif kind == "playback.start" and event.get("turn_id") == self.generation:
                    self.playing_turn = self.generation
                elif kind == "playback.end" and event.get("turn_id") == self.playing_turn:
                    self.playing_turn = None
                elif kind == "ping":
                    await self.send("pong")
        finally:
            worker.cancel()
            if self.reply_task is not None:
                self.reply_task.cancel()
            await asyncio.gather(worker, *([self.reply_task] if self.reply_task else []), return_exceptions=True)
