import asyncio
import json
import unittest

import httpx
import numpy as np
from fastapi.testclient import TestClient

from voice_agent.audio import EnergyVAD, pcm16, read_wav, wav_bytes
from voice_agent.backends import Backends, Decision, SentenceBuffer
from voice_agent.config import Settings
from voice_agent.server import create_app
from voice_agent.session import Session
from voice_agent.workers import asr_app, tts_app


class AudioTests(unittest.TestCase):
    def test_wav_round_trip(self):
        original = np.sin(np.arange(1600) / 10) * 0.2
        samples, rate = read_wav(wav_bytes(pcm16(original)))
        self.assertEqual(rate, 16000)
        np.testing.assert_allclose(samples, original, atol=0.0001)

    def test_reject_stereo_or_empty(self):
        with self.assertRaises(ValueError):
            read_wav(wav_bytes(b""))

    def test_vad_endpoint_and_preroll_with_arbitrary_packets(self):
        cfg = Settings(vad_min_speech_ms=200, vad_silence_ms=300)
        audio = pcm16(np.concatenate([np.zeros(3200), np.full(4800, 0.1), np.zeros(6400)]))
        vad = EnergyVAD(cfg)
        events = []
        for pos in range(0, len(audio), 274):
            events.extend(vad.feed(audio[pos:pos + 274]))
        self.assertEqual([event[0] for event in events], ["start", "end"])
        self.assertIsNotNone(events[-1][1])
        self.assertEqual(events[-1][1][:6400], bytes(6400))

    def test_vad_discards_click_and_bounds_long_speech(self):
        vad = EnergyVAD(Settings(vad_min_speech_ms=200, vad_max_utterance_ms=1000))
        events = vad.feed(pcm16(np.concatenate([np.full(320, .1), np.zeros(16000)])))
        self.assertIsNone(events[-1][1])
        events = vad.feed(pcm16(np.full(32000, .1)))
        self.assertGreaterEqual(sum(kind == "end" for kind, _ in events), 2)

    def test_sentence_buffer_preserves_content(self):
        buffer = SentenceBuffer(limit=25)
        text = "\u4eca\u5929\u7684\u6e29\u5ea6\u662f23.5\u5ea6\u3002Hello world. \u8fd9\u662f\u4e0b\u4e00\u53e5\uff01\u6700\u540e\u4e00\u70b9"
        segments = []
        for part in [text[:11], text[11:20], text[20:]]:
            segments += buffer.feed(part)
        segments.append(buffer.flush())
        self.assertEqual("".join(segments).replace(" ", ""), text.replace(" ", ""))
        self.assertIn("23.5", "".join(segments))
        self.assertTrue(all(len(part) <= 25 for part in segments))


class SplitStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b': heartbeat\n\ndata: {"choices":[{"delta":{"content":"Hel'
        yield b'lo"}}]}\n\ndata: {"choices":[]}\n\ndata: [DONE]\n\n'


class BackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_chat_and_authorization(self):
        def handler(request):
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            self.assertTrue(json.loads(request.content)["stream"])
            return httpx.Response(200, stream=SplitStream())
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        backend = Backends(Settings(llm_base_url="http://llm/v1", llm_api_key="test-key"), client)
        self.assertEqual([x async for x in backend.llm([])], ["Hello"])
        await backend.close()

    async def test_jev_schema_and_probability_threshold(self):
        def handler(request):
            data = json.loads(request.content)
            self.assertEqual(request.url.path, "/v1/systemone")
            self.assertIn("interrupt", data["questions"])
            self.assertTrue(data["state"]["assistant_speaking"])
            return httpx.Response(200, json={"answers": {"respond": {"noul": .9}, "interrupt": {"noul": .2}}})
        backend = Backends(Settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        decision = await backend.decide("Uh-huh", [], True)
        self.assertTrue(decision.respond)
        self.assertFalse(decision.interrupt)
        self.assertEqual(decision.source, "jev")
        await backend.close()

    async def test_jev_failure_preserves_playback_and_allows_idle(self):
        for response in [httpx.Response(503), httpx.Response(200, json={}),
                         httpx.Response(200, json={"answers": {"respond": {"noul": 2}, "interrupt": {"noul": .8}}})]:
            backend = Backends(Settings(), httpx.AsyncClient(transport=httpx.MockTransport(lambda req: response)))
            self.assertFalse((await backend.decide("A new question", [], True)).interrupt)
            self.assertTrue((await backend.decide("A new question", [], False)).respond)
            await backend.close()

    async def test_pcm_http_stream_alignment(self):
        class PCMStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for item in [b'\x01', b'\x02\x03', b'\x04']:
                    yield item
        backend = Backends(Settings(), httpx.AsyncClient(transport=httpx.MockTransport(
            lambda req: httpx.Response(200, headers={"X-Sample-Rate": "24000"}, stream=PCMStream()))))
        parts = [chunk async for _, chunk in backend.synthesize("Hello")]
        self.assertTrue(all(len(chunk) % 2 == 0 for chunk in parts))
        self.assertEqual(b"".join(parts), b'\x01\x02\x03\x04')
        await backend.close()


class FakeSocket:
    def __init__(self):
        self.events = []

    async def send_json(self, event):
        self.events.append(event)


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_api_key_fails_promptly(self):
        cfg = Settings()
        backend = Backends(cfg)
        socket = FakeSocket()
        session = Session(socket, backend, cfg)
        session.history = [{"role": "user", "content": "hello"}]
        await asyncio.wait_for(session.respond(0), timeout=1)
        self.assertEqual(socket.events[-1]["type"], "turn.failed")
        await backend.close()

    async def test_cancel_stops_old_audio_and_cleans_stream(self):
        class SlowBackend:
            closed = False

            async def llm(self, messages):
                try:
                    yield "The first sentence! "
                    await asyncio.sleep(60)
                finally:
                    self.closed = True

            async def synthesize(self, text):
                yield 24000, b'\x00\x00' * 100
                await asyncio.sleep(60)

        socket, backend = FakeSocket(), SlowBackend()
        session = Session(socket, backend, Settings())
        session.reply_task = asyncio.create_task(session.respond(0))
        for _ in range(100):
            if any(e["type"] == "audio.chunk" for e in socket.events):
                break
            await asyncio.sleep(.002)
        self.assertTrue(any(e["type"] == "audio.chunk" for e in socket.events))
        await asyncio.wait_for(session.cancel(), timeout=1)
        self.assertTrue(backend.closed)
        self.assertEqual(session.generation, 1)
        self.assertEqual(socket.events[-1]["type"], "turn.cancelled")
        self.assertTrue(session.history)

    async def test_backchannel_does_not_cancel_current_reply(self):
        cfg = Settings(mock=True)
        backend = Backends(cfg)

        async def decide(*args):
            return Decision(False, False, "test")

        backend.decide = decide
        session = Session(FakeSocket(), backend, cfg)
        session.playing_turn = 0
        worker = asyncio.create_task(session.handle_inputs())
        await session.enqueue("Uh-huh")
        await asyncio.wait_for(session.inputs.join(), timeout=1)
        self.assertEqual(session.generation, 0)
        self.assertEqual(session.ws.events[-1]["type"], "input.ignored")
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await backend.close()


class GatewayTests(unittest.TestCase):
    def test_mock_text_audio_multiturn_reset(self):
        with TestClient(create_app(Settings(mock=True))) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertTrue(client.get("/ready").json()["ok"])
            with client.websocket_connect("/ws") as ws:
                self.assertEqual(ws.receive_json()["type"], "ready")
                for text in ["Hello", "Continue"]:
                    ws.send_json({"type": "text", "text": text})
                    events = []
                    for _ in range(100):
                        event = ws.receive_json()
                        events.append(event)
                        if event["type"] in {"turn.done", "turn.failed"}:
                            break
                    self.assertEqual(events[-1]["type"], "turn.done")
                    self.assertTrue(any(event["type"] == "audio.chunk" for event in events))
                ws.send_json({"type": "reset"})
                self.assertEqual(ws.receive_json()["type"], "turn.cancelled")
                ws.send_json({"type": "ping"})
                self.assertEqual(ws.receive_json()["type"], "pong")

    def test_microphone_pcm_reaches_asr_and_reply(self):
        with TestClient(create_app(Settings(mock=True))) as client:
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                pcm = pcm16(np.concatenate([np.full(8000, .12), np.zeros(16000)]))
                for pos in range(0, len(pcm), 3200):
                    ws.send_bytes(pcm[pos:pos + 3200])
                events = []
                for _ in range(100):
                    event = ws.receive_json()
                    events.append(event)
                    if event["type"] == "turn.done":
                        break
                self.assertTrue(any(event["type"] == "vad.end" for event in events))
                self.assertTrue(any(event["type"] == "transcript" for event in events))
                self.assertEqual(events[-1]["type"], "turn.done")

    def test_invalid_input_and_origin(self):
        with TestClient(create_app(Settings(mock=True))) as client:
            with self.assertRaises(Exception):
                with client.websocket_connect("/ws", headers={"Origin": "https://elsewhere.example"}):
                    pass
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                for raw in ["not JSON", "[]"]:
                    ws.send_text(raw)
                    self.assertEqual(ws.receive_json()["type"], "error")
                ws.send_bytes(b'\x00')
                self.assertEqual(ws.receive_json()["type"], "error")


class WorkerTests(unittest.TestCase):
    def test_asr_worker_audio_validation(self):
        with TestClient(asr_app(mock=True)) as client:
            self.assertTrue(client.get("/health").json()["ok"])
            response = client.post("/transcribe", content=wav_bytes(pcm16(np.full(1600, .1))))
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["text"])
            self.assertEqual(client.post("/transcribe", content=b'invalid').status_code, 400)
            self.assertEqual(client.post("/transcribe", content=b'x' * 1000000).status_code, 413)

    def test_tts_worker_pcm_headers_and_chunks(self):
        with TestClient(tts_app(mock=True)) as client:
            response = client.post("/synthesize", json={"text": "Hello"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(int(response.headers["X-Sample-Rate"]), 24000)
            self.assertEqual(len(response.content), 9600)
            self.assertEqual(client.post("/synthesize", json={"text": " "}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
