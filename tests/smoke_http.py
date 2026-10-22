"""Real loopback HTTP/WebSocket smoke test; mock model workers and API fixtures.

Run: .venv/bin/python tests/smoke_http.py
No model weights, external network, API keys or persistent services needed.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import numpy as np
from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from voice_agent.audio import pcm16  # noqa: E402


class API(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/systemone":
            assert "questions" in payload
            data = {"answers": {"respond": {"noul": .95}, "interrupt": {"noul": .95}}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif self.path == "/v1/chat/completions":
            assert payload["stream"] and payload["messages"][-1]["role"] == "user"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for text in ["Hello! ", "This test reply was delivered over HTTP. "]:
                data = {"choices": [{"delta": {"content": text}}]}
                self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.send_error(404)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    api = ThreadingHTTPServer(("127.0.0.1", 0), API)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    api_url = f"http://127.0.0.1:{api.server_port}"
    ports = {name: free_port() for name in ["asr", "tts", "gateway"]}
    env = dict(os.environ, MOCK_MODE="false", JEV_ENABLED="true", LLM_API_KEY="smoke-test",
               LLM_BASE_URL=api_url + "/v1", JEV_URL=api_url,
               ASR_URL=f"http://127.0.0.1:{ports['asr']}", TTS_URL=f"http://127.0.0.1:{ports['tts']}")
    processes = []
    with tempfile.TemporaryDirectory(prefix="voice-agent-smoke-") as directory:
        logs = []
        try:
            for name, port in ports.items():
                output = open(Path(directory) / (name + ".log"), "w+")
                logs.append(output)
                args = [sys.executable, "-m", "voice_agent", name, "--host", "127.0.0.1", "--port", str(port)]
                if name != "gateway":
                    args.append("--mock")
                processes.append(subprocess.Popen(args, cwd=ROOT, env=env, stdout=output, stderr=output))
            with httpx.Client(trust_env=False, timeout=2) as client:
                for name, port in ports.items():
                    deadline = time.monotonic() + 20
                    while True:
                        if any(process.poll() is not None for process in processes):
                            raise RuntimeError("Service exited before readiness")
                        try:
                            if client.get(f"http://127.0.0.1:{port}/health").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError("Startup timeout: " + name)
                        time.sleep(.05)
                result = client.get(f"http://127.0.0.1:{ports['gateway']}/ready").json()
                assert result["ok"], result
                print("All worker and API health checks passed.")
            with connect(f"ws://127.0.0.1:{ports['gateway']}/ws", proxy=None) as ws:
                assert json.loads(ws.recv(timeout=10))["type"] == "ready"
                audio = pcm16(np.concatenate([np.full(8000, .1), np.zeros(16000)]))
                for offset in range(0, len(audio), 3200):
                    ws.send(audio[offset:offset + 3200])

                def receive_turn():
                    events = []
                    for _ in range(100):
                        event = json.loads(ws.recv(timeout=10))
                        events.append(event)
                        if event["type"] == "error":
                            raise RuntimeError(event)
                        if event["type"] == "turn.done":
                            break
                    assert events[-1]["type"] == "turn.done"
                    assert any(event["type"] == "audio.chunk" for event in events)
                    return events

                events = receive_turn()
                assert any(event["type"] == "transcript" for event in events)
                assert any(event["type"] == "decision" and event["source"] == "jev" for event in events)
                turn = events[-1]["turn_id"]
                ws.send(json.dumps({"type": "playback.start", "turn_id": turn}))
                ws.send(json.dumps({"type": "text", "text": "Please give a different answer."}))
                events = receive_turn()
                assert any(event["type"] == "turn.cancelled" and event["reason"] == "barge_in" for event in events)
                ws.send(json.dumps({"type": "reset"}))
                assert json.loads(ws.recv(timeout=10))["type"] == "turn.cancelled"
                print("PCM → ASR HTTP → JEV HTTP → LLM SSE → TTS HTTP → WebSocket audio passed.")
                print("Playback barge-in and session reset passed.")
        except Exception:
            for output in logs:
                output.flush()
                output.seek(0)
                print(output.read()[-6000:], file=sys.stderr)
            raise
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for output in logs:
                output.close()
            api.shutdown()
            api.server_close()


if __name__ == "__main__":
    main()
