from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from .backends import Backends
from .config import ROOT, Settings
from .session import Session


def create_app(settings=None, backends=None):
    cfg = settings or Settings.from_env()
    models = backends or Backends(cfg)
    active = set()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await models.close()

    app = FastAPI(title="ASR LLM TTS Agent", lifespan=lifespan)
    web = ROOT / "voice_agent" / "web"
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/")
    async def index():
        return FileResponse(web / "index.html")

    @app.get("/health")
    async def health():
        return {"ok": True, "mock": cfg.mock, "active_sessions": len(active),
                "jev_enabled": cfg.jev_enabled}

    @app.get("/ready")
    async def ready():
        if cfg.mock:
            return {"ok": True, "mode": "mock"}
        statuses = {}
        for name, base in [("asr", cfg.asr_url), ("tts", cfg.tts_url)] + (
                [("jev", cfg.jev_url)] if cfg.jev_enabled else []):
            try:
                response = await models.http.get(base + "/health", timeout=3)
                response.raise_for_status()
                statuses[name] = response.json().get("ok", True)
            except (httpx.HTTPError, ValueError, AttributeError):
                statuses[name] = False
        statuses["llm_configured"] = bool(cfg.llm_api_key)
        return {"ok": all(statuses.values()), "services": statuses}

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        # Same-origin browser requests only. Non-browser clients may omit Origin.
        origin = ws.headers.get("origin")
        if origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc != ws.headers.get("host"):
                await ws.close(code=1008)
                return
        await ws.accept()
        if len(active) >= cfg.max_sessions:
            await ws.close(code=1013, reason="Too many sessions")
            return
        active.add(ws)
        try:
            await Session(ws, models, cfg).run()
        except WebSocketDisconnect:
            pass
        finally:
            active.discard(ws)

    return app
