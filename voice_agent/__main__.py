import argparse
import os

from .config import Settings, load_env


def main():
    parser = argparse.ArgumentParser(description="ASR → LLM API → TTS with JEV decisions")
    parser.add_argument("service", choices=["gateway", "asr", "tts", "doctor"])
    parser.add_argument("--mock", action="store_true", help="No models/API; deterministic text and test tone")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--component", choices=["gateway", "asr", "tts", "jev"], default="gateway",
                        help="Component to inspect with doctor, in its own Python environment")
    args = parser.parse_args()
    load_env()
    cfg = Settings.from_env()
    if args.mock:
        cfg.mock = True
    if args.service == "doctor":
        from .doctor import diagnose
        raise SystemExit(diagnose(cfg, args.component))
    import uvicorn
    if args.service == "gateway":
        from .server import create_app
        app = create_app(cfg)
    else:
        from .workers import asr_app, tts_app
        app = (asr_app if args.service == "asr" else tts_app)(mock=cfg.mock)
    ports = {"gateway": 8090, "asr": 8091, "tts": 8092}
    port = args.port or int(os.getenv(args.service.upper() + "_PORT", ports[args.service]))
    uvicorn.run(app, host=args.host or os.getenv("HOST", "127.0.0.1"), port=port,
                ws_max_size=65536, ws_max_queue=16)


if __name__ == "__main__":
    main()
