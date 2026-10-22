import importlib.util
import sys

from .paths import DEFAULT_PATHS, project_path, use_source


def diagnose(cfg, component="gateway"):
    checks = {}
    dependencies = {
        "gateway": ["fastapi", "uvicorn", "httpx", "numpy"],
        "asr": ["fastapi", "uvicorn", "numpy", "torch", "transformers", "soundfile", "qwen_asr"],
        "tts": ["fastapi", "uvicorn", "numpy", "torch", "torchaudio", "onnxruntime", "hyperpyyaml",
                "modelscope", "librosa", "inflect", "omegaconf", "conformer", "diffusers", "wetext"],
        "jev": ["fastapi", "uvicorn", "torch", "transformers", "peft", "trainjev"],
    }
    if component in {"asr", "tts", "jev"}:
        try:
            use_source(component.upper() + "_REPO")
        except RuntimeError as exc:
            print(f"MISS {exc}")
            return 1
    print(f"Component: {component}; Python: {sys.executable}")
    for name in dependencies[component]:
        checks["dependency: " + name] = importlib.util.find_spec(name) is not None
    if component == "gateway":
        checks["WebSocket transport (websockets or wsproto)"] = any(
            importlib.util.find_spec(name) is not None for name in ("websockets", "wsproto"))
    if not cfg.mock:
        for key in DEFAULT_PATHS:
            if component == "gateway" or key.lower().startswith(component):
                if key.startswith("JEV_") and not cfg.jev_enabled:
                    continue
                checks[f"{key}: {project_path(key)}"] = project_path(key).exists()
        if component == "gateway":
            checks["LLM_API_KEY configured"] = bool(cfg.llm_api_key)
    for label, ok in checks.items():
        print(f"{'OK  ' if ok else 'MISS'} {label}")
    print("Check model dependencies in their separate ASR/TTS/JEV environments; see GET /ready for service readiness.")
    return 0 if all(checks.values()) else 1
