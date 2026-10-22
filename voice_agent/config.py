import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None):
    path = path or Path(os.getenv("VOICE_AGENT_ENV", str(Path.cwd() / ".env")))
    if not path.exists() and "VOICE_AGENT_ENV" not in os.environ:
        path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key.strip().isidentifier():
            raise ValueError(f"Invalid .env line: {key}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


@dataclass
class Settings:
    mock: bool = False
    asr_url: str = "http://127.0.0.1:8091"
    tts_url: str = "http://127.0.0.1:8092"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen-plus"
    llm_timeout: float = 60
    llm_max_tokens: int = 512
    system_prompt: str = "You are a voice assistant. Reply concisely in the same language as the user using natural speech and no Markdown."
    jev_enabled: bool = True
    jev_url: str = "http://127.0.0.1:8020"
    jev_timeout: float = 3
    jev_threshold: float = 0.65
    vad_threshold: float = 0.015
    vad_min_speech_ms: int = 240
    vad_silence_ms: int = 650
    vad_preroll_ms: int = 200
    vad_max_utterance_ms: int = 15000
    max_history_turns: int = 10
    max_sessions: int = 4

    @classmethod
    def from_env(cls):
        load_env()
        defaults = cls()
        values = {}
        for key in cls.__dataclass_fields__:
            raw = os.getenv("MOCK_MODE" if key == "mock" else key.upper())
            default = getattr(defaults, key)
            if raw is not None:
                values[key] = (raw.lower() in {"1", "true", "yes"} if isinstance(default, bool)
                               else type(default)(raw))
        obj = cls(**values)
        if not 0 <= obj.jev_threshold <= 1 or not 0 < obj.vad_threshold < 1:
            raise ValueError("Invalid JEV/VAD threshold")
        if min(obj.vad_min_speech_ms, obj.vad_silence_ms, obj.max_history_turns,
               obj.max_sessions, obj.vad_max_utterance_ms) <= 0 or obj.vad_preroll_ms < 0:
            raise ValueError("Durations and limits must be positive")
        return obj
