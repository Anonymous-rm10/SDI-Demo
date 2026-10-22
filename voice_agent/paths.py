"""Resolve source and model locations relative to this source checkout."""
import os
import sys
from pathlib import Path

from .config import ROOT

DEFAULT_PATHS = {
    "ASR_REPO": "third_party/Qwen3-ASR",
    "ASR_MODEL": "models/Qwen3-ASR-0.6B",
    "TTS_REPO": "third_party/CosyVoice",
    "TTS_MODEL": "models/Fun-CosyVoice3-0.5B",
    "TTS_PROMPT_WAV": "third_party/CosyVoice/asset/zero_shot_prompt.wav",
    "JEV_REPO": "third_party/train-jev",
    "JEV_BASE": "models/Qwen3.5-4B",
    "JEV_ADAPTER": "models/train-jev-4b-lora",
}


def project_path(key):
    path = Path(os.getenv(key) or DEFAULT_PATHS[key]).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def use_source(key):
    repo = project_path(key)
    if not repo.is_dir():
        raise RuntimeError(f"{key} directory not found: {repo}")
    entries = [str(repo)]
    if key == "TTS_REPO":
        matcha = repo / "third_party/Matcha-TTS"
        if not (matcha / "matcha").is_dir():
            raise RuntimeError(f"Matcha-TTS source missing: {matcha}")
        entries.append(str(matcha))
    sys.path[:0] = entries
    return repo
