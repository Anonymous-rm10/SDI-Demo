"""Run bundled model commands in the current Python environment.

Example: python scripts/model.py train-jev -- python -m trainjev.train --help
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = {
    "qwen3-asr": ROOT / "third_party/Qwen3-ASR",
    "cosyvoice": ROOT / "third_party/CosyVoice",
    "train-jev": ROOT / "third_party/train-jev",
    "fireredasr2s": ROOT / "third_party/FireRedASR2S",
    "matcha-tts": ROOT / "third_party/CosyVoice/third_party/Matcha-TTS",
}


def prepare(component, command, workdir="."):
    base = COMPONENTS[component]
    cwd = (base / workdir).resolve()
    if not cwd.is_dir() or not cwd.is_relative_to(base):
        raise ValueError("Working directory must exist inside the selected component")
    if not command:
        raise ValueError("Supply a command after --")
    command = list(command)
    if command[0] in {"python", "python3"}:
        command[0] = sys.executable
    elif command[0] == "torchrun":
        command = [sys.executable, "-m", "torch.distributed.run", *command[1:]]
    env = dict(os.environ)
    paths = [str(base), *(str(path) for path in COMPONENTS.values() if path != base)]
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHON_BIN", sys.executable)
    env.setdefault("MODEL_PROJECT_ROOT", str(ROOT))
    env.setdefault("QWEN_PROJECT_DIR", str(COMPONENTS["qwen3-asr"]))
    return cwd, command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=".", help="Working directory relative to the component")
    parser.add_argument("--dry-run", action="store_true", help="Print command details without running it")
    parser.add_argument("component", choices=COMPONENTS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        cwd, command, env = prepare(args.component, command, args.workdir)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"cwd": str(cwd), "command": command, "PYTHONPATH": env["PYTHONPATH"]}, indent=2))
        return
    os.chdir(cwd)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
