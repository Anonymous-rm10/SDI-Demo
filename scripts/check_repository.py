"""Check vendored provenance and files Git would include. No external network."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_SOURCE_BYTES = 20 * 1024 * 1024
WEIGHT_SUFFIXES = {".pt", ".pth", ".safetensors", ".onnx", ".ckpt", ".engine", ".plan"}


def main():
    errors = []
    manifest = json.loads((ROOT / "third_party/manifest.json").read_text())
    verified = set()
    total = 0
    for component in manifest["components"]:
        base = ROOT / component["path"]
        if not (base / "LICENSE").is_file():
            errors.append(f"Missing upstream license: {component['path']}")
        for entry in component["files"]:
            path = base / entry["path"]
            relative = path.relative_to(ROOT).as_posix()
            verified.add(relative)
            if not path.is_file() or path.is_symlink():
                errors.append(f"Missing/nonregular vendored file: {relative}")
                continue
            data = path.read_bytes()
            total += len(data)
            if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["bytes"]:
                errors.append(f"Vendored file changed; review attribution and manifest: {relative}")
    for path in (ROOT / "third_party").rglob(".git"):
        errors.append(f"Nested Git metadata: {path.relative_to(ROOT)}")
    top = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    if top.returncode or Path(top.stdout.strip()).resolve() != ROOT:
        errors.append("Initialize this directory as the repository root: git init -b main")
        candidates = []
    else:
        result = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                check=True, capture_output=True)
        candidates = sorted(set(result.stdout.decode().strip("\0").split("\0")) - {""})
    candidate_set = set(candidates)
    for relative in candidates:
        path = ROOT / relative
        parts = Path(relative).parts
        if (path.name.startswith(".env") and path.name != ".env.example") or any(
                part in {".venv", "venv", "__pycache__", ".cache", ".git"} or part.startswith(".venv-") for part in parts):
            errors.append(f"Private/generated file would be published: {relative}")
        if path.suffix.lower() in WEIGHT_SUFFIXES or (parts[0] == "models" and relative != "models/README.md"):
            errors.append(f"Model artifact would be published: {relative}")
        if path.is_symlink():
            errors.append(f"Symlink must be reviewed before publishing: {relative}")
        elif path.is_file() and path.stat().st_size > MAX_SOURCE_BYTES:
            errors.append(f"Source file exceeds 20 MiB: {relative}")
    for relative in verified - candidate_set:
        errors.append(f"Vendored file is excluded from Git: {relative}")
    if errors:
        print("Repository check failed:")
        for error in errors:
            print("-", error)
        return 1
    print(f"Verified {len(verified)} vendored files ({total / 1024 / 1024:.2f} MiB).")
    print(f"Checked {len(candidates)} Git candidates; no nested Git metadata, weights or private environment files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
